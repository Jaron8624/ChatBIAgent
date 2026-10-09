/** API 客户端 — 通过 Vite 代理转发到后端 :8000 */

const BASE = '/api'

export interface ApiCtx {
  userId: number
  role: string
}

let ctx: ApiCtx = { userId: 1, role: 'guest' }

export function setApiContext(c: Partial<ApiCtx>) {
  ctx = { ...ctx, ...c }
}

export function getApiContext() {
  return ctx
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    'X-User-Id': String(ctx.userId),
    'X-User-Role': ctx.role,
    ...(init?.headers as Record<string, string>),
  }
  const res = await fetch(`${BASE}${path}`, { ...init, headers })
  if (!res.ok) {
    let detail = res.statusText
    try {
      const body = await res.json()
      detail = body.detail || JSON.stringify(body)
    } catch {
      /* ignore */
    }
    throw new Error(`${res.status}: ${detail}`)
  }
  return res.json() as Promise<T>
}

export const api = {
  get: <T>(path: string) => request<T>(path),
  post: <T>(path: string, body?: unknown) =>
    request<T>(path, {
      method: 'POST',
      body: body === undefined ? undefined : JSON.stringify(body),
    }),
  del: <T>(path: string) => request<T>(path, { method: 'DELETE' }),
}

/** SSE 流式对话：POST /chat/orchestrate/stream */
export interface StreamEvent {
  type: string
  content?: string
  intent?: string
  confidence?: number
  latency_ms?: number
  node?: string
  preview?: string
  tools_called?: number
  iterations?: number
  subtasks?: number
  steps?: number
  passed?: boolean
  error?: string
  spans?: Record<string, unknown>[]
  model_used?: string
  session_id?: string
  option?: Record<string, unknown>
}

export async function streamChat(
  body: {
    query: string
    user_id: number
    session_id: string
    knowledge_base_id: string
    messages: { role: string; content: string }[]
  },
  onEvent: (evt: StreamEvent) => void,
  signal?: AbortSignal,
) {
  const res = await fetch(`${BASE}/chat/orchestrate/stream`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-User-Id': String(ctx.userId),
      'X-User-Role': ctx.role,
    },
    body: JSON.stringify(body),
    signal,
  })
  if (!res.ok || !res.body) {
    throw new Error(`流式请求失败: ${res.status}`)
  }
  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buf = ''
  try {
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      buf += decoder.decode(value, { stream: true })
      const parts = buf.split('\n\n')
      buf = parts.pop() || ''
      for (const part of parts) {
        const line = part.split('\n').find((l) => l.startsWith('data: '))
        if (!line) continue
        try {
          onEvent(JSON.parse(line.slice(6)))
        } catch {
          /* ignore malformed */
        }
      }
    }
    // flush remaining buffer (handle case where server doesn't end with \n\n)
    buf += decoder.decode()
    if (buf.trim()) {
      const line = buf.split('\n').find((l) => l.startsWith('data: '))
      if (line) {
        try {
          onEvent(JSON.parse(line.slice(6)))
        } catch {
          /* ignore malformed */
        }
      }
    }
  } finally {
    reader.cancel().catch(() => {})
  }
}
