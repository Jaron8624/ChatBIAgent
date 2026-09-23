import { useEffect, useState } from 'react'
import { api } from '../lib/api'
import { Card, Empty, Loading, Tag } from '../components/ui'

interface McpTool {
  name: string
  description?: string
  inputSchema?: { properties?: Record<string, { type?: string; description?: string }>; required?: string[] }
  category?: string
}

const CATEGORY_COLOR: Record<string, 'blue' | 'green' | 'yellow' | 'purple' | 'gray' | 'red'> = {
  order: 'blue',
  refund: 'red',
  delivery: 'green',
  coupon: 'yellow',
  complaint: 'yellow',
  memory: 'purple',
  context: 'purple',
  knowledge: 'purple',
}

function ToolCard({ tool, onTry }: { tool: McpTool; onTry: (t: McpTool) => void }) {
  const props = tool.inputSchema?.properties || {}
  return (
    <div
      style={{
        border: '1px solid var(--border)',
        borderRadius: 8,
        padding: '12px 14px',
        background: 'var(--bg-2)',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
        <code style={{ color: 'var(--accent)', fontSize: 12.5, fontWeight: 600 }}>{tool.name}</code>
        {tool.category && <Tag color={CATEGORY_COLOR[tool.category] || 'gray'}>{tool.category}</Tag>}
        <button className="btn small" style={{ marginLeft: 'auto' }} onClick={() => onTry(tool)}>
          调用
        </button>
      </div>
      <div style={{ fontSize: 12, color: 'var(--text-1)', marginBottom: 8, lineHeight: 1.5 }}>
        {tool.description || '—'}
      </div>
      {Object.keys(props).length > 0 && (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 5 }}>
          {Object.entries(props).map(([k, v]) => (
            <span
              key={k}
              style={{
                fontFamily: 'var(--mono)',
                fontSize: 10.5,
                padding: '2px 7px',
                borderRadius: 5,
                background: 'var(--bg-0)',
                border: '1px solid var(--border)',
                color: 'var(--text-2)',
              }}
            >
              {k}
              <span style={{ opacity: 0.6 }}>:{v.type || 'any'}</span>
              {tool.inputSchema?.required?.includes(k) && (
                <span style={{ color: 'var(--red)' }}>*</span>
              )}
            </span>
          ))}
        </div>
      )}
    </div>
  )
}

export default function McpView() {
  const [tools, setTools] = useState<McpTool[] | null>(null)
  const [error, setError] = useState('')
  const [servers, setServers] = useState<{ server_count: number; tools: unknown[] } | null>(null)
  const [selected, setSelected] = useState<McpTool | null>(null)
  const [argsText, setArgsText] = useState('{}')
  const [result, setResult] = useState('')
  const [calling, setCalling] = useState(false)

  const load = () => {
    api
      .get<{ tools: McpTool[] }>('/mcp/tools')
      .then((r) => { setTools(r.tools || []); setError('') })
      .catch((e) => setError((e as Error).message))
    api
      .get<{ server_count: number; tools: unknown[] }>('/mcp/servers')
      .then(setServers)
      .catch(() => {})
  }

  useEffect(load, [])

  const tryTool = (t: McpTool) => {
    setSelected(t)
    const props = t.inputSchema?.properties || {}
    const defaults: Record<string, unknown> = {}
    for (const [k, v] of Object.entries(props)) {
      if (v.type === 'string') defaults[k] = ''
      else if (v.type === 'integer' || v.type === 'number') defaults[k] = 0
      else if (v.type === 'boolean') defaults[k] = false
      else if (v.type === 'array') defaults[k] = []
      else defaults[k] = null
    }
    setArgsText(JSON.stringify(defaults, null, 2))
    setResult('')
  }

  const call = async () => {
    if (!selected) return
    setCalling(true)
    setResult('')
    try {
      const args = JSON.parse(argsText)
      const r = await api.post<{ result: unknown; error?: unknown }>('/mcp/tools/call', {
        name: selected.name,
        arguments: args,
      })
      setResult(JSON.stringify(r, null, 2))
    } catch (e) {
      setResult(`❌ ${(e as Error).message}`)
    }
    setCalling(false)
  }

  if (error) return <div className="error-text">加载失败：{error}</div>
  if (!tools) return <Loading text="加载 MCP 工具" />

  return (
    <div>
      <div className="stats-grid">
        <div className="stat-card">
          <div className="label">注册工具数</div>
          <div className="value blue">{tools.length}</div>
        </div>
        <div className="stat-card">
          <div className="label">外部 MCP Server</div>
          <div className="value">{servers?.server_count ?? '—'}</div>
          <div className="sub">stdio / HTTP 双模式</div>
        </div>
        <div className="stat-card">
          <div className="label">协议</div>
          <div className="value" style={{ fontSize: 15 }}>
            JSON-RPC 2.0
          </div>
          <div className="sub">自研 MCP 协议层</div>
        </div>
      </div>

      {selected && (
        <Card
          title={`调用 ${selected.name}`}
          hint="JSON-RPC tools/call"
          actions={
            <button className="btn small" onClick={() => setSelected(null)}>
              关闭
            </button>
          }
        >
          <textarea
            className="textarea mono"
            style={{ minHeight: 120 }}
            value={argsText}
            onChange={(e) => setArgsText(e.target.value)}
          />
          <div style={{ marginTop: 10, display: 'flex', gap: 8 }}>
            <button className="btn primary" disabled={calling} onClick={call}>
              {calling ? '调用中…' : '执行调用'}
            </button>
          </div>
          {result && <pre className="code-block" style={{ marginTop: 12, maxHeight: 320, overflow: 'auto' }}>{result}</pre>}
        </Card>
      )}

      <Card title="工具注册表" hint={`${tools.length} 个工具`}>
        {tools.length === 0 ? (
          <Empty />
        ) : (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))', gap: 12 }}>
            {tools.map((t) => (
              <ToolCard key={t.name} tool={t} onTry={tryTool} />
            ))}
          </div>
        )}
      </Card>

      <Card title="外部 MCP Server" hint="通过 MCP_POOL 环境变量配置">
        {servers ? (
          <div style={{ fontSize: 13, color: 'var(--text-1)' }}>
            已注册 <b style={{ color: 'var(--accent)' }}>{servers.server_count}</b> 个外部 Server，
            合并工具 {Array.isArray(servers.tools) ? servers.tools.length : 0} 个。
            <button
              className="btn small"
              style={{ marginLeft: 12 }}
              onClick={async () => {
                try {
                  await api.post('/mcp/servers/connect')
                  load()
                } catch (e) {
                  alert((e as Error).message)
                }
              }}
            >
              连接全部
            </button>
          </div>
        ) : (
          <Empty text="暂无外部 Server（未配置 MCP_POOL）" />
        )}
      </Card>
    </div>
  )
}
