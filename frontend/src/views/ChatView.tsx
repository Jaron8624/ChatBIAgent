import { useEffect, useRef, useState } from 'react'
import { api, getApiContext, streamChat, type StreamEvent } from '../lib/api'
import * as echarts from 'echarts'
import { TraceTimeline as TraceView } from '../components/TraceTimeline'
import type { TraceSpan } from '../lib/traceTimeline'

interface ToolCall {
  tool: string
  args: Record<string, unknown>
}

interface ReactMeta {
  iterations: number
  tool_calls?: ToolCall[]
  forced_final?: boolean
  trace?: { round: number; thought?: string; action?: string; observation?: string }[]
}

interface ChatMsg {
  role: 'user' | 'assistant'
  content: string
  intent?: string
  confidence?: number
  requiresHuman?: boolean
  latencyMs?: number
  reactMeta?: ReactMeta
  mcpResults?: string[]
  ragResult?: string
  feedback?: 'up' | 'down'
  streaming?: boolean
  traceSpans?: TraceSpan[]
  progressNodes?: { node: string; status: 'start' | 'complete' | 'error'; latencyMs?: number }[]
  chartOption?: Record<string, unknown>
}

interface SessionResumePayload {
  session_id: string
  messages: { role: string; content: string }[]
}

const HELP_CARDS = [
  { title: '趋势分析', quote: '最近三个月各品类的销售额是多少？' },
  { title: '对比分析', quote: '11月和2月的订单量对比，为什么差异这么大？' },
  { title: '根因分析', quote: '家具类目差评率为什么比其他品类高？' },
  { title: '跨库对比', quote: '各州订单量 vs Chinook 各国销售额，帮我对比两个数据集' },
]

function greeting() {
  const h = new Date().getHours()
  if (h < 6) return '夜深了'
  if (h < 12) return '上午好'
  if (h < 18) return '下午好'
  return '晚上好'
}

/** 实时节点进度条 */
function NodeProgress({ nodes }: { nodes: { node: string; status: 'start' | 'complete' | 'error'; latencyMs?: number }[] }) {
  if (!nodes.length) return null
  const nodeNames: Record<string, string> = {
    supervisor: '意图识别',
    retrieve_all: '并行检索',
    rag_agent: '知识检索',
    memory_agent: '记忆检索',
    mcp_tools: '工具调用',
    react_loop: 'ReAct 迭代',
    coordinate: '协调执行',
    plan: '任务规划',
    explore: '探索检索',
    synthesize: '汇总生成',
    reflection: '质量审核',
    direct: '直接回答',
  }
  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 8 }}>
      {nodes.map((n, i) => {
        const dotColor = n.status === 'error' ? '#dc2626' : n.status === 'complete' ? '#16a34a' : '#8a8a8a'
        return (
          <div
            key={i}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 5,
              padding: '2px 10px',
              borderRadius: 12,
              fontSize: 11,
              background: '#f0f1f3',
              color: '#4a4a4a',
              border: '1px solid #e2e4e8',
            }}
          >
            <span style={{
              width: 6,
              height: 6,
              borderRadius: '50%',
              background: dotColor,
              display: 'inline-block',
              flexShrink: 0,
            }} />
            <span>{nodeNames[n.node] || n.node}</span>
            {n.latencyMs !== undefined && (
              <span style={{ color: '#8a8a8a', fontSize: 10 }}>{n.latencyMs.toFixed(0)}ms</span>
            )}
          </div>
        )
      })}
    </div>
  )
}

/** ECharts 图表渲染组件 */
function ChartView({ option }: { option: Record<string, unknown> }) {
  const chartRef = useRef<HTMLDivElement>(null)
  const chartInstance = useRef<echarts.ECharts | null>(null)

  useEffect(() => {
    if (!chartRef.current) return
    if (!chartInstance.current) {
      chartInstance.current = echarts.init(chartRef.current)
    }
    chartInstance.current.setOption(option, true)

    const handleResize = () => chartInstance.current?.resize()
    window.addEventListener('resize', handleResize)
    return () => window.removeEventListener('resize', handleResize)
  }, [option])

  useEffect(() => {
    return () => {
      chartInstance.current?.dispose()
      chartInstance.current = null
    }
  }, [])

  return (
    <div
      ref={chartRef}
      style={{
        marginTop: 10,
        width: '100%',
        height: 320,
        border: '1px solid var(--border)',
        borderRadius: 6,
        background: '#fff',
      }}
    />
  )
}

/** ReAct 执行轨迹折叠面板 */
function ReactTrace({ meta }: { meta: ReactMeta }) {
  const [open, setOpen] = useState(false)
  const [openStep, setOpenStep] = useState<number | null>(null)
  return (
    <div
      style={{
        marginTop: 10,
        border: '1px solid var(--border)',
        background: '#f7f8fa',
        borderRadius: 6,
        padding: '8px 12px',
      }}
    >
      <div
        style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer' }}
        onClick={() => setOpen(!open)}
      >
        <span style={{ color: 'var(--text-1)', fontSize: 12, fontWeight: 500 }}>▶ ReAct 轨迹</span>
        <span style={{ fontSize: 11, color: 'var(--text-2)', background: '#fff', padding: '1px 6px', borderRadius: 4, border: '1px solid var(--border)' }}>
          {meta.iterations} 轮迭代
        </span>
        {meta.tool_calls && (
          <span style={{ fontSize: 11, color: 'var(--text-2)', background: '#fff', padding: '1px 6px', borderRadius: 4, border: '1px solid var(--border)' }}>
            {meta.tool_calls.length} 次工具调用
          </span>
        )}
        {meta.forced_final && (
          <span style={{ fontSize: 11, color: '#ca8a04', background: '#fff', padding: '1px 6px', borderRadius: 4, border: '1px solid rgba(202,138,4,0.3)' }}>
            强制摘要兜底
          </span>
        )}
        <span style={{ marginLeft: 'auto', color: 'var(--text-2)', fontSize: 11 }}>
          {open ? '收起 ▴' : '展开 ▾'}
        </span>
      </div>
      {open && (
        <div style={{ marginTop: 10 }}>
          {meta.tool_calls?.map((c, i) => (
            <div
              key={i}
              style={{
                fontFamily: 'var(--mono)',
                fontSize: 11.5,
                color: 'var(--text-1)',
                padding: '4px 0',
                borderBottom: '1px dashed var(--border)',
              }}
            >
              <span style={{ color: 'var(--green)' }}>✓</span>{' '}
              <span style={{ color: 'var(--accent)' }}>{c.tool}</span>(
              {JSON.stringify(c.args)})
            </div>
          ))}
          {meta.trace?.map((t, i) => (
            <div key={i} style={{ marginTop: 6 }}>
              <div
                style={{ cursor: 'pointer', fontSize: 11.5, color: 'var(--text-2)' }}
                onClick={() => setOpenStep(openStep === i ? null : i)}
              >
                Round {t.round} · {t.action || ''} {openStep === i ? '▴' : '▾'}
              </div>
              {openStep === i && (
                <div style={{ paddingLeft: 12, marginTop: 4 }}>
                  {t.thought && (
                    <div style={{ fontSize: 11.5, color: 'var(--text-2)', marginBottom: 4 }}>
                      💭 {t.thought}
                    </div>
                  )}
                  {t.observation && (
                    <pre className="code-block" style={{ maxHeight: 160, overflow: 'auto' }}>
                      {t.observation}
                    </pre>
                  )}
                </div>
              )}
            </div>
          ))}
          {!meta.trace && !meta.tool_calls?.length && (
            <div style={{ fontSize: 11.5, color: 'var(--text-2)' }}>无详细轨迹数据</div>
          )}
        </div>
      )}
    </div>
  )
}

function MessageBubble({ msg, onFeedback }: { msg: ChatMsg; onFeedback?: (satisfied: boolean) => void }) {
  const isUser = msg.role === 'user'
  return (
    <div style={{ display: 'flex', gap: 12, marginBottom: 20 }}>
      {/* 头像 — 纯色无渐变 */}
      <div
        style={{
          width: 28,
          height: 28,
          borderRadius: 6,
          flexShrink: 0,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontSize: 12,
          fontWeight: 700,
          color: '#fff',
          background: isUser ? '#8a8a8a' : 'var(--accent)',
        }}
      >
        {isUser ? 'U' : 'E'}
      </div>
      <div style={{ flex: 1, minWidth: 0 }}>
        {/* 头部信息行 */}
        <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 6 }}>
          <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-0)' }}>
            {isUser ? '用户' : 'NexusMind Agent'}
          </span>
          {msg.intent && (
            <span style={{ fontSize: 11, color: 'var(--text-2)', background: 'var(--bg-2)', padding: '1px 6px', borderRadius: 4 }}>
              {msg.intent}
            </span>
          )}
          {msg.requiresHuman && (
            <span style={{ fontSize: 11, color: 'var(--red)', background: 'rgba(220,38,38,0.06)', padding: '1px 6px', borderRadius: 4 }}>
              已升级人工
            </span>
          )}
          {msg.latencyMs !== undefined && (
            <span style={{ fontSize: 11, color: 'var(--text-2)', marginLeft: 'auto' }}>
              {msg.latencyMs.toFixed(0)} ms
            </span>
          )}
        </div>

        {/* 消息内容 — 白色背景浅灰边框 */}
        <div
          style={{
            background: '#fff',
            border: '1px solid var(--border)',
            borderRadius: 6,
            padding: '12px 14px',
            fontSize: 13.5,
            lineHeight: 1.7,
            color: 'var(--text-0)',
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-word',
          }}
        >
          {msg.content || (msg.streaming ? '▍' : '')}
          {msg.streaming && msg.content && <span className="blink">▍</span>}
        </div>

        {msg.reactMeta && <ReactTrace meta={msg.reactMeta} />}
        {msg.chartOption && <ChartView option={msg.chartOption} />}
        {msg.traceSpans && <TraceView spans={msg.traceSpans} />}
        {msg.progressNodes && msg.progressNodes.length > 0 && <NodeProgress nodes={msg.progressNodes} />}

        {/* 反馈 — 更简洁 */}
        {!isUser && !msg.streaming && msg.content && (
          <div style={{ display: 'flex', gap: 6, marginTop: 8, alignItems: 'center' }}>
            <button
              className="btn small"
              style={{
                padding: '2px 8px',
                fontSize: 11,
                opacity: msg.feedback ? 1 : 0.5,
                ...(msg.feedback === 'up' ? { borderColor: 'var(--green)', color: 'var(--green)' } : {}),
              }}
              onClick={() => onFeedback?.(true)}
            >
              👍
            </button>
            <button
              className="btn small"
              style={{
                padding: '2px 8px',
                fontSize: 11,
                opacity: msg.feedback ? 1 : 0.5,
                ...(msg.feedback === 'down' ? { borderColor: 'var(--red)', color: 'var(--red)' } : {}),
              }}
              onClick={() => onFeedback?.(false)}
            >
              👎
            </button>
            {msg.feedback && (
              <span style={{ fontSize: 11, color: 'var(--text-2)' }}>已反馈</span>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

export default function ChatView({
  initialSession,
  onInitialSessionLoaded,
}: {
  initialSession?: SessionResumePayload | null
  onInitialSessionLoaded?: () => void
} = {}) {
  const [messages, setMessages] = useState<ChatMsg[]>(() =>
    (initialSession?.messages || [])
      .filter((message) => message.role === 'user' || message.role === 'assistant')
      .map((message) => ({ role: message.role as ChatMsg['role'], content: message.content })),
  )
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [sessionId, setSessionId] = useState(() =>
    initialSession?.session_id || localStorage.getItem('nexusmind.activeSessionId') || `web-${Date.now()}`,
  )
  const [kbList, setKbList] = useState<{ knowledge_bases: { knowledge_base_id: string }[] } | null>(null)
  const [kbId, setKbId] = useState('默认知识库')
  const bottomRef = useRef<HTMLDivElement>(null)
  const abortRef = useRef<AbortController | null>(null)

  useEffect(() => {
    api
      .get<{ knowledge_bases: { knowledge_base_id: string }[] }>(
        `/knowledge-bases?user_id=${getApiContext().userId}`,
      )
      .then((r) => {
        setKbList(r)
        if (r.knowledge_bases?.length) setKbId(r.knowledge_bases[0].knowledge_base_id)
      })
      .catch(() => {})
  }, [])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  useEffect(() => {
    if (initialSession) {
      setSessionId(initialSession.session_id)
      setMessages(
        initialSession.messages
          .filter((message) => message.role === 'user' || message.role === 'assistant')
          .map((message) => ({ role: message.role as ChatMsg['role'], content: message.content })),
      )
      localStorage.setItem('nexusmind.activeSessionId', initialSession.session_id)
      onInitialSessionLoaded?.()
      return
    }
    if (busy || sessionId.startsWith('web-')) return
    let current = true
    api
      .post<SessionResumePayload>('/session/resume', { session_id: sessionId })
      .then((session) => {
        if (!current) return
        setMessages(
          session.messages
            .filter((message) => message.role === 'user' || message.role === 'assistant')
            .map((message) => ({ role: message.role as ChatMsg['role'], content: message.content })),
        )
      })
      .catch(() => {
        if (!current) return
        localStorage.removeItem('nexusmind.activeSessionId')
        setSessionId(`web-${Date.now()}`)
        setMessages([])
      })
    return () => {
      current = false
    }
  }, [initialSession, onInitialSessionLoaded, busy, sessionId])

  const send = async (text?: string) => {
    const query = (text ?? input).trim()
    if (!query || busy) return
    setInput('')
    setBusy(true)

    const history = messages.map((m) => ({ role: m.role, content: m.content }))
    setMessages((prev) => [...prev, { role: 'user', content: query }])

    const assistant: ChatMsg = { role: 'assistant', content: '', streaming: true, progressNodes: [] }
    setMessages((prev) => [...prev, assistant])

    let answer = ''
    let intent = ''
    let confidence = 0
    let latency = 0
    let lastMcp: string[] = []
    let lastRag = ''
    let finalRequiresHuman = false
    let finalReact: ReactMeta | undefined = undefined
    let traceSpans: TraceSpan[] = []
    let flushTimer: ReturnType<typeof setTimeout> | null = null
    const progressNodes: ChatMsg['progressNodes'] = []
    const abortController = new AbortController()
    abortRef.current = abortController

    try {
      await streamChat(
        {
          query,
          user_id: getApiContext().userId,
          session_id: sessionId,
          knowledge_base_id: kbId,
          messages: history,
        },
        (evt: StreamEvent) => {
          if (evt.type === 'intent') {
            intent = evt.intent || ''
            confidence = evt.confidence || 0
          }
          if (evt.type === 'node_start') {
            progressNodes.push({ node: evt.node || '', status: 'start' })
            setMessages((prev) => {
              const copy = [...prev]
              const last = copy[copy.length - 1]
              copy[copy.length - 1] = { ...last, progressNodes: [...progressNodes] }
              return copy
            })
          }
          if (evt.type === 'node_complete') {
            const idx = progressNodes.findIndex((n) => n.node === evt.node && n.status === 'start')
            if (idx >= 0) {
              progressNodes[idx] = { node: evt.node || '', status: evt.error ? 'error' : 'complete', latencyMs: evt.latency_ms }
            } else {
              progressNodes.push({ node: evt.node || '', status: evt.error ? 'error' : 'complete', latencyMs: evt.latency_ms })
            }
            setMessages((prev) => {
              const copy = [...prev]
              const last = copy[copy.length - 1]
              copy[copy.length - 1] = { ...last, progressNodes: [...progressNodes] }
              return copy
            })
          }
          if (evt.type === 'answer') {
            answer += evt.content || ''
            // throttle: 每 30ms 批量刷新一次，避免 React batching 导致文字"突然出现"
            if (!flushTimer) {
              flushTimer = window.setTimeout(() => {
                setMessages((prev) => {
                  const copy = [...prev]
                  copy[copy.length - 1] = { ...copy[copy.length - 1], content: answer }
                  return copy
                })
                flushTimer = null
              }, 30)
            }
          }
          if (evt.type === 'chart' && evt.option) {
            setMessages((prev) => {
              const copy = [...prev]
              copy[copy.length - 1] = { ...copy[copy.length - 1], chartOption: evt.option }
              return copy
            })
          }
          if (evt.type === 'trace' && evt.spans) {
            traceSpans = (evt.spans as unknown) as TraceSpan[]
          }
          if (evt.type === 'session' && evt.session_id) {
            setSessionId(evt.session_id)
            localStorage.setItem('nexusmind.activeSessionId', evt.session_id)
          }
          if (evt.type === 'finished') {
            latency = evt.latency_ms || 0
            if (evt.session_id) {
              setSessionId(evt.session_id)
              localStorage.setItem('nexusmind.activeSessionId', evt.session_id)
            }
          }
          if (evt.type === 'error') {
            answer += `\n❌ ${evt.content || '未知错误'}`
          }
        },
        abortController.signal,
      )
    } catch (e) {
      if ((e as Error).name === 'AbortError') {
        // 用户主动取消，不追加错误信息
      } else {
        answer = `❌ 请求失败：${(e as Error).message}`
      }
    }

    // 清理 flushTimer 避免泄漏
    if (flushTimer) {
      clearTimeout(flushTimer)
      flushTimer = null
    }

    setMessages((prev) => {
      const copy = [...prev]
      copy[copy.length - 1] = {
        ...copy[copy.length - 1],
        content: answer,
        streaming: false,
        intent,
        confidence,
        latencyMs: latency,
        requiresHuman: finalRequiresHuman,
        reactMeta: finalReact,
        mcpResults: lastMcp,
        ragResult: lastRag,
        traceSpans,
        progressNodes: [...progressNodes],
      }
      return copy
    })

    setBusy(false)
  }

  const feedback = async (idx: number, satisfied: boolean) => {
    setMessages((prev) => {
      const copy = [...prev]
      copy[idx] = { ...copy[idx], feedback: satisfied ? 'up' : 'down' }
      return copy
    })
    try {
      await api.post('/monitor/feedback', { session_id: sessionId, satisfied })
    } catch {
      /* ignore */
    }
  }

  // ======== AIDE 风格输入卡片（空态/非空态共用） ========
  const inputCard = (
    <div
      style={{
        width: '100%',
        maxWidth: 780,
        margin: '0 auto',
        background: '#fff',
        border: '1px solid var(--border)',
        borderRadius: 14,
        boxShadow: '0 2px 12px rgba(0,0,0,0.06)',
        overflow: 'hidden',
      }}
    >
      <textarea
        style={{
          width: '100%',
          border: 'none',
          outline: 'none',
          resize: 'none',
          padding: '16px 18px 8px',
          fontSize: 14,
          lineHeight: 1.6,
          minHeight: 56,
          maxHeight: 160,
          fontFamily: 'var(--sans)',
          color: 'var(--text-0)',
          background: 'transparent',
        }}
        placeholder="请提出问题或需求，⇧+⏎ 换行，⏎ 发送"
        value={input}
        onChange={(e) => setInput(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault()
            send()
          }
        }}
      />
      {/* 工具条 */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 10,
          padding: '8px 14px 12px',
        }}
      >
        {/* 新会话 */}
        <button
          className="btn"
          title="新会话"
          style={{
            padding: '4px 10px',
            fontSize: 14,
            background: '#fff',
            border: '1px solid var(--border)',
            borderRadius: 8,
          }}
          onClick={() => {
            abortRef.current?.abort()
            abortRef.current = null
            setMessages([])
            setSessionId(`web-${Date.now()}`)
            localStorage.removeItem('nexusmind.activeSessionId')
            setBusy(false)
          }}
        >
          ＋
        </button>
        {/* 知识库选择 */}
        <select
          className="select"
          title="知识库"
          style={{ padding: '4px 8px', fontSize: 12, background: '#fff', border: '1px solid var(--border)', borderRadius: 8 }}
          value={kbId}
          onChange={(e) => setKbId(e.target.value)}
        >
          <option value="默认知识库">默认知识库</option>
          {kbList?.knowledge_bases?.map((kb) => (
            <option key={kb.knowledge_base_id} value={kb.knowledge_base_id}>
              {kb.knowledge_base_id}
            </option>
          ))}
        </select>
        {/* Agent 标识 + 连接状态 */}
        <span style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 13, fontWeight: 600, color: 'var(--text-0)' }}>
          <span
            style={{
              width: 18,
              height: 18,
              borderRadius: 5,
              background: 'linear-gradient(135deg, var(--accent) 0%, var(--purple) 100%)',
              color: '#fff',
              fontSize: 11,
              fontWeight: 700,
              display: 'inline-flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            N
          </span>
          NexusMind
          <span
            style={{
              width: 7,
              height: 7,
              borderRadius: '50%',
              background: 'var(--green)',
              display: 'inline-block',
            }}
          />
        </span>
        <span style={{ flex: 1 }} />
        {/* 发送 */}
        <button
          title="发送"
          disabled={busy || !input.trim()}
          onClick={() => send()}
          style={{
            width: 32,
            height: 32,
            borderRadius: 8,
            border: 'none',
            background: busy || !input.trim() ? 'var(--bg-3)' : 'var(--text-0)',
            color: busy || !input.trim() ? 'var(--text-2)' : '#fff',
            fontSize: 16,
            cursor: busy || !input.trim() ? 'not-allowed' : 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          ↑
        </button>
      </div>
    </div>
  )

  // ======== 空态：AIDE 首页 ========
  if (messages.length === 0) {
    return (
      <div
        style={{
          flex: 1,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          padding: '0 24px',
          background: '#fff',
          minHeight: 0,
          overflowY: 'auto',
        }}
      >
        <div style={{ fontSize: 30, fontWeight: 500, color: 'var(--text-0)', marginBottom: 36, letterSpacing: 0.5 }}>
          {greeting()}，需要我做什么吗？
        </div>
        {inputCard}
        {/* 我可以帮你 */}
        <div style={{ width: '100%', maxWidth: 780, margin: '28px auto 0' }}>
          <div style={{ display: 'flex', alignItems: 'center', marginBottom: 12 }}>
            <span style={{ fontSize: 12.5, color: 'var(--text-2)' }}>我可以帮你</span>
            <span style={{ flex: 1 }} />
            <span style={{ fontSize: 12, color: 'var(--text-2)', letterSpacing: 2 }}>‹ ›</span>
          </div>
          <div style={{ display: 'flex', gap: 12 }}>
            {HELP_CARDS.map((c) => (
              <div
                key={c.title}
                onClick={() => send(c.quote)}
                style={{
                  flex: 1,
                  minWidth: 0,
                  background: '#fff',
                  border: '1px solid var(--border)',
                  borderRadius: 10,
                  padding: '12px 14px',
                  cursor: 'pointer',
                  transition: 'border-color 0.12s, box-shadow 0.12s',
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.borderColor = 'var(--border-light)'
                  e.currentTarget.style.boxShadow = '0 2px 8px rgba(0,0,0,0.05)'
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.borderColor = 'var(--border)'
                  e.currentTarget.style.boxShadow = 'none'
                }}
              >
                <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-0)', marginBottom: 6 }}>
                  {c.title}
                </div>
                <div
                  style={{
                    fontSize: 12,
                    color: 'var(--text-2)',
                    lineHeight: 1.5,
                    display: '-webkit-box',
                    WebkitLineClamp: 2,
                    WebkitBoxOrient: 'vertical',
                    overflow: 'hidden',
                  }}
                >
                  “{c.quote}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    )
  }

  // ======== 非空态：消息列表 + 底部输入卡片 ========
  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', background: '#fff' }}>
      <div style={{ flex: 1, overflowY: 'auto', minHeight: 0, padding: '20px 24px' }}>
        <div style={{ maxWidth: 860, margin: '0 auto' }}>
          {messages.map((m, i) => (
            <MessageBubble key={i} msg={m} onFeedback={(sat) => feedback(i, sat)} />
          ))}
          <div ref={bottomRef} />
        </div>
      </div>
      <div style={{ padding: '0 24px 18px' }}>{inputCard}</div>
    </div>
  )
}
