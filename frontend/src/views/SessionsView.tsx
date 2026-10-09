import { useEffect, useState } from 'react'
import { api } from '../lib/api'
import { Card, Empty, Loading, Tag, fmtTime } from '../components/ui'

interface SessionItem {
  session_id: string
  summary: string
  token_count: number
  updated_at: string
  message_count: number
  run_status?: string
  recoverable?: boolean
}

interface ResumedMsg {
  role: string
  content: string
}

export default function SessionsView({
  onContinue,
}: {
  onContinue: (session: { session_id: string; messages: ResumedMsg[] }) => void
}) {
  const [sessions, setSessions] = useState<SessionItem[] | null>(null)
  const [error, setError] = useState('')
  const [resumed, setResumed] = useState<{ session_id: string; messages: ResumedMsg[]; summary: string } | null>(null)

  const load = () => {
    api
      .get<{ sessions: SessionItem[] }>('/session/list?limit=30')
      .then((r) => setSessions(r.sessions || []))
      .catch((e) => setError((e as Error).message))
  }

  useEffect(load, [])

  const resume = async (sid: string) => {
    try {
      const r = await api.post<{ session_id: string; messages: ResumedMsg[]; summary: string }>(
        '/session/resume',
        { session_id: sid },
      )
      setResumed(r)
    } catch (e) {
      alert((e as Error).message)
    }
  }

  const remove = async (sid: string) => {
    if (!confirm(`确定删除会话 ${sid}？`)) return
    try {
      await api.del(`/session/${encodeURIComponent(sid)}`)
      if (resumed?.session_id === sid) setResumed(null)
      load()
    } catch (e) {
      alert((e as Error).message)
    }
  }

  const recover = async (sid: string) => {
    try {
      const result = await api.post<{ final_response: string }>(
        '/session/recover',
        { session_id: sid },
      )
      setResumed((previous) => {
        if (!previous || previous.session_id !== sid) return previous
        const last = previous.messages[previous.messages.length - 1]
        if (last?.role === 'assistant' && last.content === result.final_response) return previous
        return {
          ...previous,
          messages: [...previous.messages, { role: 'assistant', content: result.final_response }],
        }
      })
      load()
    } catch (e) {
      alert((e as Error).message)
    }
  }

  if (error) return <div className="error-text">加载失败：{error}</div>
  if (!sessions) return <Loading text="加载会话列表" />

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: 16, alignItems: 'start' }}>
      <Card
        title="会话列表"
        hint="Redis zlib 压缩存储 · 多会话索引"
        actions={
          <button className="btn small" onClick={load}>
            刷新
          </button>
        }
      >
        {sessions.length === 0 ? (
          <Empty text="暂无会话" />
        ) : (
          sessions.map((s) => (
            <div
              key={s.session_id}
              style={{
                border: '1px solid var(--border)',
                borderRadius: 8,
                padding: '10px 14px',
                marginBottom: 8,
                background: 'var(--bg-2)',
                cursor: 'pointer',
              }}
              onClick={() => resume(s.session_id)}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                <code style={{ fontSize: 11.5, color: 'var(--accent)' }}>{s.session_id}</code>
                <Tag color="gray">{s.message_count} 条消息</Tag>
                <Tag color="blue">{s.token_count} tokens</Tag>
                {(s.run_status === 'interrupted' || s.run_status === 'running') && <Tag color="yellow">待恢复</Tag>}
                <span style={{ flex: 1 }} />
                <span style={{ fontSize: 11, color: 'var(--text-2)' }}>{fmtTime(s.updated_at)}</span>
              </div>
              <div style={{ fontSize: 12, color: 'var(--text-1)', display: 'flex', alignItems: 'center', gap: 8 }}>
                <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {s.summary || '（无摘要）'}
                </span>
                {s.recoverable && (
                  <button
                    className="btn small"
                    onClick={(e) => {
                      e.stopPropagation()
                      recover(s.session_id)
                    }}
                  >
                    恢复执行
                  </button>
                )}
                <button
                  className="btn small danger"
                  onClick={(e) => {
                    e.stopPropagation()
                    remove(s.session_id)
                  }}
                >
                  删除
                </button>
              </div>
            </div>
          ))
        )}
      </Card>

      <Card title={resumed ? `会话回放 — ${resumed.session_id}` : '会话回放'} hint="点击左侧会话查看完整消息历史">
        {!resumed ? (
          <Empty text="← 选择一个会话进行恢复回放" />
        ) : (
          <>
            {resumed.summary && (
              <div
                style={{
                  border: '1px solid rgba(171,125,248,0.3)',
                  background: 'rgba(171,125,248,0.06)',
                  borderRadius: 8,
                  padding: '10px 12px',
                  marginBottom: 12,
                  fontSize: 12,
                  color: 'var(--text-1)',
                }}
              >
                <div style={{ color: 'var(--purple)', fontSize: 11, marginBottom: 4 }}>压缩摘要</div>
                {resumed.summary}
              </div>
            )}
            <div style={{ maxHeight: 520, overflowY: 'auto' }}>
              {resumed.messages.map((m, i) => (
                <div
                  key={i}
                  style={{
                    marginBottom: 10,
                    display: 'flex',
                    gap: 8,
                  }}
                >
                  <Tag color={m.role === 'user' ? 'gray' : m.role === 'assistant' ? 'blue' : 'purple'}>
                    {m.role}
                  </Tag>
                  <div
                    style={{
                      flex: 1,
                      fontSize: 12.5,
                      color: 'var(--text-1)',
                      background: 'var(--bg-2)',
                      borderRadius: 6,
                      padding: '8px 10px',
                      whiteSpace: 'pre-wrap',
                      wordBreak: 'break-word',
                      maxHeight: 160,
                      overflow: 'auto',
                    }}
                  >
                    {typeof m.content === 'string' ? m.content : JSON.stringify(m.content)}
                  </div>
                </div>
              ))}
            </div>
            <button
              className="btn small"
              style={{ marginTop: 12 }}
              onClick={() => onContinue(resumed)}
            >
              继续对话
            </button>
          </>
        )}
      </Card>
    </div>
  )
}
