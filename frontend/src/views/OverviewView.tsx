import { useEffect, useState } from 'react'
import { api, getApiContext } from '../lib/api'
import { Card, StatCard, Tag, fmtTime } from '../components/ui'

interface SessionItem {
  session_id: string
  summary: string
  token_count: number
  updated_at: string
  message_count: number
}

export default function OverviewView({ onNavigate }: { onNavigate: (v: string) => void }) {
  const [tools, setTools] = useState<number | null>(null)
  const [sessions, setSessions] = useState<SessionItem[]>([])
  const [metrics, setMetrics] = useState<{ total_calls: number; success_rate: number; p99: number } | null>(null)
  const [skills, setSkills] = useState<number | null>(null)
  const [rules, setRules] = useState<{ total: number; pending: number } | null>(null)

  useEffect(() => {
    api.get<{ tools: unknown[] }>('/mcp/tools').then((r) => setTools(r.tools?.length ?? 0)).catch(() => {})
    api
      .get<{ sessions: SessionItem[] }>(`/session/list?limit=5`)
      .then((r) => setSessions(r.sessions || []))
      .catch(() => {})
    api
      .get<{ metrics: { total_calls: number; success_rate: number; p99_latency_ms: number }[] }>('/monitor/metrics')
      .then((r) => {
        const m = r.metrics || []
        setMetrics({
          total_calls: m.reduce((s, x) => s + x.total_calls, 0),
          success_rate: m.length ? m.reduce((s, x) => s + x.success_rate, 0) / m.length : 1,
          p99: m.length ? Math.max(...m.map((x) => x.p99_latency_ms)) : 0,
        })
      })
      .catch(() => {})
    api
      .get<{ stats: { total_skills: number } }>('/skills/stats')
      .then((r) => setSkills(r.stats?.total_skills ?? 0))
      .catch(() => {})
    api
      .get<{ rules: { status: string }[] }>('/evolution/rules')
      .then((r) => {
        const all = r.rules || []
        setRules({ total: all.length, pending: all.filter((x) => x.status === 'pending').length })
      })
      .catch(() => {})
  }, [])

  const quickLinks = [
    { view: 'chat', icon: '◍', title: 'Agent 对话', desc: '多轮对话调试，查看 ReAct 轨迹与转人工判定' },
    { view: 'eval', icon: '◈', title: '运行黄金测试集', desc: '8 用例双层评测（过程指标 + LLM Judge）' },
    { view: 'mcp', icon: '⚙', title: 'MCP 工具面板', desc: '查看工具注册表并直接调用调试' },
    { view: 'monitor', icon: '◔', title: '监控大盘', desc: '滑动窗口指标与动态权重' },
  ]

  return (
    <div style={{ maxWidth: 1000, margin: '0 auto' }}>
      {/* 系统横幅 */}
      <div
        style={{
          background: 'linear-gradient(135deg, rgba(79,156,247,0.12), rgba(171,125,248,0.10))',
          border: '1px solid rgba(79,156,247,0.25)',
          borderRadius: 10,
          padding: '22px 24px',
          marginBottom: 20,
        }}
      >
        <div style={{ fontSize: 20, fontWeight: 700, marginBottom: 6 }}>NexusMind Agent 控制台</div>
        <div style={{ fontSize: 13, color: 'var(--text-1)', lineHeight: 1.7 }}>
          企业级外卖客服 Agent 系统 —— LangGraph 编排 · 自研 MCP 协议层 · ReAct 循环 · RAG 混合检索 ·
          分层记忆 · 双层评测 · 自进化闭环
        </div>
        <div style={{ display: 'flex', gap: 8, marginTop: 12, flexWrap: 'wrap' }}>
          <Tag color="blue">JSON-RPC 2.0</Tag>
          <Tag color="purple">ReAct Loop</Tag>
          <Tag color="green">Hybrid RAG</Tag>
          <Tag color="yellow">LLM-as-Judge</Tag>
          <Tag color="blue">Multi-Agent</Tag>
          <Tag color="purple">ContextStore</Tag>
        </div>
      </div>

      {/* 指标卡片 */}
      <div className="stats-grid">
        <StatCard label="MCP 工具" value={tools ?? '—'} tone="blue" sub="含 recall_context" />
        <StatCard
          label="窗口内调用"
          value={metrics?.total_calls ?? '—'}
          sub="近 5 分钟滑动窗口"
        />
        <StatCard
          label="平均成功率"
          value={metrics ? `${(metrics.success_rate * 100).toFixed(0)}%` : '—'}
          tone={metrics && metrics.success_rate >= 0.95 ? 'ok' : 'warn'}
        />
        <StatCard label="Skills" value={skills ?? '—'} sub="技能模板" />
        <StatCard
          label="进化规则"
          value={rules ? rules.total : '—'}
          tone={rules && rules.pending > 0 ? 'warn' : 'ok'}
          sub={rules ? `${rules.pending} 待审核` : undefined}
        />
        <StatCard label="活跃会话" value={sessions.length} sub={`UID ${getApiContext().userId}`} />
      </div>

      {/* 快捷入口 */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))', gap: 12, marginBottom: 16 }}>
        {quickLinks.map((q) => (
          <div
            key={q.view}
            onClick={() => onNavigate(q.view)}
            style={{
              border: '1px solid var(--border)',
              borderRadius: 8,
              padding: '14px 16px',
              background: 'var(--bg-1)',
              cursor: 'pointer',
              transition: 'border-color 0.15s',
            }}
            onMouseEnter={(e) => (e.currentTarget.style.borderColor = 'var(--accent)')}
            onMouseLeave={(e) => (e.currentTarget.style.borderColor = 'var(--border)')}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
              <span style={{ color: 'var(--accent)', fontSize: 16 }}>{q.icon}</span>
              <span style={{ fontSize: 13.5, fontWeight: 600 }}>{q.title}</span>
            </div>
            <div style={{ fontSize: 12, color: 'var(--text-2)', lineHeight: 1.5 }}>{q.desc}</div>
          </div>
        ))}
      </div>

      {/* 最近会话 */}
      <Card
        title="最近会话"
        hint="Redis 存储 · zlib 压缩 · 滑动窗口摘要"
        actions={
          <button className="btn small" onClick={() => onNavigate('sessions')}>
            查看全部
          </button>
        }
      >
        {sessions.length === 0 ? (
          <div className="empty-state">暂无会话 — 去对话页发起第一轮对话</div>
        ) : (
          <table>
            <thead>
              <tr>
                <th>会话</th>
                <th>摘要</th>
                <th>消息数</th>
                <th>Token</th>
                <th>更新时间</th>
              </tr>
            </thead>
            <tbody>
              {sessions.map((s) => (
                <tr key={s.session_id}>
                  <td>
                    <code style={{ color: 'var(--accent)', fontSize: 11.5 }}>{s.session_id}</code>
                  </td>
                  <td style={{ maxWidth: 320, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {s.summary || '—'}
                  </td>
                  <td>{s.message_count}</td>
                  <td>{s.token_count}</td>
                  <td style={{ color: 'var(--text-2)', whiteSpace: 'nowrap' }}>{fmtTime(s.updated_at)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Card>
    </div>
  )
}
