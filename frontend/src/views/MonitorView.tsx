import { useEffect, useState } from 'react'
import { api } from '../lib/api'
import { Card, Empty, Loading, StatCard, Tag } from '../components/ui'

interface AgentMetric {
  agent_name: string
  total_calls: number
  success_calls: number
  success_rate: number
  p99_latency_ms: number
  avg_latency_ms: number
  avg_tokens: number
}

interface AgentMetricsDict {
  [name: string]: AgentMetric
}

export default function MonitorView() {
  const [metrics, setMetrics] = useState<AgentMetricsDict | null>(null)
  const [weights, setWeights] = useState<Record<string, number> | null>(null)
  const [error, setError] = useState('')

  const load = () => {
    api
      .get<{ metrics: AgentMetricsDict; weights: Record<string, number> }>('/monitor/metrics')
      .then((r) => {
        setMetrics(r.metrics || {})
        setWeights(r.weights || {})
        setError('')
      })
      .catch((e) => setError((e as Error).message))
  }

  useEffect(load, [])
  useEffect(() => {
    const t = setInterval(load, 15_000)
    return () => clearInterval(t)
  }, [])

  if (error) return <div className="error-text">加载失败：{error}</div>
  if (!metrics) return <Loading text="加载监控指标" />

  const metricList = Object.values(metrics)
  const totalCalls = metricList.reduce((s, m) => s + m.total_calls, 0)
  const avgP99 = metricList.length ? metricList.reduce((s, m) => s + m.p99_latency_ms, 0) / metricList.length : 0

  return (
    <div>
      <div className="stats-grid">
        <StatCard label="监控 Agent 数" value={metricList.length} tone="blue" />
        <StatCard label="窗口内总调用" value={totalCalls} sub="1000 条 / 5 分钟滑窗" />
        <StatCard
          label="平均 P99 延迟"
          value={`${avgP99.toFixed(0)}ms`}
          tone={avgP99 > 3000 ? 'bad' : avgP99 > 1500 ? 'warn' : 'ok'}
        />
      </div>

      <Card
        title="Agent 指标"
        hint="滑动窗口（1000 条 / 300s）实时聚合，15s 自动刷新"
        actions={
          <button className="btn small" onClick={load}>
            刷新
          </button>
        }
      >
        {metricList.length === 0 ? (
          <Empty text="窗口内暂无调用记录 — 去对话页发几条消息试试" />
        ) : (
          <table>
            <thead>
              <tr>
                <th>Agent</th>
                <th>调用量</th>
                <th>成功率</th>
                <th>P99 延迟</th>
                <th>平均延迟</th>
                <th>平均 Token</th>
              </tr>
            </thead>
            <tbody>
              {metricList.map((m) => (
                <tr key={m.agent_name}>
                  <td>
                    <code style={{ color: 'var(--accent)' }}>{m.agent_name}</code>
                  </td>
                  <td>{m.total_calls}</td>
                  <td>
                    <span style={{ color: m.success_rate >= 0.95 ? 'var(--green)' : 'var(--red)' }}>
                      {(m.success_rate * 100).toFixed(1)}%
                    </span>
                  </td>
                  <td>
                    <span
                      style={{
                        color:
                          m.p99_latency_ms > 5000
                            ? 'var(--red)'
                            : m.p99_latency_ms > 3000
                              ? 'var(--yellow)'
                              : 'var(--text-1)',
                      }}
                    >
                      {m.p99_latency_ms.toFixed(0)}ms
                    </span>
                  </td>
                  <td>{m.avg_latency_ms.toFixed(0)}ms</td>
                  <td>{m.avg_tokens}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Card>

      <Card title="动态权重" hint="成功率 < 95% 减半 · P99 > 3s 减半 · > 5s 降至 0.1 · 优秀提升至 1.2x">
        {!weights || Object.keys(weights).length === 0 ? (
          <Empty />
        ) : (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(150px, 1fr))', gap: 10 }}>
            {Object.entries(weights).map(([name, w]) => {
              const tone = w < 0.5 ? 'bad' : w > 1 ? 'ok' : 'blue'
              const pct = Math.min((w / 2) * 100, 100)
              return (
                <div
                  key={name}
                  style={{
                    border: '1px solid var(--border)',
                    borderRadius: 8,
                    padding: '10px 12px',
                    background: 'var(--bg-2)',
                  }}
                >
                  <div style={{ fontSize: 12, marginBottom: 6 }}>
                    <code>{name}</code>
                  </div>
                  <div
                    style={{
                      height: 6,
                      background: 'var(--bg-0)',
                      borderRadius: 3,
                      overflow: 'hidden',
                      marginBottom: 6,
                    }}
                  >
                    <div
                      style={{
                        height: '100%',
                        width: `${pct}%`,
                        borderRadius: 3,
                        background:
                          tone === 'bad' ? 'var(--red)' : tone === 'ok' ? 'var(--green)' : 'var(--accent)',
                        transition: 'width 0.4s',
                      }}
                    />
                  </div>
                  <div style={{ fontSize: 11.5, fontFamily: 'var(--mono)' }}>
                    ×{w.toFixed(2)}{' '}
                    {w < 1 && <Tag color="red">降权</Tag>}
                    {w > 1 && <Tag color="green">提升</Tag>}
                  </div>
                </div>
              )
            })}
          </div>
        )}
      </Card>
    </div>
  )
}
