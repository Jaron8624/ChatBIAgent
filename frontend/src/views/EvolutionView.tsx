import { useEffect, useState } from 'react'
import { api } from '../lib/api'
import { Card, Empty, Loading, StatCard, Tag, fmtTime } from '../components/ui'

interface EvoRule {
  rule_id: string
  trigger_pattern?: string
  action_prompt?: string
  source_feedback?: string
  status: 'pending' | 'accepted' | 'discarded' | 'merged'
  confidence?: number
  created_at?: string
  usage_count?: number
  success_count?: number
}

const STATUS_COLOR: Record<string, 'yellow' | 'green' | 'red' | 'gray'> = {
  pending: 'yellow',
  accepted: 'green',
  discarded: 'red',
  merged: 'gray',
}

export default function EvolutionView() {
  const [rules, setRules] = useState<EvoRule[] | null>(null)
  const [stats, setStats] = useState<Record<string, unknown> | null>(null)
  const [loopStatus, setLoopStatus] = useState<Record<string, unknown> | null>(null)
  const [filter, setFilter] = useState<'all' | 'pending'>('all')
  const [busy, setBusy] = useState('')
  const [error, setError] = useState('')

  const load = () => {
    api
      .get<{ rules: EvoRule[] }>(`/evolution/rules${filter === 'pending' ? '?status=pending' : ''}`)
      .then((r) => setRules(r.rules || []))
      .catch((e) => setError((e as Error).message))
    api
      .get<{ stats: Record<string, unknown> }>('/evolution/stats')
      .then((r) => setStats(r.stats))
      .catch(() => {})
    api
      .get<{ status: Record<string, unknown> }>('/evolution/loop-status')
      .then((r) => setLoopStatus(r.status))
      .catch(() => {})
  }

  useEffect(load, [filter])

  const review = async (ruleId: string, decision: 'accept' | 'merge' | 'discard') => {
    setBusy(ruleId)
    try {
      await api.post('/evolution/review', { rule_id: ruleId, decision })
      load()
    } catch (e) {
      alert((e as Error).message)
    }
    setBusy('')
  }

  const tick = async () => {
    setBusy('tick')
    try {
      const r = await api.post<{ result: unknown }>('/evolution/tick')
      alert(`自进化 tick 完成：\n${JSON.stringify(r.result, null, 2)}`)
      load()
    } catch (e) {
      alert((e as Error).message)
    }
    setBusy('')
  }

  const autoReview = async () => {
    setBusy('auto')
    try {
      const r = await api.post<{ accepted_count: number }>('/evolution/auto-review')
      alert(`自动接受 ${r.accepted_count} 条高置信规则`)
      load()
    } catch (e) {
      alert((e as Error).message)
    }
    setBusy('')
  }

  if (error) return <div className="error-text">加载失败：{error}</div>

  const pendingCount = rules?.filter((r) => r.status === 'pending').length ?? 0

  return (
    <div>
      <div className="stats-grid">
        <StatCard label="规则总数" value={String(stats?.total_rules ?? rules?.length ?? '—')} tone="blue" />
        <StatCard label="待审核" value={String(pendingCount)} tone={pendingCount > 0 ? 'warn' : 'ok'} />
        <StatCard label="已接受" value={String(stats?.accepted_rules ?? '—')} tone="ok" />
        <StatCard
          label="进化闭环"
          value={loopStatus?.running ? '运行中' : '空闲'}
          tone={loopStatus?.running ? 'ok' : undefined}
          sub={loopStatus?.running ? `间隔 ${String(loopStatus?.interval_s ?? '—')}s` : '手动 tick 触发'}
        />
      </div>

      <Card
        title="进化规则审核"
        hint="BadCase → 规则提炼 → 人工决策（人在环上）"
        actions={
          <>
            <button
              className={`btn small ${filter === 'all' ? 'primary' : ''}`}
              onClick={() => setFilter('all')}
            >
              全部
            </button>
            <button
              className={`btn small ${filter === 'pending' ? 'primary' : ''}`}
              onClick={() => setFilter('pending')}
            >
              待审核
            </button>
            <button className="btn small" disabled={busy === 'tick'} onClick={tick}>
              {busy === 'tick' ? '执行中…' : '⟳ 手动 Tick'}
            </button>
            <button className="btn small" disabled={busy === 'auto'} onClick={autoReview}>
              自动审核
            </button>
          </>
        }
      >
        {rules === null ? (
          <Loading />
        ) : rules.length === 0 ? (
          <Empty text="暂无进化规则 — 评测产生 BadCase 后自动提炼" />
        ) : (
          rules.map((r) => (
            <div
              key={r.rule_id}
              style={{
                border: '1px solid var(--border)',
                borderRadius: 8,
                padding: '12px 14px',
                marginBottom: 10,
                background: 'var(--bg-2)',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
                <code style={{ fontSize: 11.5, color: 'var(--text-2)' }}>{r.rule_id}</code>
                <Tag color={STATUS_COLOR[r.status] || 'gray'}>{r.status}</Tag>
                {r.confidence !== undefined && (
                  <Tag color={r.confidence >= 0.8 ? 'green' : 'yellow'}>
                    置信 {(r.confidence * 100).toFixed(0)}%
                  </Tag>
                )}
                {r.usage_count !== undefined && r.usage_count > 0 && (
                  <Tag color="blue">使用 {r.usage_count} 次</Tag>
                )}
                <span style={{ flex: 1 }} />
                <span style={{ fontSize: 11, color: 'var(--text-2)' }}>{fmtTime(r.created_at)}</span>
              </div>
              {r.trigger_pattern && (
                <div style={{ fontSize: 12.5, marginBottom: 4 }}>
                  <span style={{ color: 'var(--text-2)' }}>触发模式：</span>
                  <code style={{ color: 'var(--accent)' }}>{r.trigger_pattern}</code>
                </div>
              )}
              {r.action_prompt && (
                <div style={{ fontSize: 12.5, color: 'var(--text-1)', marginBottom: 4 }}>
                  <span style={{ color: 'var(--text-2)' }}>改进动作：</span>
                  {r.action_prompt}
                </div>
              )}
              {r.source_feedback && (
                <div style={{ fontSize: 11.5, color: 'var(--text-2)', marginBottom: 6 }}>
                  反馈来源：{r.source_feedback}
                </div>
              )}
              {r.status === 'pending' && (
                <div style={{ display: 'flex', gap: 8 }}>
                  <button
                    className="btn small primary"
                    disabled={busy === r.rule_id}
                    onClick={() => review(r.rule_id, 'accept')}
                  >
                    ✓ 接受
                  </button>
                  <button
                    className="btn small"
                    disabled={busy === r.rule_id}
                    onClick={() => review(r.rule_id, 'merge')}
                  >
                    ⇄ 合并
                  </button>
                  <button
                    className="btn small danger"
                    disabled={busy === r.rule_id}
                    onClick={() => review(r.rule_id, 'discard')}
                  >
                    ✗ 驳回
                  </button>
                </div>
              )}
            </div>
          ))
        )}
      </Card>

      <Card title="自进化闭环" hint="BadCase 自动回流 → 规则提炼 → 审核 → 反哺词表/知识库/评测集">
        <div style={{ fontSize: 12.5, color: 'var(--text-1)', lineHeight: 1.9 }}>
          <div>1. 评测低分回答自动进入 BadCase 库</div>
          <div>2. 进化循环定期分析失败模式，提炼候选规则（pattern + suggestion + confidence）</div>
          <div>3. 高置信（≥0.8）规则可自动接受，其余人工审核</div>
          <div>4. 接受的规则反哺三个方向：意图词表 / 知识库 FAQ / 评测集回归用例</div>
        </div>
        {loopStatus && (
          <pre className="code-block" style={{ marginTop: 12 }}>
            {JSON.stringify(loopStatus, null, 2)}
          </pre>
        )}
      </Card>
    </div>
  )
}
