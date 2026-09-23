import type { ReactNode } from 'react'

export function Card({
  title,
  hint,
  children,
  actions,
}: {
  title?: string
  hint?: string
  children: ReactNode
  actions?: ReactNode
}) {
  return (
    <div className="card">
      {(title || actions) && (
        <div className="card-title">
          {title}
          {hint && <span className="hint">{hint}</span>}
          {actions && (
            <span style={{ marginLeft: 'auto', display: 'flex', gap: 8 }}>{actions}</span>
          )}
        </div>
      )}
      {children}
    </div>
  )
}

export function StatCard({
  label,
  value,
  tone,
  sub,
}: {
  label: string
  value: string | number
  tone?: 'ok' | 'warn' | 'bad' | 'blue'
  sub?: string
}) {
  return (
    <div className="stat-card">
      <div className="label">{label}</div>
      <div className={`value ${tone || ''}`}>{value}</div>
      {sub && <div className="sub">{sub}</div>}
    </div>
  )
}

export function Loading({ text = '加载中' }: { text?: string }) {
  return (
    <div className="loading">
      <div className="spinner" />
      {text}
    </div>
  )
}

export function Empty({ text = '暂无数据' }: { text?: string }) {
  return <div className="empty-state">{text}</div>
}

export function Tag({
  children,
  color,
}: {
  children: ReactNode
  color: 'blue' | 'green' | 'yellow' | 'red' | 'purple' | 'gray'
}) {
  return <span className={`tag ${color}`}>{children}</span>
}

/** 意图标签配色 */
export function IntentTag({ intent }: { intent: string }) {
  const color: 'blue' | 'green' | 'yellow' | 'purple' | 'gray' =
    intent === 'knowledge_rag'
      ? 'purple'
      : ['order_query', 'delivery_track', 'refund_request', 'coupon_query', 'complaint', 'ticket_handler'].includes(
            intent,
          )
        ? 'blue'
        : intent === 'chitchat'
          ? 'gray'
          : 'yellow'
  return <Tag color={color}>{intent || '—'}</Tag>
}

/** 时间格式化 */
export function fmtTime(ts?: string | number) {
  if (!ts) return '—'
  const d = new Date(ts)
  if (isNaN(d.getTime())) return String(ts)
  return d.toLocaleString('zh-CN', { hour12: false })
}
