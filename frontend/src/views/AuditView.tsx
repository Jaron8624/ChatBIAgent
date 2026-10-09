import { useEffect, useState } from 'react'
import { api, getApiContext } from '../lib/api'
import { Card, Empty, Loading, Tag, fmtTime } from '../components/ui'

interface AuditLog {
  user_id: number
  action: string
  resource: string
  success: boolean
  timestamp: number
  details?: string
  ip_address?: string
}

export default function AuditView() {
  const [tab, setTab] = useState<'audit' | 'dangerous'>('audit')
  const [logs, setLogs] = useState<AuditLog[] | null>(null)
  const [perms, setPerms] = useState<{ role: string; permissions: string[] } | null>(null)
  const [error, setError] = useState('')

  const load = () => {
    const path = tab === 'audit' ? '/acl/audit-logs?limit=200' : '/acl/dangerous-logs?limit=100'
    api
      .get<{ logs: AuditLog[] }>(path)
      .then((r) => setLogs(r.logs || []))
      .catch((e) => {
        setLogs([])
        setError((e as Error).message)
      })
    api
      .get<{ role: string; permissions: string[] }>('/acl/permissions')
      .then(setPerms)
      .catch(() => setPerms(null))
  }

  useEffect(load, [tab])

  return (
    <div>
      <Card
        title={`当前身份：${getApiContext().role}`}
        hint="角色权限由 ACL 控制（admin / editor / viewer / guest）"
      >
        {perms ? (
          <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
            <Tag color="blue">{perms.role}</Tag>
            {perms.permissions.map((p) => (
              <Tag key={p} color="green">
                {p}
              </Tag>
            ))}
          </div>
        ) : (
          <Empty text="当前身份无权限查看权限列表（需 admin）" />
        )}
      </Card>

      <div style={{ display: 'flex', gap: 6, marginBottom: 16 }}>
        <button className={`btn ${tab === 'audit' ? 'primary' : ''}`} onClick={() => { setError(''); setTab('audit') }}>
          全部审计日志
        </button>
        <button className={`btn ${tab === 'dangerous' ? 'primary' : ''}`} onClick={() => { setError(''); setTab('dangerous') }}>
          危险操作日志
        </button>
        <button className="btn" onClick={() => { setError(''); load() }}>
          刷新
        </button>
      </div>

      <Card title={tab === 'audit' ? '审计日志' : '危险操作日志'} hint="环形缓冲，上限 5000 条">
        {error && <div className="error-text" style={{ marginBottom: 10 }}>{error}</div>}
        {logs === null ? (
          <Loading />
        ) : logs.length === 0 ? (
          <Empty text="暂无日志" />
        ) : (
          <table>
            <thead>
              <tr>
                <th>时间</th>
                <th>用户</th>
                <th>操作</th>
                <th>资源</th>
                <th>结果</th>
                <th>详情</th>
              </tr>
            </thead>
            <tbody>
              {logs.map((l, i) => (
                <tr key={i}>
                  <td style={{ whiteSpace: 'nowrap', color: 'var(--text-2)' }}>{fmtTime(l.timestamp * 1000)}</td>
                  <td>
                    <code>{l.user_id}</code>
                  </td>
                  <td>
                    <code style={{ color: 'var(--accent)' }}>{l.action}</code>
                  </td>
                  <td>{l.resource}</td>
                  <td>{l.success ? <Tag color="green">成功</Tag> : <Tag color="red">失败</Tag>}</td>
                  <td style={{ color: 'var(--text-2)', maxWidth: 300, wordBreak: 'break-all' }}>
                    {l.details || '—'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Card>

      <Card title="安全机制" hint="四级角色 · 危险操作二次确认 · 审计留痕">
        <div style={{ fontSize: 12.5, color: 'var(--text-1)', lineHeight: 1.9 }}>
          <div>· <b>角色分级</b>：admin（全部）{' > '}editor（常规读写）{' > '}viewer（只读）{' > '}guest（仅对话）</div>
          <div>· <b>危险操作</b>：退款执行、知识库删除等需 token 二次确认（5 分钟过期）</div>
          <div>· <b>审计日志</b>：所有敏感操作记录 谁/何时/做了什么/成功与否</div>
          <div>· <b>转人工红线</b>：食品安全、账户安全场景强制升级人工（见对话视图）</div>
        </div>
      </Card>
    </div>
  )
}
