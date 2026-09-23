import { useEffect, useState } from 'react'
import { api } from '../lib/api'
import { Card, Empty, Loading, StatCard, Tag } from '../components/ui'

interface Skill {
  id?: string
  name: string
  description: string
  prompt_template: string
  examples?: string[]
  category: string
  tags?: string[]
  version?: string
  usage_count?: number
  success_rate?: number
  created_at?: string
}

export default function SkillsView() {
  const [skills, setSkills] = useState<Skill[] | null>(null)
  const [stats, setStats] = useState<Record<string, unknown> | null>(null)
  const [error, setError] = useState('')
  const [showCreate, setShowCreate] = useState(false)
  const [form, setForm] = useState({
    name: '',
    description: '',
    prompt_template: '',
    category: 'general',
    tags: '',
    examples: '',
  })
  const [creating, setCreating] = useState(false)

  const load = () => {
    api
      .get<{ skills: Skill[] }>('/skills')
      .then((r) => setSkills(r.skills || []))
      .catch((e) => setError((e as Error).message))
    api
      .get<{ stats: Record<string, unknown> }>('/skills/stats')
      .then((r) => setStats(r.stats))
      .catch(() => {})
  }

  useEffect(load, [])

  const create = async () => {
    if (!form.name.trim() || !form.prompt_template.trim()) {
      alert('名称和 Prompt 模板必填')
      return
    }
    setCreating(true)
    try {
      await api.post('/skills', {
        name: form.name,
        description: form.description,
        prompt_template: form.prompt_template,
        category: form.category,
        tags: form.tags.split(',').map((t) => t.trim()).filter(Boolean),
        examples: form.examples.split('\n').map((t) => t.trim()).filter(Boolean),
      })
      setShowCreate(false)
      setForm({ name: '', description: '', prompt_template: '', category: 'general', tags: '', examples: '' })
      load()
    } catch (e) {
      alert((e as Error).message)
    }
    setCreating(false)
  }

  if (error) return <div className="error-text">加载失败：{error}</div>

  const categories = (stats?.categories as Record<string, number>) || {}

  return (
    <div>
      <div className="stats-grid">
        <StatCard label="Skills 总数" value={String(stats?.total_skills ?? skills?.length ?? '—')} tone="blue" />
        <StatCard label="累计使用" value={String(stats?.total_usage ?? '—')} />
        <StatCard label="分类数" value={Object.keys(categories).length} />
      </div>

      <Card
        title="技能模板库"
        hint="自进化接受的规则沉淀为 Skill，按相似度注入 Agent Prompt"
        actions={
          <button className="btn small primary" onClick={() => setShowCreate(!showCreate)}>
            {showCreate ? '收起' : '＋ 新建 Skill'}
          </button>
        }
      >
        {showCreate && (
          <div
            style={{
              border: '1px solid var(--border-light)',
              borderRadius: 8,
              padding: 14,
              marginBottom: 16,
              background: 'var(--bg-2)',
              display: 'flex',
              flexDirection: 'column',
              gap: 10,
            }}
          >
            <div style={{ display: 'flex', gap: 10 }}>
              <input
                className="input"
                style={{ flex: 1 }}
                placeholder="名称（如：超时赔付话术）"
                value={form.name}
                onChange={(e) => setForm({ ...form, name: e.target.value })}
              />
              <select
                className="select"
                value={form.category}
                onChange={(e) => setForm({ ...form, category: e.target.value })}
              >
                {['general', 'refund', 'complaint', 'delivery', 'policy', 'safety'].map((c) => (
                  <option key={c} value={c}>
                    {c}
                  </option>
                ))}
              </select>
            </div>
            <input
              className="input"
              placeholder="描述"
              value={form.description}
              onChange={(e) => setForm({ ...form, description: e.target.value })}
            />
            <textarea
              className="textarea"
              placeholder="Prompt 模板（支持 {query} 占位符）"
              value={form.prompt_template}
              onChange={(e) => setForm({ ...form, prompt_template: e.target.value })}
            />
            <div style={{ display: 'flex', gap: 10 }}>
              <input
                className="input"
                style={{ flex: 1 }}
                placeholder="标签（逗号分隔）"
                value={form.tags}
                onChange={(e) => setForm({ ...form, tags: e.target.value })}
              />
              <input
                className="input"
                style={{ flex: 1 }}
                placeholder="示例（每行一条）"
                value={form.examples}
                onChange={(e) => setForm({ ...form, examples: e.target.value })}
              />
            </div>
            <div>
              <button className="btn primary" disabled={creating} onClick={create}>
                {creating ? '创建中…' : '创建 Skill'}
              </button>
            </div>
          </div>
        )}

        {skills === null ? (
          <Loading />
        ) : skills.length === 0 ? (
          <Empty text="暂无 Skills" />
        ) : (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(340px, 1fr))', gap: 12 }}>
            {skills.map((s) => (
              <div
                key={s.id || s.name}
                style={{
                  border: '1px solid var(--border)',
                  borderRadius: 8,
                  padding: '12px 14px',
                  background: 'var(--bg-2)',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
                  <span style={{ color: 'var(--purple)' }}>✦</span>
                  <span style={{ fontSize: 13, fontWeight: 600 }}>{s.name}</span>
                  <Tag color="blue">{s.category}</Tag>
                  {s.version && <Tag color="gray">v{s.version}</Tag>}
                </div>
                <div style={{ fontSize: 12, color: 'var(--text-1)', marginBottom: 8, lineHeight: 1.5 }}>
                  {s.description || '—'}
                </div>
                {s.tags && s.tags.length > 0 && (
                  <div style={{ display: 'flex', gap: 5, flexWrap: 'wrap', marginBottom: 8 }}>
                    {s.tags.map((t) => (
                      <Tag key={t} color="gray">
                        {t}
                      </Tag>
                    ))}
                  </div>
                )}
                <details>
                  <summary style={{ fontSize: 11.5, color: 'var(--text-2)', cursor: 'pointer' }}>
                    Prompt 模板 ▾
                  </summary>
                  <pre className="code-block" style={{ marginTop: 8, maxHeight: 180, overflow: 'auto' }}>
                    {s.prompt_template}
                  </pre>
                </details>
                {(s.usage_count !== undefined || s.success_rate !== undefined) && (
                  <div style={{ marginTop: 8, fontSize: 11.5, color: 'var(--text-2)' }}>
                    使用 {s.usage_count ?? 0} 次 · 成功率{' '}
                    {s.success_rate !== undefined ? `${(s.success_rate * 100).toFixed(0)}%` : '—'}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </Card>
    </div>
  )
}
