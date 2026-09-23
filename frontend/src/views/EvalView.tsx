import { useEffect, useState } from 'react'
import { api } from '../lib/api'
import { Card, Empty, Loading, StatCard, Tag } from '../components/ui'

interface CaseResult {
  case_id: string
  description?: string
  process?: {
    intent_match?: boolean
    tool_accuracy?: number | null
    tool_efficiency?: number | null
    requires_human_match?: boolean
    keyword_coverage?: number
    called_tools?: string[]
    route_hit?: boolean | null
  }
  result?: { judge_score?: number | null; passed?: boolean | null }
  react_meta?: { iterations?: number; tool_calls?: string[]; forced_final?: boolean } | null
  response_preview?: string
  error?: string
}

interface GoldenReport {
  total?: number
  intent_pass_rate?: string // "8/8"
  human_escalation_pass_rate?: string
  route_pass_rate?: string // "2/2" | "N/A"
  avg_keyword_coverage?: number
  avg_judge_score?: number
  generated_at?: string
  report_path?: string
  results?: CaseResult[]
}

/** "8/8" -> [8, 8] */
function parseRate(s?: string): [number, number] | null {
  if (!s) return null
  const m = s.match(/^(\d+)\s*\/\s*(\d+)$/)
  return m ? [Number(m[1]), Number(m[2])] : null
}

const JUDGE_DIMS = [
  ['relevance', '相关性'],
  ['accuracy', '准确性'],
  ['completeness', '完整性'],
  ['safety', '安全性'],
]

function casePass(c: CaseResult): boolean {
  const p = c.process || {}
  const judgeOk = c.result?.judge_score === null || c.result?.judge_score === undefined ? true : c.result.judge_score >= 6
  return Boolean(p.intent_match) && Boolean(p.requires_human_match) && judgeOk
}

export default function EvalView() {
  const [tab, setTab] = useState<'golden' | 'badcases' | 'single'>('golden')
  const [golden, setGolden] = useState<GoldenReport | null>(null)
  const [running, setRunning] = useState(false)
  const [badCases, setBadCases] = useState<{ total: number; cases: unknown[] } | null>(null)
  const [summary, setSummary] = useState<Record<string, unknown> | null>(null)
  // 单条评测
  const [sq, setSq] = useState('')
  const [sr, setSr] = useState('')
  const [sc, setSc] = useState('')
  const [singleResult, setSingleResult] = useState<{
    overall_score: number
    passed: boolean
    dimensions: { dimension: string; score: number; comment?: string }[]
  } | null>(null)
  const [singleBusy, setSingleBusy] = useState(false)

  const loadBadCases = () => {
    api
      .get<{ total: number; cases: unknown[] }>('/evaluation/bad-cases?limit=50')
      .then(setBadCases)
      .catch(() => {})
    api
      .get<{ summary: Record<string, unknown> }>('/evaluation/summary')
      .then((r) => setSummary(r.summary))
      .catch(() => {})
  }

  useEffect(() => {
    if (tab === 'badcases') loadBadCases()
  }, [tab])

  const runGolden = async () => {
    setRunning(true)
    setGolden(null)
    try {
      const r = await api.post<GoldenReport>('/evaluation/golden')
      setGolden(r)
    } catch (e) {
      alert(`评测失败：${(e as Error).message}`)
    }
    setRunning(false)
  }

  const runSingle = async () => {
    if (!sq.trim() || !sr.trim()) return
    setSingleBusy(true)
    setSingleResult(null)
    try {
      const r = await api.post<{
        overall_score: number
        passed: boolean
        dimensions: { dimension: string; score: number; comment?: string }[]
      }>('/evaluation/evaluate', { query: sq, response: sr, context: sc })
      setSingleResult(r)
    } catch (e) {
      alert(`评测失败：${(e as Error).message}`)
    }
    setSingleBusy(false)
  }

  const intentRate = parseRate(golden?.intent_pass_rate)
  const humanRate = parseRate(golden?.human_escalation_pass_rate)
  const routeRate = parseRate(golden?.route_pass_rate)
  const passCount = golden?.results?.filter(casePass).length ?? 0

  return (
    <div>
      {/* Tabs */}
      <div style={{ display: 'flex', gap: 6, marginBottom: 16 }}>
        {(
          [
            ['golden', '黄金测试集'],
            ['badcases', 'BadCase 库'],
            ['single', '单条评测'],
          ] as const
        ).map(([id, label]) => (
          <button key={id} className={`btn ${tab === id ? 'primary' : ''}`} onClick={() => setTab(id)}>
            {label}
          </button>
        ))}
      </div>

      {tab === 'golden' && (
        <>
          <Card
            title="黄金测试集 · 双层评测"
            hint="过程层（意图/工具/路由/升级）+ 结果层（LLM-as-Judge 四维）"
            actions={
              <button className="btn primary" disabled={running} onClick={runGolden}>
                {running ? '评测运行中…' : '▶ 运行评测'}
              </button>
            }
          >
            {running && <Loading text="正在执行 8 个用例（真实调用编排链路 + LLM Judge）…" />}
            {!running && !golden && <Empty text="点击「运行评测」执行黄金测试集（约 1~2 分钟）" />}
          </Card>

          {golden && (
            <>
              <div className="stats-grid">
                <StatCard
                  label="综合通过"
                  value={`${passCount}/${golden.total ?? '—'}`}
                  tone={passCount === golden.total ? 'ok' : 'warn'}
                />
                <StatCard
                  label="意图准确率"
                  value={intentRate ? `${intentRate[0]}/${intentRate[1]}` : '—'}
                  tone={intentRate && intentRate[0] === intentRate[1] ? 'ok' : 'warn'}
                />
                <StatCard
                  label="升级判定"
                  value={humanRate ? `${humanRate[0]}/${humanRate[1]}` : '—'}
                  tone={humanRate && humanRate[0] === humanRate[1] ? 'ok' : 'warn'}
                />
                <StatCard
                  label="Judge 均分"
                  value={golden.avg_judge_score?.toFixed(1) ?? '—'}
                  tone={golden.avg_judge_score !== undefined && golden.avg_judge_score >= 8 ? 'ok' : 'warn'}
                  sub="满分 10"
                />
                <StatCard
                  label="关键词覆盖"
                  value={golden.avg_keyword_coverage !== undefined ? `${(golden.avg_keyword_coverage * 100).toFixed(0)}%` : '—'}
                  tone="blue"
                />
                <StatCard
                  label="路由命中"
                  value={routeRate ? `${routeRate[0]}/${routeRate[1]}` : golden.route_pass_rate || '—'}
                  tone={routeRate && routeRate[0] === routeRate[1] ? 'ok' : routeRate ? 'warn' : undefined}
                  sub="react / coordinate"
                />
              </div>

              <Card title="用例明细">
                <table>
                  <thead>
                    <tr>
                      <th>用例</th>
                      <th>意图</th>
                      <th>工具精度</th>
                      <th>调用链</th>
                      <th>关键词</th>
                      <th>路由</th>
                      <th>升级</th>
                      <th>Judge</th>
                      <th>结果</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(golden.results || []).map((c) => {
                      const p = c.process || {}
                      const react = c.react_meta
                      return (
                        <tr key={c.case_id}>
                          <td>
                            <div style={{ color: 'var(--text-0)' }}>{c.case_id}</div>
                            <div style={{ fontSize: 11, color: 'var(--text-2)', maxWidth: 200 }}>{c.description}</div>
                            {react && (
                              <div style={{ fontSize: 10.5, color: 'var(--purple)', marginTop: 2 }}>
                                ReAct {react.iterations} 轮{react.forced_final ? '（强制兜底）' : ''}
                              </div>
                            )}
                            {c.response_preview && (
                              <div
                                style={{
                                  fontSize: 10.5,
                                  color: 'var(--text-2)',
                                  marginTop: 3,
                                  maxWidth: 220,
                                  overflow: 'hidden',
                                  textOverflow: 'ellipsis',
                                  whiteSpace: 'nowrap',
                                }}
                                title={c.response_preview}
                              >
                                {c.response_preview}
                              </div>
                            )}
                          </td>
                          <td>{p.intent_match ? <Tag color="green">✓</Tag> : <Tag color="red">✗</Tag>}</td>
                          <td>
                            {p.tool_accuracy === null || p.tool_accuracy === undefined
                              ? '—'
                              : `${(p.tool_accuracy * 100).toFixed(0)}%`}
                          </td>
                          <td>
                            <div style={{ display: 'flex', gap: 3, flexWrap: 'wrap', maxWidth: 140 }}>
                              {(p.called_tools || []).map((t, i) => (
                                <code key={i} style={{ fontSize: 10.5, color: 'var(--accent)' }}>
                                  {t}
                                </code>
                              ))}
                              {(!p.called_tools || p.called_tools.length === 0) && <span style={{ color: 'var(--text-2)' }}>—</span>}
                            </div>
                          </td>
                          <td>
                            {p.keyword_coverage !== undefined
                              ? `${(p.keyword_coverage * 100).toFixed(0)}%`
                              : '—'}
                          </td>
                          <td>
                            {p.route_hit === null || p.route_hit === undefined
                              ? '—'
                              : p.route_hit
                                ? <Tag color="green">✓</Tag>
                                : <Tag color="red">✗</Tag>}
                          </td>
                          <td>
                            {p.requires_human_match ? <Tag color="green">✓</Tag> : <Tag color="red">✗</Tag>}
                          </td>
                          <td>
                            {c.result?.judge_score !== null && c.result?.judge_score !== undefined ? (
                              <span
                                style={{
                                  color:
                                    c.result.judge_score >= 8
                                      ? 'var(--green)'
                                      : c.result.judge_score >= 6
                                        ? 'var(--yellow)'
                                        : 'var(--red)',
                                }}
                              >
                                {c.result.judge_score.toFixed(1)}
                              </span>
                            ) : (
                              '—'
                            )}
                          </td>
                          <td>{casePass(c) ? <Tag color="green">PASS</Tag> : <Tag color="red">FAIL</Tag>}</td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
                <div style={{ marginTop: 10, fontSize: 11.5, color: 'var(--text-2)' }}>
                  生成时间：{golden.generated_at || '—'}
                  {golden.report_path && (
                    <>
                      {'　'}报告已持久化：<code>{golden.report_path}</code>
                    </>
                  )}
                </div>
              </Card>
            </>
          )}
        </>
      )}

      {tab === 'badcases' && (
        <>
          {summary && (
            <div className="stats-grid">
              {Object.entries(summary).slice(0, 6).map(([k, v]) => (
                <StatCard key={k} label={k} value={String(v)} />
              ))}
            </div>
          )}
          <Card title="BadCase 库" hint="评测未通过的用例自动入库，回流自进化">
            {badCases === null ? (
              <Loading />
            ) : badCases.total === 0 ? (
              <Empty text="暂无 BadCase 🎉" />
            ) : (
              <table>
                <thead>
                  <tr>
                    <th>评测 ID</th>
                    <th>Query</th>
                    <th>得分</th>
                  </tr>
                </thead>
                <tbody>
                  {(badCases.cases as { evaluation_id: string; query: string; overall_score: number }[]).map((c) => (
                    <tr key={c.evaluation_id}>
                      <td>
                        <code style={{ fontSize: 11 }}>{c.evaluation_id}</code>
                      </td>
                      <td>{c.query}</td>
                      <td>
                        <span style={{ color: 'var(--red)' }}>{c.overall_score?.toFixed(1)}</span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </Card>
        </>
      )}

      {tab === 'single' && (
        <Card title="单条 LLM-as-Judge 评测" hint="输入 query + response，按四维度打分">
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            <textarea
              className="textarea"
              placeholder="Query（用户问题）"
              value={sq}
              onChange={(e) => setSq(e.target.value)}
            />
            <textarea
              className="textarea"
              placeholder="Response（Agent 回答）"
              value={sr}
              onChange={(e) => setSr(e.target.value)}
            />
            <textarea
              className="textarea"
              placeholder="Context（可选，参考上下文）"
              value={sc}
              onChange={(e) => setSc(e.target.value)}
            />
            <div>
              <button className="btn primary" disabled={singleBusy} onClick={runSingle}>
                {singleBusy ? 'Judge 评分中…' : '开始评测'}
              </button>
            </div>
          </div>

          {singleResult && (
            <div style={{ marginTop: 16 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 12 }}>
                <span
                  style={{
                    fontSize: 26,
                    fontFamily: 'var(--mono)',
                    fontWeight: 700,
                    color: singleResult.passed ? 'var(--green)' : 'var(--red)',
                  }}
                >
                  {singleResult.overall_score.toFixed(1)}
                </span>
                {singleResult.passed ? <Tag color="green">PASS</Tag> : <Tag color="red">FAIL</Tag>}
                <span style={{ fontSize: 12, color: 'var(--text-2)' }}>满分 10</span>
              </div>
              <table>
                <thead>
                  <tr>
                    <th>维度</th>
                    <th>得分</th>
                    <th>评语</th>
                  </tr>
                </thead>
                <tbody>
                  {singleResult.dimensions.map((d) => (
                    <tr key={d.dimension}>
                      <td>{JUDGE_DIMS.find(([k]) => k === d.dimension)?.[1] || d.dimension}</td>
                      <td>
                        <span
                          style={{
                            color: d.score >= 8 ? 'var(--green)' : d.score >= 6 ? 'var(--yellow)' : 'var(--red)',
                            fontFamily: 'var(--mono)',
                          }}
                        >
                          {d.score.toFixed(1)}
                        </span>
                      </td>
                      <td style={{ color: 'var(--text-1)' }}>{d.comment || '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Card>
      )}
    </div>
  )
}
