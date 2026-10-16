import { useState } from 'react'
import { buildTraceTimeline, type TraceSpan } from '../lib/traceTimeline'

export function TraceTimeline({ spans }: { spans: TraceSpan[] }) {
  const [open, setOpen] = useState(false)
  const { rows, duration } = buildTraceTimeline(spans)
  if (!rows.length) return null
  const root = rows[0].span
  const firstAnswer = root.tags?.['request.ttft_ms']
  return (
    <section style={{ marginTop: 10, border: '1px solid var(--border)', borderRadius: 6, background: '#fff' }}>
      <button type="button" aria-expanded={open} onClick={() => setOpen(!open)}
        style={{ display: 'flex', gap: 12, width: '100%', padding: '10px 12px', border: 0,
          background: '#f7f8fa', color: 'var(--text-0)', cursor: 'pointer', textAlign: 'left' }}>
        <strong>请求耗时追踪</strong>
        <span>{rows.length} spans · {duration.toFixed(0)} ms</span>
        {typeof firstAnswer === 'number' && <span>首段回答 {firstAnswer.toFixed(0)} ms</span>}
        <span style={{ marginLeft: 'auto' }}>{open ? '收起' : '展开'}</span>
      </button>
      {open && <div style={{ padding: 12, overflowX: 'auto' }}>
        <div style={{ fontSize: 11, color: 'var(--text-2)', marginBottom: 10 }}>Trace ID {root.traceId}</div>
        <div style={{ minWidth: 600 }}>
          <div style={{ display: 'flex', paddingLeft: 210, paddingRight: 76, marginBottom: 6, justifyContent: 'space-between', fontSize: 10 }}>
            <span>0 ms</span><span>{(duration / 2).toFixed(0)} ms</span><span>{duration.toFixed(0)} ms</span>
          </div>
          {rows.map(({ span, depth, offsetPercent, widthPercent }) => (
            <div key={span.spanId} title={`${span.operationName} · ${span.tags?.layer || ''} · ${span.status}${span.error ? ` · ${span.error}` : ''}`}
              style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 7, fontSize: 11 }}>
              <div style={{ width: 200, paddingLeft: Math.min(depth, 8) * 12, flexShrink: 0,
                overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {depth > 0 ? '└ ' : ''}{span.operationName}
              </div>
              <div style={{ flex: 1, height: 14, background: '#f0f1f3', position: 'relative' }}>
                <div style={{ position: 'absolute', left: `${offsetPercent}%`, width: `${widthPercent}%`,
                  minWidth: widthPercent ? 1 : 0, height: '100%', borderRadius: 2,
                  background: span.status !== 'ok' ? '#c24141' : span.tags?.layer === 'L3' ? '#7755aa' : '#356bc5' }} />
              </div>
              <div style={{ width: 66, textAlign: 'right', flexShrink: 0 }}>{span.duration.toFixed(1)} ms</div>
            </div>
          ))}
        </div>
      </div>}
    </section>
  )
}
