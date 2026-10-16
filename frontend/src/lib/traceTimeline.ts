export interface TraceSpan {
  traceId: string
  spanId: string
  parentId?: string | null
  operationName: string
  startTime: number
  duration: number
  tags: Record<string, unknown>
  status: string
  error?: string | null
}

export interface TraceRow {
  span: TraceSpan
  depth: number
  offsetPercent: number
  widthPercent: number
}

/** Preserve nesting and actual start offsets; never sum overlapping durations. */
export function buildTraceTimeline(input: TraceSpan[]): { rows: TraceRow[]; duration: number } {
  const seen = new Set<string>()
  const spans = input.slice(0, 2048).filter((span) => {
    if (!span.spanId || seen.has(span.spanId) || !Number.isFinite(span.startTime) ||
        !Number.isFinite(span.duration) || span.duration < 0) return false
    seen.add(span.spanId)
    return true
  })
  if (!spans.length) return { rows: [], duration: 0 }
  const start = Math.min(...spans.map((span) => span.startTime))
  const duration = Math.max(0, ...spans.map((span) => span.startTime + span.duration - start))
  const ordered = [...spans].sort((a, b) => a.startTime - b.startTime || a.spanId.localeCompare(b.spanId))
  const children = new Map<string, TraceSpan[]>()
  for (const span of ordered) {
    const parent = span.parentId && seen.has(span.parentId) ? span.parentId : ''
    children.set(parent, [...(children.get(parent) || []), span])
  }
  const visited = new Set<string>()
  const rows: TraceRow[] = []
  const walk = (span: TraceSpan, depth: number) => {
    if (visited.has(span.spanId)) return
    visited.add(span.spanId)
    const offsetPercent = duration ? Math.max(0, (span.startTime - start) / duration * 100) : 0
    rows.push({ span, depth, offsetPercent,
      widthPercent: duration ? Math.min(100 - offsetPercent, span.duration / duration * 100) : 0 })
    for (const child of children.get(span.spanId) || []) walk(child, depth + 1)
  }
  for (const root of children.get('') || []) walk(root, 0)
  // Malformed/cyclic legacy trees must not hang or disappear from the viewer.
  for (const span of ordered) walk(span, 0)
  return { rows, duration }
}
