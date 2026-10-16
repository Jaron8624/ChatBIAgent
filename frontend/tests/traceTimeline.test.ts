import { test } from 'node:test'
import assert from 'node:assert/strict'
import { buildTraceTimeline, type TraceSpan } from '../src/lib/traceTimeline.ts'

const span = (id: string, start: number, duration: number, parentId?: string): TraceSpan => ({
  traceId: '1'.repeat(32), spanId: id, parentId, operationName: id, startTime: start,
  duration, status: 'ok', tags: {},
})

test('nested and parallel spans retain start offsets and wall-clock duration', () => {
  const tree = buildTraceTimeline([span('tool', 1050, 20, 'turn'), span('root', 1000, 100),
    span('turn', 1010, 70, 'root'), span('parallel', 1020, 60, 'root')])
  assert.equal(tree.duration, 100)
  assert.deepEqual(tree.rows.map((row) => [row.span.spanId, row.depth, row.offsetPercent]),
    [['root', 0, 0], ['turn', 1, 10], ['tool', 2, 50], ['parallel', 1, 20]])
})

test('cycle, orphan, invalid and duplicate spans are bounded and safe', () => {
  const tree = buildTraceTimeline([span('a', 0, 0, 'b'), span('b', 0, 0, 'a'),
    span('orphan', 0, 1, 'missing'), span('a', 0, 4), span('invalid', NaN, 2)])
  assert.equal(tree.rows.length, 3)
  assert(tree.rows.every((row) => Number.isFinite(row.widthPercent)))
})

test('empty and zero-duration traces have no invalid geometry', () => {
  assert.deepEqual(buildTraceTimeline([]), { rows: [], duration: 0 })
  assert.equal(buildTraceTimeline([span('zero', 1, 0)]).rows[0].offsetPercent, 0)
})
