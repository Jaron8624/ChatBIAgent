import { useEffect, useRef, useState } from 'react'
import { api, getApiContext } from '../lib/api'
import { Card, Empty, Loading, Tag, fmtTime } from '../components/ui'

interface KbFile {
  file_hash: string
  file_name: string
  uploaded_at: string
}

export default function KnowledgeView() {
  const [kbs, setKbs] = useState<string[] | null>(null)
  const [error, setError] = useState('')
  const [newKbName, setNewKbName] = useState('')
  const [selectedKb, setSelectedKb] = useState<string | null>(null)
  const [files, setFiles] = useState<KbFile[] | null>(null)
  const [uploading, setUploading] = useState(false)
  const [uploadMsg, setUploadMsg] = useState('')
  const fileRef = useRef<HTMLInputElement>(null)

  const loadKbs = () => {
    api
      .get<{ knowledge_bases: string[] }>(`/knowledge-bases?user_id=${getApiContext().userId}`)
      .then((r) => { setKbs(r.knowledge_bases || []); setError('') })
      .catch((e) => setError((e as Error).message))
  }

  useEffect(loadKbs, [])

  const loadFiles = (kbId: string) => {
    setSelectedKb(kbId)
    setFiles(null)
    api
      .get<{ files: KbFile[] }>(
        `/knowledge-bases/${encodeURIComponent(kbId)}/files?user_id=${getApiContext().userId}`,
      )
      .then((r) => { setFiles(r.files || []); setError('') })
      .catch(() => setFiles([]))
  }

  const createKb = async () => {
    if (!newKbName.trim()) return
    try {
      await api.post(
        `/knowledge-bases?knowledge_base_id=${encodeURIComponent(newKbName.trim())}&user_id=${getApiContext().userId}`,
      )
      setNewKbName('')
      loadKbs()
    } catch (e) {
      alert((e as Error).message)
    }
  }

  const deleteKb = async (kbId: string) => {
    if (!confirm(`确定删除知识库「${kbId}」？该操作不可恢复。`)) return
    try {
      await api.del(
        `/knowledge-bases/${encodeURIComponent(kbId)}?user_id=${getApiContext().userId}`,
      )
      if (selectedKb === kbId) {
        setSelectedKb(null)
        setFiles(null)
      }
      loadKbs()
    } catch (e) {
      alert((e as Error).message)
    }
  }

  const upload = async () => {
    const f = fileRef.current?.files?.[0]
    if (!f || !selectedKb) return
    setUploading(true)
    setUploadMsg('')
    const form = new FormData()
    form.append('file', f)
    form.append('knowledge_base_id', selectedKb)
    form.append('user_id', String(getApiContext().userId))
    try {
      const res = await fetch('/api/document_upload', {
        method: 'POST',
        headers: {
          'X-User-Id': String(getApiContext().userId),
          'X-User-Role': getApiContext().role,
        },
        body: form,
      })
      const body = await res.json().catch(() => ({}))
      if (!res.ok) throw new Error(body.detail || res.statusText)
      setUploadMsg(`✅ ${f.name} 上传成功（后台 embedding 进行中）`)
      if (fileRef.current) fileRef.current.value = ''
      setTimeout(() => loadFiles(selectedKb), 1500)
    } catch (e) {
      setUploadMsg(`❌ 上传失败：${(e as Error).message}`)
    }
    setUploading(false)
  }

  const deleteFile = async (fileHash: string) => {
    if (!selectedKb || !confirm('确定删除该文件及其向量索引？')) return
    try {
      await api.del(
        `/knowledge-bases/${encodeURIComponent(selectedKb)}/files/${fileHash}?user_id=${getApiContext().userId}`,
      )
      loadFiles(selectedKb)
    } catch (e) {
      alert((e as Error).message)
    }
  }

  if (error) return <div className="error-text">加载失败：{error}</div>
  if (!kbs) return <Loading text="加载知识库" />

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1.4fr', gap: 16, alignItems: 'start' }}>
      {/* 左：知识库列表 */}
      <div>
        <Card title="知识库" hint={`${kbs.length} 个`}>
          <div style={{ display: 'flex', gap: 8, marginBottom: 14 }}>
            <input
              className="input"
              style={{ flex: 1 }}
              placeholder="新知识库名称"
              value={newKbName}
              onChange={(e) => setNewKbName(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && createKb()}
            />
            <button className="btn primary" onClick={createKb}>
              创建
            </button>
          </div>
          {kbs.length === 0 ? (
            <Empty text="暂无知识库" />
          ) : (
            kbs.map((kb) => (
              <div
                key={kb}
                onClick={() => loadFiles(kb)}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 8,
                  padding: '9px 12px',
                  borderRadius: 6,
                  cursor: 'pointer',
                  marginBottom: 4,
                  border: `1px solid ${selectedKb === kb ? 'rgba(37,99,235,0.35)' : 'transparent'}`,
                  background: selectedKb === kb ? 'rgba(37,99,235,0.06)' : 'transparent',
                }}
              >
                <span style={{ color: 'var(--purple)' }}>▥</span>
                <span style={{ fontSize: 13 }}>{kb}</span>
                <span style={{ flex: 1 }} />
                <button
                  className="btn small danger"
                  onClick={(e) => {
                    e.stopPropagation()
                    deleteKb(kb)
                  }}
                >
                  删除
                </button>
              </div>
            ))
          )}
        </Card>

        <Card title="父子分块 · 混合检索" hint="RAG 架构说明">
          <div style={{ fontSize: 12.5, color: 'var(--text-1)', lineHeight: 1.8 }}>
            <div>· <b>子块</b>（~512 token）用于精确检索命中</div>
            <div>· <b>父块</b>（~2048 token）作为返回上下文，保证语义完整</div>
            <div>· <b>Dense</b>（COSINE）+ <b>BM25</b> 稀疏向量双路检索</div>
            <div>· <b>RRF</b> 融合排序 + <b>LLM Reranker</b> 精排</div>
            <div>· 检索失败自动降级为单路稠密检索</div>
          </div>
        </Card>
      </div>

      {/* 右：文件管理 */}
      <Card
        title={selectedKb ? `文件 — ${selectedKb}` : '文件'}
        hint={selectedKb ? '点击左侧知识库查看' : ''}
      >
        {!selectedKb ? (
          <Empty text="← 先选择一个知识库" />
        ) : (
          <>
            <div style={{ display: 'flex', gap: 8, marginBottom: 14, alignItems: 'center' }}>
              <input ref={fileRef} type="file" accept=".pdf,.docx,.doc" style={{ fontSize: 12, color: 'var(--text-2)' }} />
              <button className="btn primary" disabled={uploading} onClick={upload}>
                {uploading ? '上传中…' : '上传'}
              </button>
            </div>
            {uploadMsg && <div style={{ fontSize: 12, marginBottom: 10 }}>{uploadMsg}</div>}
            {files === null ? (
              <Loading text="加载文件列表" />
            ) : files.length === 0 ? (
              <Empty text="知识库为空，上传 PDF/DOCX 文档" />
            ) : (
              <table>
                <thead>
                  <tr>
                    <th>文件名</th>
                    <th>上传时间</th>
                    <th style={{ width: 70 }}>操作</th>
                  </tr>
                </thead>
                <tbody>
                  {files.map((f) => (
                    <tr key={f.file_hash}>
                      <td>{f.file_name}</td>
                      <td style={{ color: 'var(--text-2)' }}>{fmtTime(f.uploaded_at)}</td>
                      <td>
                        <button className="btn small danger" onClick={() => deleteFile(f.file_hash)}>
                          删除
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </>
        )}
        {selectedKb && files && files.length > 0 && (
          <div style={{ marginTop: 12 }}>
            <Tag color="purple">Milvus 向量索引</Tag>{' '}
            <Tag color="blue">父子分块</Tag>{' '}
            <Tag color="green">去重上传</Tag>
          </div>
        )}
      </Card>
    </div>
  )
}
