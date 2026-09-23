import { useEffect, useRef, useState } from 'react'
import { api, getApiContext, setApiContext } from './lib/api'
import ChatView from './views/ChatView'
import McpView from './views/McpView'
import KnowledgeView from './views/KnowledgeView'
import EvalView from './views/EvalView'
import MonitorView from './views/MonitorView'
import EvolutionView from './views/EvolutionView'
import SkillsView from './views/SkillsView'
import AuditView from './views/AuditView'
import OverviewView from './views/OverviewView'
import SessionsView from './views/SessionsView'

type ViewId =
  | 'chat'
  | 'overview'
  | 'sessions'
  | 'mcp'
  | 'knowledge'
  | 'eval'
  | 'monitor'
  | 'evolution'
  | 'skills'
  | 'audit'

const CONSOLE_NAV: { id: ViewId; label: string; icon: string }[] = [
  { id: 'overview', label: '概览', icon: '◎' },
  { id: 'mcp', label: 'MCP 工具', icon: '⚙' },
  { id: 'eval', label: '评测中心', icon: '◈' },
  { id: 'monitor', label: '监控', icon: '◔' },
  { id: 'evolution', label: '自进化', icon: '↻' },
  { id: 'audit', label: '审计', icon: '◐' },
]

const VIEW_TITLE: Record<ViewId, string> = {
  chat: '对话',
  overview: '概览',
  sessions: '会话',
  mcp: 'MCP 工具',
  knowledge: '知识库',
  eval: '评测中心',
  monitor: '监控',
  evolution: '自进化',
  skills: 'Skills',
  audit: '审计',
}

const ROLES = ['admin', 'operator', 'viewer', 'user', 'guest']

export default function App() {
  const [view, setView] = useState<ViewId>('chat')
  const [chatResetKey, setChatResetKey] = useState(0)
  const [connected, setConnected] = useState<'ok' | 'bad' | 'pending'>('pending')
  const [role, setRole] = useState(getApiContext().role)
  const [userId, setUserId] = useState(getApiContext().userId)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [globalSearch, setGlobalSearch] = useState('')
  const searchRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    let alive = true
    const probe = async () => {
      try {
        await api.get<{ tools: unknown[] }>('/mcp/tools')
        if (alive) setConnected('ok')
      } catch {
        if (alive) setConnected('bad')
      }
    }
    probe()
    const timer = setInterval(probe, 30_000)
    return () => {
      alive = false
      clearInterval(timer)
    }
  }, [])

  // ⌘K 聚焦全局搜索
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault()
        searchRef.current?.focus()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  const newChat = () => {
    setChatResetKey((k) => k + 1)
    setView('chat')
  }

  const goSearch = () => {
    if (globalSearch.trim()) {
      setView('sessions')
      setGlobalSearch('')
    }
  }

  return (
    <div className="app-shell">
      {/* ======== 侧边栏（AIDE 风格） ======== */}
      <aside className="sidebar">
        <div className="sidebar-logo" style={{ borderBottom: 'none', paddingBottom: 8 }}>
          <div className="logo-mark">N</div>
          <div className="title" style={{ fontWeight: 700 }}>NexusMind</div>
        </div>

        <nav className="sidebar-nav" style={{ paddingTop: 4 }}>
          {/* 主操作 */}
          <div className={`nav-item ${view === 'chat' && chatResetKey >= 0 ? '' : ''}`} onClick={newChat}>
            <span className="icon">✎</span>
            <span>新会话</span>
          </div>
          <div className={`nav-item ${view === 'sessions' ? 'active' : ''}`} onClick={() => setView('sessions')}>
            <span className="icon">🔍</span>
            <span>搜索会话</span>
          </div>
          <div className={`nav-item ${view === 'knowledge' ? 'active' : ''}`} onClick={() => setView('knowledge')}>
            <span className="icon">📖</span>
            <span>知识库</span>
          </div>
          <div className={`nav-item ${view === 'skills' ? 'active' : ''}`} onClick={() => setView('skills')}>
            <span className="icon">👥</span>
            <span>专家 · Skills · MCP</span>
            <span className="badge" style={{ background: 'rgba(124,58,237,0.12)', color: 'var(--purple)', borderColor: 'rgba(124,58,237,0.3)' }}>new</span>
          </div>

          {/* 控制台 */}
          <div className="nav-group-label">控制台</div>
          {CONSOLE_NAV.map((item) => (
            <div
              key={item.id}
              className={`nav-item ${view === item.id ? 'active' : ''}`}
              onClick={() => setView(item.id)}
            >
              <span className="icon">{item.icon}</span>
              <span>{item.label}</span>
            </div>
          ))}

          {/* 会话 */}
          <div className="nav-group-label">会话</div>
          <div
            className={`nav-item ${view === 'chat' ? 'active' : ''}`}
            onClick={() => setView('chat')}
          >
            <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
              新的会话
            </span>
          </div>
        </nav>

        {/* 设置 */}
        <div className="sidebar-footer">
          <div
            className="nav-item"
            onClick={() => setSettingsOpen((o) => !o)}
            style={{ justifyContent: 'flex-start' }}
          >
            <span className="icon">⚙</span>
            <span>设置</span>
            <span style={{ marginLeft: 'auto', fontSize: 10, color: 'var(--text-2)' }}>
              {settingsOpen ? '▴' : '▾'}
            </span>
          </div>
          {settingsOpen && (
            <div style={{ padding: '8px 4px 2px' }}>
              <div className="role-switch">
                <span>身份</span>
                <select
                  value={role}
                  onChange={(e) => {
                    setRole(e.target.value)
                    setApiContext({ role: e.target.value })
                  }}
                >
                  {ROLES.map((r) => (
                    <option key={r} value={r}>
                      {r}
                    </option>
                  ))}
                </select>
              </div>
              <div className="role-switch" style={{ marginTop: 8 }}>
                <span>UID</span>
                <input
                  className="input"
                  style={{ flex: 1, padding: '5px 8px', fontSize: 12 }}
                  type="number"
                  min={0}
                  value={userId}
                  onChange={(e) => {
                    const v = Number(e.target.value) || 0
                    setUserId(v)
                    setApiContext({ userId: v })
                  }}
                />
              </div>
            </div>
          )}
        </div>
      </aside>

      {/* ======== 主区域 ======== */}
      <main className="main-area">
        <div className="topbar" style={{ justifyContent: 'center', position: 'relative' }}>
          {/* 非对话页时左上角显示标题 */}
          {view !== 'chat' && (
            <span style={{ position: 'absolute', left: 20, fontSize: 14, fontWeight: 600 }}>
              {VIEW_TITLE[view]}
            </span>
          )}
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 8,
              background: '#fff',
              border: '1px solid var(--border)',
              borderRadius: 8,
              padding: '6px 12px',
              width: 420,
              maxWidth: '46%',
              color: 'var(--text-2)',
            }}
          >
            <span style={{ fontSize: 13 }}>🔍</span>
            <input
              ref={searchRef}
              value={globalSearch}
              onChange={(e) => setGlobalSearch(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && goSearch()}
              placeholder="全域搜索数据资产、会话记录、工具…"
              style={{
                flex: 1,
                border: 'none',
                outline: 'none',
                fontSize: 12.5,
                background: 'transparent',
                color: 'var(--text-0)',
              }}
            />
            <span
              style={{
                fontSize: 10.5,
                border: '1px solid var(--border-light)',
                borderRadius: 4,
                padding: '1px 5px',
                color: 'var(--text-2)',
                flexShrink: 0,
              }}
            >
              ⌘K
            </span>
          </div>
          {/* 连接状态 */}
          <div className="conn-dot" style={{ position: 'absolute', right: 20 }}>
            <span className={`dot ${connected}`} />
          </div>
        </div>

        <div className="view-container" style={view === 'chat' ? { padding: 0, display: 'flex', flexDirection: 'column' } : undefined}>
          {view === 'chat' && <ChatView key={chatResetKey} />}
          {view === 'overview' && <OverviewView onNavigate={(v) => setView(v as ViewId)} />}
          {view === 'sessions' && <SessionsView />}
          {view === 'mcp' && <McpView />}
          {view === 'knowledge' && <KnowledgeView />}
          {view === 'eval' && <EvalView />}
          {view === 'monitor' && <MonitorView />}
          {view === 'evolution' && <EvolutionView />}
          {view === 'skills' && <SkillsView />}
          {view === 'audit' && <AuditView />}
        </div>
      </main>
    </div>
  )
}
