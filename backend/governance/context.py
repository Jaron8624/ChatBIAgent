"""Only trusted code may install a resolved business domain on an execution."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field


@dataclass(frozen=True)
class BusinessContext:
    tenant: str = "default"
    user_id: int = 0
    domain: str = "default"
    project: str = ""
    team: str = ""
    role: str = "member"
    experience: str = "standard"
    policy: dict = field(default_factory=dict)
    policy_hash: str = ""
    signals: list = field(default_factory=list, compare=False)


_context = ContextVar("business_domain_context", default=None)


def current():
    return _context.get()


@contextmanager
def business_context(value):
    token = _context.set(value)
    try:
        yield value
    finally:
        _context.reset(token)


def permits(payload, ctx=None):
    ctx = ctx or current()
    # Unscoped legacy evolved assets are quarantined, not implicitly global.
    if (
        not ctx
        or payload.get("tenant_id") != ctx.tenant
        or payload.get("domain_id") != ctx.domain
    ):
        return False
    scope = payload.get("scope")
    return (
        scope == "global"
        or (scope == "team" and ctx.team and payload.get("team_id") == ctx.team)
        or (scope == "user" and str(payload.get("owner_id")) == str(ctx.user_id))
    )
