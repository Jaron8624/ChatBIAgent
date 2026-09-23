#!/usr/bin/env python3
"""为面试项目文档生成 15 张主题配图（1440×810）。"""
import asyncio
from pathlib import Path
from playwright.async_api import async_playwright

BASE = Path(__file__).resolve().parents[1]
HTML = BASE / "docs/面试项目文档/assets/diagram-preview.html"
OUT_DIR = BASE / "docs/面试项目文档/assets"

NAMES = {
    "d01": "01-intent-routing.png",
    "d02": "02-mcp-tool-lifecycle.png",
    "d03": "03-react-evidence-loop.png",
    "d04": "04-multi-agent-parallel.png",
    "d05": "05-context-lazy-loading.png",
    "d06": "06-hybrid-retrieval.png",
    "d07": "07-memory-layers.png",
    "d08": "08-routing-pyramid.png",
    "d09": "09-observability-gates.png",
    "d10": "10-evaluation-loop.png",
    "d11": "11-sql-guard.png",
    "d12": "12-rbac-audit.png",
    "d13": "13-badcase-loop.png",
    "d14": "14-session-recovery.png",
    "d15": "15-deployment-topology.png",
}

async def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": 1440, "height": 810})
        url = HTML.resolve().as_uri()
        for cid, fname in NAMES.items():
            await page.goto(url, wait_until="networkidle")
            await page.evaluate(
                """(cid) => {
                    document.querySelectorAll('.card').forEach(c => c.style.display = 'none');
                    const el = document.getElementById(cid);
                    if (el) {
                        el.style.display = 'block';
                        el.style.position = 'absolute';
                        el.style.top = '0';
                        el.style.left = '0';
                        el.style.margin = '0';
                        el.style.padding = '58px 70px';
                    }
                }""",
                cid,
            )
            await page.wait_for_timeout(500)
            out = OUT_DIR / fname
            await page.screenshot(path=str(out), full_page=False)
            print(f"Saved {out}")
        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
