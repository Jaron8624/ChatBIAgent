#!/usr/bin/env python3
"""渲染一页纸概览 PNG（deviceScaleFactor=2）。"""
import asyncio
from pathlib import Path
from playwright.async_api import async_playwright

BASE = Path(__file__).resolve().parents[1]
HTML = BASE / "docs/diagrams/html/project-onepager.html"
OUT = BASE / "docs/images/项目一页纸概览.png"


async def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        ctx = await browser.new_context(device_scale_factor=2)
        page = await ctx.new_page()
        await page.set_viewport_size({"width": 820, "height": 1300})
        await page.goto(HTML.resolve().as_uri(), wait_until="networkidle")
        await page.wait_for_timeout(400)
        card = await page.query_selector(".card")
        await card.screenshot(path=str(OUT), omit_background=False)
        await browser.close()
    print("done:", OUT)


if __name__ == "__main__":
    asyncio.run(main())
