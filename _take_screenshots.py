# -*- coding: utf-8 -*-
"""Take screenshots of all app tabs using Playwright."""
import asyncio, os, sys
sys.stdout.reconfigure(encoding="utf-8")

from playwright.async_api import async_playwright

BASE_URL  = "http://localhost:8502"
SHOTS_DIR = r"C:\CELESTE\QLAB\trabajo\demo\screenshots"
os.makedirs(SHOTS_DIR, exist_ok=True)

# (tab button text substring, output filename)
TABS = [
    ("Chat",           "01_chat.png"),
    ("Paper Browser",  "02_browser.png"),
    ("Visualizaciones","03_analytics.png"),
]

async def click_tab(page, text: str):
    """Click a Streamlit tab by partial text match."""
    try:
        await page.locator(f"button[role='tab']:has-text('{text}')").first.click(timeout=8000)
        return True
    except Exception:
        pass
    try:
        await page.get_by_role("tab", name=text, exact=False).first.click(timeout=5000)
        return True
    except Exception:
        pass
    return False

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page    = await browser.new_page(viewport={"width": 1400, "height": 900})

        print("Loading app...", flush=True)
        await page.goto(BASE_URL, wait_until="networkidle", timeout=30000)
        await page.wait_for_timeout(5000)

        for tab_text, filename in TABS:
            print(f"Navigating to tab '{tab_text}'...", flush=True)
            ok = await click_tab(page, tab_text)
            if not ok:
                print(f"  Could not click tab '{tab_text}', capturing current state.", flush=True)
            await page.wait_for_timeout(3000)
            out = os.path.join(SHOTS_DIR, filename)
            await page.screenshot(path=out, full_page=True)
            print(f"  Saved: {out}", flush=True)

        await browser.close()
    print("All screenshots done.", flush=True)

asyncio.run(main())
