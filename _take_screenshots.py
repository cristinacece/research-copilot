# -*- coding: utf-8 -*-
"""Take screenshots of all 4 app pages using Playwright."""
import asyncio, os, sys
sys.stdout.reconfigure(encoding="utf-8")

from playwright.async_api import async_playwright

BASE_URL  = "http://localhost:8502"
SHOTS_DIR = r"C:\CELESTE\QLAB\trabajo\demo\screenshots"
os.makedirs(SHOTS_DIR, exist_ok=True)

# (sidebar label, click text subset, output filename)
PAGES = [
    ("Chat",          "Chat",          "01_chat.png"),
    ("Paper Browser", "Paper Browser", "02_browser.png"),
    ("Analytics",     "Analytics",     "03_analytics.png"),
    ("Configuracion", "Configuraci",   "04_settings.png"),
]

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page    = await browser.new_page(viewport={"width": 1400, "height": 900})

        print("Loading app...", flush=True)
        await page.goto(BASE_URL, wait_until="networkidle", timeout=30000)
        await page.wait_for_timeout(4000)

        # screenshot initial state (Chat page)
        out0 = os.path.join(SHOTS_DIR, "01_chat.png")
        await page.screenshot(path=out0, full_page=True)
        print(f"Saved: {out0}", flush=True)

        for name, click_text, filename in PAGES[1:]:
            print(f"Navigating to {name}...", flush=True)
            try:
                # find and click radio option containing the text
                await page.locator(f"label:has-text('{click_text}')").first.click(timeout=8000)
            except Exception as e:
                print(f"  click error: {e}", flush=True)
                try:
                    await page.get_by_text(click_text, exact=False).first.click(timeout=5000)
                except Exception as e2:
                    print(f"  fallback error: {e2}", flush=True)

            await page.wait_for_timeout(4000)
            out = os.path.join(SHOTS_DIR, filename)
            await page.screenshot(path=out, full_page=True)
            print(f"Saved: {out}", flush=True)

        await browser.close()
    print("All screenshots done.", flush=True)

asyncio.run(main())
