"""Screenshot docs/index.html with Playwright (verifies page renders)."""
from playwright.sync_api import sync_playwright
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
url = "file://" + os.path.join(ROOT, "docs", "index.html")
out = os.path.join(ROOT, "docs", "assets", "preview.png")
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1280, "height": 900})
    pg.goto(url)
    pg.wait_for_timeout(1200)
    pg.screenshot(path=out, full_page=False)
    # quiz smoke: click first correct option (index 1 for Q1)
    pg.click("#qo .opt >> nth=1")
    pg.wait_for_timeout(400)
    txt = pg.inner_text("#qf")
    assert "Correct" in txt, txt
    pg.screenshot(path=os.path.join(ROOT, "docs", "assets", "preview_quiz.png"))
    b.close()
print("screenshots ok:", out)
