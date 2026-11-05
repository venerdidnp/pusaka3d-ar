"""Automated, headless-browser verification of the Pusaka3D AR Explainer.

This is *not* a substitute for testing the actual AR session (WebXR /
Android Scene Viewer / iOS Quick Look) on a real phone -- that handoff to
the OS-level AR viewer cannot be driven from a headless browser and must be
checked on real hardware (see README.md "Known deviations"). What this
script *does* verify, in a real Chromium engine via Playwright, is
everything up to that handoff: the page loads, <model-viewer> parses and
renders the GLB (a real WebGL canvas, not a stub), all hotspots from
hotspots.json are injected at the right slots, clicking one opens the info
panel with the right bilingual text, and the language/quiz-mode controls
work.

Requires: `pip install playwright && python -m playwright install chromium`
(a heavier, optional dependency -- see requirements.txt). Requires the app
to be served over http(s), not opened as a file:// URL (the hotspot fetch()
of hotspots.json needs a real origin) -- start one with
`python -m http.server 8743` from this directory first.

Run: python verify_ar_app.py [base_url]   (default http://localhost:8743)
"""
from __future__ import annotations

import os
import sys

from playwright.sync_api import sync_playwright

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8743"
HERE = os.path.dirname(os.path.abspath(__file__))
SCREENSHOT_DIR = os.path.join(HERE, "screenshots")

_checks = 0


def check(name, condition):
    global _checks
    status = "ok" if condition else "FAIL"
    print(f"[{status}] {name}")
    assert condition, f"failed: {name}"
    _checks += 1


def main():
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)
    page_errors = []

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chromium", args=["--use-gl=swiftshader", "--enable-webgl", "--ignore-gpu-blocklist"])
        page = browser.new_page(viewport={"width": 390, "height": 800})
        page.on("pageerror", lambda exc: page_errors.append(str(exc)))
        page.goto(f"{BASE_URL}/index.html", wait_until="networkidle", timeout=30000)
        page.wait_for_timeout(2500)

        mv = page.evaluate("""
            () => {
                const el = document.querySelector('model-viewer');
                return {
                    exists: !!el, loaded: el ? el.loaded : false,
                    modelIsVisible: el ? el.modelIsVisible : false,
                    hasCanvas: !!(el && el.shadowRoot && el.shadowRoot.querySelector('canvas')),
                };
            }
        """)
        check("model-viewer element exists", mv["exists"])
        check("model reported loaded", mv["loaded"])
        check("model reported visible (rendered)", mv["modelIsVisible"])
        check("a real WebGL <canvas> was created inside model-viewer", mv["hasCanvas"])

        hotspot_count = page.evaluate("document.querySelectorAll('.hotspot').length")
        check(f"5 hotspots injected from hotspots.json (found {hotspot_count})", hotspot_count == 5)
        check("AR launch button present", page.evaluate("!!document.getElementById('ar-button')"))

        page.click("button[slot='hotspot-stupa']")
        page.wait_for_timeout(300)
        title = page.inner_text("#info-title")
        check(f"clicking a hotspot opens the info panel (title='{title}')",
              not page.evaluate("document.getElementById('info-panel').hidden") and len(title) > 0)

        page.screenshot(path=os.path.join(SCREENSHOT_DIR, "m5_ar_infopanel_en.png"))
        page.click("#lang-toggle")
        page.click("button[slot='hotspot-stupa']")
        page.wait_for_timeout(300)
        title_id = page.inner_text("#info-title")
        check(f"language toggle changes the panel text ('{title}' -> '{title_id}')", title != title_id)

        page.screenshot(path=os.path.join(SCREENSHOT_DIR, "m5_ar_infopanel_id.png"))
        page.click("#info-close")
        page.screenshot(path=os.path.join(SCREENSHOT_DIR, "m5_ar_overview.png"))
        page.click("#quiz-toggle")
        page.wait_for_timeout(300)
        quiz_target = page.inner_text("#quiz-target")
        check(f"quiz mode picks a target ('{quiz_target}')",
              not page.evaluate("document.getElementById('quiz-banner').hidden") and len(quiz_target) > 0)

        page.screenshot(path=os.path.join(SCREENSHOT_DIR, "m5_ar_quizmode.png"))
        check("no uncaught JS errors on the page", len(page_errors) == 0)

        browser.close()

    print(f"\nALL CHECKS PASSED ({_checks} ok)")
    print("NOTE: this verifies everything up to the AR launch. The actual WebXR/"
          "Scene Viewer/Quick Look handoff needs a real phone -- see README.md.")


if __name__ == "__main__":
    main()
