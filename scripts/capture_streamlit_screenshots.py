from pathlib import Path

from playwright.sync_api import sync_playwright


BASE_URL = "http://localhost:8501"
OUT_DIR = Path("evidence_screenshots")
REPORT_HTML = Path("LOCAL_SCREENSHOT_REPORT_Group44.html")
REPORT_PDF = Path("LOCAL_SCREENSHOT_REPORT_Group44.pdf")

TABS = [
    "Dashboard",
    "Crawling",
    "Text Mining",
    "Index",
    "Search",
    "Ranking",
    "Recommendations",
    "Evaluation",
    "Analytics",
]


def safe_name(name: str) -> str:
    return name.lower().replace(" ", "_")


def launch_browser(playwright):
    for kwargs in (
        {"headless": True},
        {"headless": True, "channel": "msedge"},
        {"headless": True, "channel": "chrome"},
    ):
        try:
            return playwright.chromium.launch(**kwargs)
        except Exception:
            continue
    raise RuntimeError("Could not launch Chromium, Edge, or Chrome through Playwright.")


def capture() -> list[dict]:
    OUT_DIR.mkdir(exist_ok=True)
    captures = []
    with sync_playwright() as p:
        browser = launch_browser(p)
        page = browser.new_page(viewport={"width": 1440, "height": 1150}, device_scale_factor=1)
        page.goto(BASE_URL, wait_until="networkidle", timeout=60000)
        page.wait_for_timeout(2500)

        for index, tab_name in enumerate(TABS, start=1):
            page.get_by_role("tab", name=tab_name).click()
            page.wait_for_timeout(1800)
            path = OUT_DIR / f"{index:02d}_{safe_name(tab_name)}.png"
            page.screenshot(path=str(path), full_page=True)
            captures.append({"tab": tab_name, "path": path.as_posix()})

        make_html_report(captures)
        report_page = browser.new_page(viewport={"width": 1440, "height": 1150})
        report_page.goto(REPORT_HTML.resolve().as_uri(), wait_until="networkidle")
        report_page.pdf(path=str(REPORT_PDF), format="A4", print_background=True)
        browser.close()
    return captures


def make_html_report(captures: list[dict]) -> None:
    cards = []
    for item in captures:
        cards.append(
            f"""
            <section class="shot">
                <h2>{item["tab"]}</h2>
                <p>Local Streamlit evidence for the {item["tab"]} tab.</p>
                <img src="{item["path"]}" alt="{item["tab"]} screenshot">
            </section>
            """
        )

    REPORT_HTML.write_text(
        f"""<!doctype html>
<html>
<head>
    <meta charset="utf-8">
    <title>Group 44 Local Screenshot Evidence</title>
    <style>
        body {{
            margin: 0;
            font-family: Arial, Helvetica, sans-serif;
            color: #17212b;
            background: #f6f8fb;
        }}
        header {{
            padding: 34px 42px;
            background: linear-gradient(135deg, #e3f3f1, #eef4ff);
            border-bottom: 1px solid #dce3e9;
        }}
        h1 {{
            margin: 0 0 8px;
            font-size: 30px;
        }}
        .meta {{
            color: #435166;
            line-height: 1.5;
        }}
        main {{
            padding: 24px 42px 42px;
        }}
        .summary {{
            background: #ffffff;
            border: 1px solid #dce3e9;
            border-radius: 8px;
            padding: 16px 18px;
            margin-bottom: 22px;
        }}
        .shot {{
            page-break-inside: avoid;
            background: #ffffff;
            border: 1px solid #dce3e9;
            border-radius: 8px;
            padding: 18px;
            margin: 0 0 24px;
        }}
        .shot h2 {{
            margin: 0 0 4px;
            font-size: 22px;
        }}
        .shot p {{
            margin: 0 0 14px;
            color: #5b6674;
        }}
        img {{
            width: 100%;
            border: 1px solid #cfd8e3;
            border-radius: 6px;
        }}
        code {{
            background: #eef4ff;
            padding: 2px 5px;
            border-radius: 4px;
        }}
    </style>
</head>
<body>
    <header>
        <h1>Group 44 Local Screenshot Evidence</h1>
        <div class="meta">
            Information Retrieval Assignment 2<br>
            Application: Group 44 HealthIR Workbench<br>
            Local URL captured: <code>{BASE_URL}</code>
        </div>
    </header>
    <main>
        <div class="summary">
            This report records local execution evidence for every Streamlit tab required by the assignment document:
            dashboard, crawling interface, text mining, index management, search, ranking visualization,
            recommendations, evaluation dashboard, and performance analytics.
        </div>
        {''.join(cards)}
    </main>
</body>
</html>
""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    captures = capture()
    for item in captures:
        print(f"{item['tab']}: {item['path']}")
    print(f"HTML report: {REPORT_HTML}")
    print(f"PDF report: {REPORT_PDF}")
