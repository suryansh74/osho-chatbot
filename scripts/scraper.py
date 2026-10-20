"""
Full transcript scraper for oshoworld.com (English + Hindi).

Saves clean transcripts into:
  data/english/<series_slug>/<discourse>.json
  data/hindi/<series_slug>/<discourse>.json

Each JSON contains:
  - title, url, language, transcript, series
  - duration (HH:MM:SS or MM:SS when available)
  - char_count, word_count (for later timestamp estimation)

Resumable. Uses Playwright only for discovery of series/discourse lists,
then requests + BeautifulSoup for the actual transcript pages.

Usage:
  python scripts/scraper.py --output-dir data
"""

import argparse
import json
import re
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from tqdm import tqdm

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

SERIES_PAGES = {
    "english": "https://oshoworld.com/audio-series-home-english",
    "hindi": "https://oshoworld.com/audio-series-home-hindi",
}


def get_all_series(page, language: str) -> list[dict]:
    base = SERIES_PAGES[language]
    series = []
    seen = set()
    page.goto(base, wait_until="networkidle", timeout=90000)
    page.wait_for_timeout(3000)

    page_num = 1
    while page_num <= 25:
        print(f"  {language} series page {page_num}")
        links = page.evaluate("""() => {
            const out = [];
            document.querySelectorAll('a[href]').forEach(a => {
                const text = (a.innerText || '').trim();
                const href = a.href;
                if (!href || !text) return;
                if (href.includes('oshoworld.com') && 
                    (text.toLowerCase().includes('view all') || 
                     /\\d+-\\d+/.test(href.split('/').pop() || ''))) {
                    out.push({title: text, href});
                }
            });
            return out;
        }""")

        for item in links:
            href = item["href"].split("#")[0]
            if href in seen or "home" in href:
                continue
            if "view all" in item["title"].lower() or re.search(r"-\d+-\d+$", href.split("/")[-1]):
                series.append({
                    "title": item["title"].replace("View all", "").strip() or href.split("/")[-1],
                    "url": href,
                    "language": language,
                })
                seen.add(href)

        # Try next page
        next_clicked = page.evaluate("""() => {
            const next = [...document.querySelectorAll('a, button')].find(el => 
                /next/i.test(el.innerText || '') && el.offsetParent !== null
            );
            if (next) { next.click(); return true; }
            return false;
        }""")
        if not next_clicked:
            break
        page.wait_for_timeout(2500)
        page_num += 1

    # Deduplicate
    unique = []
    seen_u = set()
    for s in series:
        if s["url"] not in seen_u:
            unique.append(s)
            seen_u.add(s["url"])
    return unique


def get_discourse_urls(page, series_url: str) -> list[dict]:
    """Return list of {url, duration} for discourses in a series."""
    page.goto(series_url, wait_until="networkidle", timeout=60000)
    page.wait_for_timeout(2000)
    items = page.evaluate("""() => {
        const out = [];
        // Look for table rows or list items that contain both a link and a duration
        document.querySelectorAll('a[href]').forEach(a => {
            const href = a.href;
            if (!href || href.includes('#')) return;
            const slug = href.split('/').pop() || '';
            if (!/\\-\\d{1,3}$/.test(slug)) return;

            // Look nearby for duration like [01:42:15] or 01:42:15
            let duration = null;
            let el = a;
            for (let i = 0; i < 5 && el; i++) {
                const text = el.innerText || '';
                const m = text.match(/\\[?(\\d{1,2}:\\d{2}:\\d{2}|\\d{1,2}:\\d{2})\\]?/);
                if (m) {
                    duration = m[1];
                    break;
                }
                el = el.parentElement;
            }
            out.push({url: href.split('#')[0], duration});
        });
        // Deduplicate by url
        const seen = new Set();
        return out.filter(item => {
            if (seen.has(item.url)) return false;
            seen.add(item.url);
            return true;
        });
    }""")
    return items


def clean_transcript(text: str) -> str:
    """Remove navigation, repeated blocks, and junk while keeping the real discourse."""
    if not text:
        return ""

    # Common junk markers
    junk_patterns = [
        r"Home\s*/\s*Discourses.*",
        r"UP NEXT.*",
        r"SHOW MORE.*",
        r"Language\s*:\s*english.*",
        r"Share\s*Download.*",
        r"Contact Us.*",
        r"Privacy Policy.*",
        r"Cookie.*",
        r"Upcoming Events.*",
        r"View all.*",
        r"osho dham.*",
        r"Login.*",
    ]
    for pat in junk_patterns:
        text = re.sub(pat, "", text, flags=re.IGNORECASE | re.DOTALL)

    # Remove very short repeated lines and collapse whitespace
    lines = [ln.strip() for ln in text.splitlines() if len(ln.strip()) > 40]
    # Deduplicate consecutive identical lines
    cleaned_lines = []
    prev = None
    for ln in lines:
        if ln != prev:
            cleaned_lines.append(ln)
            prev = ln

    result = "\n\n".join(cleaned_lines)
    # Prefer content that starts with typical openings if present
    openings = ["BELOVED OSHO", "BELOVED MASTER", "प्रिय ओशो", "प्रिय मास्टर"]
    for op in openings:
        idx = result.upper().find(op.upper())
        if idx != -1 and idx < 500:
            result = result[idx:]
            break

    return result.strip()


def extract_transcript(soup: BeautifulSoup) -> str:
    """Extract the main discourse text more carefully."""
    # Try to find the largest continuous text block
    candidates = []
    for tag in soup.find_all(["p", "div", "article", "section", "main"]):
        text = tag.get_text("\n", strip=True)
        if len(text) < 300:
            continue
        lower = text[:400].lower()
        if any(skip in lower for skip in [
            "home\n", "login", "share", "contact us", "privacy policy",
            "cookie", "osho dham", "upcoming events", "view all", "up next"
        ]):
            continue
        candidates.append(text)

    if not candidates:
        body = soup.get_text("\n", strip=True)
        parts = [p.strip() for p in re.split(r"\n\s*\n", body) if len(p.strip()) > 400]
        raw = "\n\n".join(parts)
    else:
        candidates.sort(key=len, reverse=True)
        raw = "\n\n".join(candidates[:12])

    return clean_transcript(raw)


def scrape_one(url: str, language: str) -> dict | None:
    try:
        r = requests.get(url, headers=HEADERS, timeout=40)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")

        title_el = soup.find("h1") or soup.find("title")
        title = title_el.get_text(strip=True) if title_el else url.split("/")[-1]
        title = re.sub(r"\s+", " ", title).strip()

        transcript = extract_transcript(soup)
        if len(transcript) < 400:
            return None

        # Duration extraction
        duration = None
        full_text = soup.get_text()
        dur_match = re.search(r"(\d{1,2}:\d{2}:\d{2}|\d{1,2}:\d{2})", full_text)
        if dur_match:
            duration = dur_match.group(1)

        char_count = len(transcript)
        word_count = len(transcript.split())

        return {
            "title": title,
            "url": url,
            "language": language,
            "transcript": transcript,
            "duration": duration,
            "char_count": char_count,
            "word_count": word_count,
        }
    except Exception as e:
        print(f"  Failed {url}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--max-series", type=int, default=None)
    parser.add_argument("--languages", nargs="+", default=["english", "hindi"],
                        help="Languages to scrape. Default is both english and hindi.")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        for lang in args.languages:
            print(f"\n=== Discovering {lang} series ===")
            series_list = get_all_series(page, lang)
            print(f"Found {len(series_list)} series")
            if args.max_series:
                series_list = series_list[: args.max_series]

            lang_dir = out / lang
            lang_dir.mkdir(exist_ok=True)

            for series in tqdm(series_list, desc=lang):
                series_slug = re.sub(r"[^a-zA-Z0-9_-]", "_", series["url"].split("/")[-1])[:90]
                series_path = lang_dir / series_slug
                series_path.mkdir(exist_ok=True)

                discourse_items = get_discourse_urls(page, series["url"])
                for item in discourse_items:
                    d_url = item["url"]
                    duration_from_list = item.get("duration")
                    slug = re.sub(r"[^a-zA-Z0-9_-]", "_", d_url.split("/")[-1])[:120]
                    json_path = series_path / f"{slug}.json"
                    if json_path.exists():
                        # Optionally update duration if missing
                        try:
                            existing = json.loads(json_path.read_text(encoding="utf-8"))
                            if not existing.get("duration") and duration_from_list:
                                existing["duration"] = duration_from_list
                                json_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
                        except Exception:
                            pass
                        continue

                    data = scrape_one(d_url, lang)
                    if data:
                        data["series"] = series["title"]
                        if duration_from_list and not data.get("duration"):
                            data["duration"] = duration_from_list
                        with open(json_path, "w", encoding="utf-8") as f:
                            json.dump(data, f, ensure_ascii=False, indent=2)
                    time.sleep(0.5)

        browser.close()

    print("\nScraping finished.")
    print(f"English transcripts: {out / 'english'}")
    print(f"Hindi transcripts:   {out / 'hindi'}")


if __name__ == "__main__":
    main()
