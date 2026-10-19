"""
Comprehensive Osho World scraper.
Discovers all English & Hindi series, extracts discourse URLs,
scrapes transcripts + audio links, and downloads audio files.

Requirements:
  pip install playwright requests beautifulsoup4 tqdm
  playwright install chromium

Usage:
  python scripts/scraper.py --download-audio --output-dir data/full_corpus

The script is resumable — it skips files that already exist.
"""

import argparse
import json
import os
import re
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from tqdm import tqdm

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

SERIES_PAGES = {
    "english": "https://oshoworld.com/audio-series-home-english",
    "hindi": "https://oshoworld.com/audio-series-home-hindi",
}

def get_series_urls(page, language: str) -> list[dict]:
    """Use Playwright to collect all series from the paginated listing page."""
    base = SERIES_PAGES[language]
    series = []
    seen = set()

    page.goto(base, wait_until="networkidle", timeout=60000)
    page.wait_for_timeout(3000)

    page_num = 1
    while True:
        print(f"  Scraping {language} series page {page_num}...")
        # Extract series links
        links = page.evaluate("""() => {
            const results = [];
            document.querySelectorAll('a[href]').forEach(a => {
                const text = a.innerText.trim();
                const href = a.href;
                if (text && href && (text.toLowerCase().includes('view all') || 
                    href.match(/-\\d+-\\d+$/) || href.includes('by-osho'))) {
                    results.push({title: text, href});
                }
            });
            return results;
        }""")

        for item in links:
            href = item["href"]
            if href not in seen and "oshoworld.com" in href and not href.endswith("#"):
                # Prefer the series overview page (contains "View all" or ends with range)
                if "view all" in item["title"].lower() or re.search(r"-\d+-\d+$", href.split("/")[-1]):
                    series.append({"title": item["title"], "url": href, "language": language})
                    seen.add(href)

        # Try to click Next
        next_btn = page.query_selector("text=Next") or page.query_selector("a:has-text('Next')")
        if next_btn and next_btn.is_visible():
            next_btn.click()
            page.wait_for_timeout(2500)
            page_num += 1
        else:
            break

        if page_num > 20:  # safety
            break

    # Deduplicate by URL
    unique = []
    seen_urls = set()
    for s in series:
        if s["url"] not in seen_urls:
            unique.append(s)
            seen_urls.add(s["url"])
    return unique


def get_discourse_urls(page, series_url: str) -> list[str]:
    """From a series page, collect all individual discourse page URLs."""
    page.goto(series_url, wait_until="networkidle", timeout=60000)
    page.wait_for_timeout(2000)

    urls = page.evaluate("""() => {
        const results = [];
        document.querySelectorAll('a[href]').forEach(a => {
            const href = a.href;
            const text = a.innerText.trim();
            // Discourse pages typically end with -01, -02, etc. or similar
            if (href && href.includes('oshoworld.com') && 
                !href.includes('#') && 
                (href.match(/-\\d{2}$/) || href.match(/-\\d+$/)) &&
                text && text.length > 3) {
                results.push(href);
            }
        });
        return [...new Set(results)];
    }""")
    return urls


def scrape_discourse(url: str) -> dict | None:
    """Scrape transcript and audio URL from a single discourse page using requests."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=40)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")

        title_el = soup.find("h1") or soup.find("title")
        title = title_el.get_text(strip=True) if title_el else url.split("/")[-1]

        # Extract transcript — large text blocks after the main content
        transcript_parts = []
        for tag in soup.find_all(["p", "div", "span"]):
            txt = tag.get_text(" ", strip=True)
            if len(txt) > 150 and not any(skip in txt[:80].lower() for skip in 
                ["home", "osho", "login", "share", "download", "contact", "privacy", "cookie"]):
                transcript_parts.append(txt)

        transcript = "\n\n".join(transcript_parts[:80])  # reasonable limit

        # Find audio
        audio_url = None
        for source in soup.find_all("source"):
            src = source.get("src", "")
            if src.endswith(".mp3"):
                audio_url = urljoin(url, src)
                break
        if not audio_url:
            for a in soup.find_all("a", href=True):
                if a["href"].endswith(".mp3"):
                    audio_url = urljoin(url, a["href"])
                    break

        return {
            "url": url,
            "title": title,
            "transcript": transcript,
            "audio_url": audio_url,
            "language": "hindi" if any(x in url.lower() for x in ["hindi", "by-osho"]) else "english",
        }
    except Exception as e:
        print(f"  Error scraping {url}: {e}")
        return None


def download_audio(audio_url: str, dest_path: Path) -> bool:
    if dest_path.exists():
        return True
    try:
        resp = requests.get(audio_url, headers=HEADERS, stream=True, timeout=120)
        resp.raise_for_status()
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"  Audio download failed {audio_url}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/full_corpus", help="Where to save JSON + audio")
    parser.add_argument("--download-audio", action="store_true", help="Also download MP3 files")
    parser.add_argument("--languages", nargs="+", default=["english", "hindi"])
    parser.add_argument("--max-series", type=int, default=None, help="Limit for testing")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = output_dir / "audio"
    meta_dir = output_dir / "transcripts"

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        all_series = []
        for lang in args.languages:
            print(f"\nDiscovering {lang} series...")
            series = get_series_urls(page, lang)
            print(f"  Found {len(series)} series")
            all_series.extend(series)

        if args.max_series:
            all_series = all_series[: args.max_series]

        print(f"\nTotal series to process: {len(all_series)}")

        for series in tqdm(all_series, desc="Series"):
            series_slug = re.sub(r"[^a-zA-Z0-9_-]", "_", series["url"].split("/")[-1])[:80]
            series_dir = meta_dir / series["language"] / series_slug
            series_dir.mkdir(parents=True, exist_ok=True)

            discourse_urls = get_discourse_urls(page, series["url"])
            print(f"\n{series['title'][:60]} → {len(discourse_urls)} discourses")

            for d_url in discourse_urls:
                safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", d_url.split("/")[-1])[:100]
                json_path = series_dir / f"{safe_name}.json"
                if json_path.exists():
                    continue

                data = scrape_discourse(d_url)
                if not data or not data.get("transcript"):
                    continue

                # Download audio if requested
                if args.download_audio and data.get("audio_url"):
                    audio_path = audio_dir / series["language"] / series_slug / f"{safe_name}.mp3"
                    if download_audio(data["audio_url"], audio_path):
                        data["local_audio"] = str(audio_path.relative_to(output_dir))

                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)

                time.sleep(0.8)  # polite delay

        browser.close()

    print("\nScraping complete.")
    print(f"Transcripts saved under: {meta_dir}")
    if args.download_audio:
        print(f"Audio saved under: {audio_dir}")


if __name__ == "__main__":
    main()
