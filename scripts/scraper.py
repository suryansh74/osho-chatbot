"""
Simple scraper for Osho World discourses.
Note: The site uses JavaScript for some listings. For full scrape, consider using Playwright.
This version works for individual discourse pages once you have the URLs.
"""

import requests
from bs4 import BeautifulSoup
import json
import os
from urllib.parse import urljoin
import time

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}

def scrape_discourse(url):
    """Scrape a single discourse page for transcript and metadata."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'lxml')
        
        # Extract title
        title = soup.find('h1') or soup.find('title')
        title_text = title.get_text(strip=True) if title else url.split('/')[-1]
        
        # Extract main transcript text - adjust selectors as needed
        # From observation, the transcript is in the body text after the title
        text_parts = []
        for p in soup.find_all(['p', 'div']):
            txt = p.get_text(strip=True)
            if len(txt) > 100 and 'Home' not in txt[:50] and 'OSHO' not in txt[:20]:
                text_parts.append(txt)
        
        transcript = '\n\n'.join(text_parts[:50])  # Limit to avoid nav
        
        # Find audio
        audio = None
        for source in soup.find_all('source'):
            if source.get('src', '').endswith('.mp3'):
                audio = urljoin(url, source['src'])
                break
        if not audio:
            for a in soup.find_all('a', href=True):
                if a['href'].endswith('.mp3'):
                    audio = urljoin(url, a['href'])
                    break
        
        return {
            'url': url,
            'title': title_text,
            'transcript': transcript,
            'audio': audio,
            'language': 'hindi' if 'hindi' in url.lower() else 'english'
        }
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None

def save_discourse(data, output_dir='data/full_corpus'):
    os.makedirs(output_dir, exist_ok=True)
    safe_title = ''.join(c if c.isalnum() or c in ' -_' else '_' for c in data['title'])[:80]
    path = os.path.join(output_dir, f"{safe_title}.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved: {path}")

if __name__ == '__main__':
    # Example: scrape a known discourse
    sample_urls = [
        'https://oshoworld.com/a-bird-on-the-wing-01',
        # Add more URLs here
    ]
    for url in sample_urls:
        data = scrape_discourse(url)
        if data and data['transcript']:
            save_discourse(data)
        time.sleep(1)  # Be polite
    print("Done. Add more URLs or implement series discovery.")
