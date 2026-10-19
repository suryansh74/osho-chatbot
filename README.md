# Osho Chatbot

Faithful RAG chatbot over Osho’s English and Hindi discourse transcripts from oshoworld.com.

## What it does
- Dropdown to choose English or Hindi
- Retrieves relevant original passages
- Produces a clear, readable answer that **does not add or alter** the original essence
- Shows original transcript excerpts + direct links to the source discourses
- Supports questions that draw from multiple discourses

## Setup

```bash
# inside your venv
pip install -r requirements.txt
playwright install chromium

# 1. Scrape all transcripts (takes time, resumable)
python scripts/scraper.py --output-dir data

# Optional test run
# python scripts/scraper.py --max-series 3 --output-dir data

# 2. Index
python scripts/index_data.py

# 3. Run
streamlit run app.py
```

Transcripts are saved under:
- `data/english/`
- `data/hindi/`

Each discourse is a JSON file with title, URL, full transcript, series name, and duration (when available).

## Notes
- The generation prompt is strict: the model may only rephrase for readability. It is forbidden from adding its own psychology or interpretations.
- Audio is intentionally not included for now.
- Web-search evidence node can be added later.
