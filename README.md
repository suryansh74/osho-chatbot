# Osho Chatbot (Personal)

A personal web-based chatbot that answers questions from Osho's full discourse corpus (English + Hindi) on [oshoworld.com](https://oshoworld.com/).

It returns **source references** and the **original audio** (local playback or link). A LangGraph node can later decide to call web search for external evidence/verification of claims.

## Features
- Full RAG over scraped transcripts
- LangGraph pipeline: retrieve → generate → decide if evidence needed → search evidence (placeholder)
- Original audio playback in the UI (after you download the MP3s)
- Hindi + English support
- Resumable scraper that can download **all** series, transcripts, and audio

## Important
This is for **personal use**. Full audio is very large (hundreds of GB). The scraper downloads everything locally to your machine. GitHub only holds the code.

## Setup

1. Clone and install:
   ```bash
   git clone https://github.com/suryansh74/osho-chatbot.git
   cd osho-chatbot
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   playwright install chromium
   ```

2. Set your Groq API key:
   ```bash
   export GROQ_API_KEY=your_key_here
   ```

3. **Scrape all data** (this will take a long time and a lot of disk space):
   ```bash
   python scripts/scraper.py --download-audio --output-dir data/full_corpus
   ```
   - Remove `--download-audio` if you only want transcripts first.
   - Use `--max-series 5` for a quick test.
   - The script is resumable — you can stop and restart it.

4. Index the transcripts:
   ```bash
   python scripts/index_data.py
   ```

5. Run the chatbot:
   ```bash
   streamlit run app.py
   ```

## Audio + Timestamps
- The chatbot shows the relevant discourse title, original page link, and plays the local MP3 when available.
- Exact sentence-level timestamps are not provided by the source site. The full original audio is served so you can listen from the start or scrub. Later you can add Whisper-based alignment if you want precise timestamps.

## Evidence / Web Search Node
The graph already has a `decide_evidence` node that checks whether the question involves a historical or external claim worth verifying. The `search_evidence` node is currently a placeholder. You can later replace it with Tavily, SerpAPI, DuckDuckGo, or any search tool.

## Disclaimer
Unofficial personal project. Osho's teachings are copyrighted. Use only for your own study.
