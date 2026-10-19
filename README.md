# Osho Chatbot

A web-based chatbot that answers questions using Osho's discourses and teachings from [oshoworld.com](https://oshoworld.com/), with source references.

Built with:
- **RAG** (Retrieval-Augmented Generation)
- **LangGraph** for the agent flow
- **Groq** API for fast LLM inference (Llama models)
- **Streamlit** for the web interface
- Local embeddings via Sentence Transformers + FAISS

## Features
- Answers based on actual Osho transcripts (Hindi & English)
- Provides reference citations to specific discourses
- Supports both English and Hindi queries (responses in the query language where possible)

## Setup

1. Clone the repo:
   ```bash
   git clone https://github.com/suryansh74/osho-chatbot.git
   cd osho-chatbot
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Set your Groq API key:
   ```bash
   export GROQ_API_KEY=your_groq_api_key_here
   ```
   Or create a `.env` file with `GROQ_API_KEY=your_key`.

4. (Optional) Scrape data:
   The `scripts/scraper.py` can fetch transcripts. Run it after installing playwright if needed for full JS sites.
   Sample data is included in `data/samples/`.

5. Index the data (creates vector store):
   ```bash
   python scripts/index_data.py
   ```

6. Run the chatbot:
   ```bash
   streamlit run app.py
   ```

## Notes
- Content is sourced from publicly available free resources on oshoworld.com. Please respect copyright and use for personal/educational purposes.
- Full corpus is large (thousands of discourses). The scraper helps you download what you need.
- Embeddings are computed locally (no extra API cost).

## Disclaimer
This is an unofficial fan project. Osho's teachings are copyrighted. Always refer to official sources.
