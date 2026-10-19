# Osho Chatbot (Personal)

Personal RAG chatbot over Osho discourses.  
Works with **your local audio collection** (recommended) or with transcripts scraped from oshoworld.com.

## Quick start with your local audio (~55 GB)

1. Install (inside the venv you already created):
   ```bash
   pip install -r requirements.txt
   ```

2. Transcribe your local audio (this is the slow step):
   ```bash
   python scripts/process_local_audio.py \
     --audio-root "/path/to/your/osho/audio/folder" \
     --output-dir data/local_corpus \
     --model base \
     --device cpu
   ```
   - Use `--model small` or `medium` for better Hindi quality (slower).
   - Use `--device cuda` if you have a GPU.
   - The script is resumable.

3. Build the vector store:
   ```bash
   python scripts/index_data.py
   ```

4. Run:
   ```bash
   streamlit run app.py
   ```

The chatbot will cite the discourse and play the original local audio file when it finds a match.

## Alternative: scrape oshoworld.com for text

If you prefer website transcripts (and optional audio download):
```bash
python scripts/scraper.py --download-audio --output-dir data/full_corpus
python scripts/index_data.py
```

## Evidence node
The LangGraph already has a decision node that can later call web search for external verification. It is currently a placeholder.

## Notes
- Full transcription of 55 GB takes time (hours to days depending on hardware).
- `.amr` files are supported by faster-whisper.
- Keep the audio files where they are; the chatbot uses the absolute paths stored in the JSON.
