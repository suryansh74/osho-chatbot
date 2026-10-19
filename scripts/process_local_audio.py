"""
Process a local Osho audio collection into transcripts for RAG.

Walks a directory tree of .mp3 / .amr / .wav files, transcribes them with
faster-whisper, and writes one JSON per file containing:
  - title (derived from folder + filename)
  - transcript
  - local_audio (absolute or relative path)
  - language (guessed)

Usage (after installing requirements):
  python scripts/process_local_audio.py --audio-root /path/to/your/osho/audio --output-dir data/local_corpus

This is resumable — existing JSON files are skipped.
Transcription is the slow part (CPU/GPU dependent).
"""

import argparse
import json
import re
from pathlib import Path

from faster_whisper import WhisperModel
from tqdm import tqdm

AUDIO_EXTENSIONS = {".mp3", ".amr", ".wav", ".m4a", ".ogg", ".flac"}


def clean_title(path: Path) -> str:
    """Build a readable title from the relative path."""
    parts = path.parts
    # Prefer the series folder name + the file stem
    stem = path.stem
    # Remove common prefixes / numbering noise
    stem = re.sub(r"^[A-Z0-9]{5,}_?", "", stem)
    stem = re.sub(r"[_-]+", " ", stem)
    stem = re.sub(r"\s+", " ", stem).strip()
    series = path.parent.name
    series = re.sub(r"[_-]+", " ", series).strip()
    if series and series.lower() not in stem.lower():
        return f"{series} - {stem}"
    return stem or path.stem


def guess_language(path: Path, model_lang: str | None) -> str:
    p = str(path).lower()
    if any(x in p for x in ["hindi", "amr", "osho hindi"]):
        return "hindi"
    if model_lang:
        return model_lang
    return "english"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-root", required=True, help="Root folder containing your Osho audio")
    parser.add_argument("--output-dir", default="data/local_corpus", help="Where to write JSON transcripts")
    parser.add_argument("--model", default="base", help="Whisper model size (tiny/base/small/medium/large-v3)")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--compute-type", default="int8", help="int8 / float16 / float32")
    parser.add_argument("--language", default=None, help="Force language code (hi / en). Leave empty for auto")
    args = parser.parse_args()

    audio_root = Path(args.audio_root).expanduser().resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not audio_root.exists():
        print(f"Audio root does not exist: {audio_root}")
        return

    print(f"Loading Whisper model '{args.model}' on {args.device}...")
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)

    audio_files = []
    for ext in AUDIO_EXTENSIONS:
        audio_files.extend(audio_root.rglob(f"*{ext}"))
    audio_files = sorted(set(audio_files))
    print(f"Found {len(audio_files)} audio files")

    for audio_path in tqdm(audio_files, desc="Transcribing"):
        # Unique relative key for the JSON name
        try:
            rel = audio_path.relative_to(audio_root)
        except ValueError:
            rel = audio_path
        safe_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", str(rel))[:180]
        json_path = output_dir / f"{safe_name}.json"

        if json_path.exists():
            continue

        try:
            segments, info = model.transcribe(
                str(audio_path),
                language=args.language,
                beam_size=5,
                vad_filter=True,
            )
            transcript = " ".join(seg.text.strip() for seg in segments).strip()
            if not transcript:
                continue

            data = {
                "title": clean_title(audio_path),
                "transcript": transcript,
                "local_audio": str(audio_path),
                "relative_audio": str(rel),
                "language": guess_language(audio_path, info.language),
                "duration": info.duration,
                "source": "local_audio",
            }
            json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Failed on {audio_path}: {e}")

    print(f"\nDone. Transcripts written to {output_dir}")
    print("Next: python scripts/index_data.py  (it will pick up data/local_corpus)")


if __name__ == "__main__":
    main()
