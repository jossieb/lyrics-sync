#!/usr/bin/env python3
"""
Audio-Focused LRC/JSON Maker with Phonetic Alignment

Modes:
- line: maakt klassiek .lrc bestand (regel-niveau synchronisatie)
- word: maakt .json bestand met word-level timings (karaoke)
"""

from __future__ import annotations
import argparse
import re
import warnings
from dataclasses import dataclass
from typing import List, Tuple, Dict
import numpy as np
import whisper
import json

warnings.filterwarnings("ignore")

@dataclass
class WordAlignment:
    word: str
    start: float
    end: float
    confidence: float

# -----------------------------
# Whisper transcription
# -----------------------------
def transcribe_with_word_alignment(audio_path: str, model_name: str = "large") -> List[WordAlignment]:
    print("🎙️ Running Whisper with word-level timing...")
    model = whisper.load_model(model_name)
    result = model.transcribe(
        audio_path,
        language="nl",
        word_timestamps=True,
        condition_on_previous_text=True,
        temperature=0.0,
        beam_size=5,
        best_of=5
    )
    word_alignments = []
    for segment in result["segments"]:
        if "words" not in segment:
            continue
        for w in segment["words"]:
            word_alignments.append(WordAlignment(
                word=w.get("word", "").strip(),
                start=float(w.get("start", 0.0)),
                end=float(w.get("end", 0.0)),
                confidence=0.9  # whisper geeft dit niet → dummy
            ))
    print(f"   Extracted {len(word_alignments)} word alignments")
    return word_alignments

# -----------------------------
# Utility
# -----------------------------
def lrc_timestamp(t: float) -> str:
    m = int(t // 60)
    s = int(t % 60)
    cs = int(round((t - int(t)) * 100))
    return f"[{m:02d}:{s:02d}.{cs:02d}]"

def save_lrc(aligned: List[Tuple[str, float]], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        for line, start in aligned:
            f.write(f"{lrc_timestamp(start)} {line}\n")
    print(f"✅ LRC saved: {out_path}")

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Karaoke LRC/JSON Generator")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--lyrics", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="large")
    parser.add_argument("--mode", choices=["line","word"], default="line")
    args = parser.parse_args()

    # Lyrics laden
    with open(args.lyrics, 'r', encoding='utf-8') as f:
        lyrics = [line.strip() for line in f if line.strip()]
    if not lyrics:
        print("❌ Geen lyrics gevonden")
        return

    # Word-level transcriptie
    word_alignments = transcribe_with_word_alignment(args.audio, args.model)
    if not word_alignments:
        print("❌ Geen woorden gevonden")
        return

    if args.mode == "line":
        # Vereenvoudigde line-based: koppel elke lyric aan eerste woord
        step = max(1, len(word_alignments)//len(lyrics))
        aligned = []
        for i, line in enumerate(lyrics):
            idx = min(i*step, len(word_alignments)-1)
            aligned.append((line, word_alignments[idx].start))
        save_lrc(aligned, args.out)
    else:  # word mode
        word_data = [
            {"word": w.word, "start": w.start, "end": w.end, "confidence": w.confidence}
            for w in word_alignments
        ]
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(word_data, f, ensure_ascii=False, indent=2)
        print(f"✅ JSON saved: {args.out}")

if __name__ == "__main__":
    main()
