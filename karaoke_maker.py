#!/usr/bin/env python3
"""
Karaoke Videoclip Maker

Input  : background video (e.g. MP4), music audio (e.g. MP3/WAV), LRC lyrics
Output : a new video with the audio replaced by the music file and
         burned-in subtitles synced from the LRC.

Dependencies:
- Python 3.9+
- ffmpeg installed and on PATH (https://ffmpeg.org/)
- No extra Python packages beyond the standard library.

Usage examples:

# Basic
python karaoke_maker.py --video static/alles.mp4 --audio static/alles.mp3 --lrc static/alles.lrc --out static/karaoke.mp4 --font "Verdana" --font-size 20 --alignment bottom --outsrt static/alles.srt

# Tune subtitle style and offset (e.g., music starts 320 ms after the video's first frame)
python karaoke_maker.py \
  --video bg.mp4 --audio song.mp3 --lrc lyrics.lrc --out karaoke.mp4 \
  --offset-ms 320 --font "Arial" --font-size 48 --alignment bottom \
  --primary-colour FFFFFF --outline-colour 000000 --outline 2 --shadow 0 \
  --back-colour 000000 --back-alpha 96
  Alignment mapping: bottom (2), middle (5), top (8) => 5 is default
  
  python karaoke_maker.py --video upload/alles.mp4 --audio upload/alles.mp3 --lrc output/alles.lrc --out output/karaoke.mp4 \
  --font "Verdana" --font-size 20--alignment bottom 
  
  
    Timing: --offset-ms 320 (positief of negatief), --fallback-duration 4.0
    Styling: --font "Arial" --font-size 48 --alignment bottom --primary-colour FFFFFF --outline-colour 000000 --outline 2 --back-colour 000000 --back-alpha 96
    Encoding: --crf 18 --preset medium --audio-bitrate 192k --pix-fmt yuv420p
    Lengtekeuze: standaard stopt bij de kortste stream; gebruik --no-shortest om de volledige video uit te renderen.

    # If your LRC only has start timestamps per line, last line duration fallback is used (default 4.0s).

Notes:
- The script converts LRC -> SRT, then uses ffmpeg's subtitles filter (libass) to burn in.
- It supports multiple timestamps per line (the same lyric repeated at different times).
- It tolerates both mm:ss.xx and mm:ss formats, and [offset:1234] tags (ms) in the LRC.
- For best results, keep your video and audio lengths roughly similar; ffmpeg will stop at the shortest by default.
"""

from __future__ import annotations

import argparse
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

# -----------------------------
# LRC parsing
# -----------------------------
TIME_TAG = re.compile(r"\[(\d{1,2}):(\d{2})(?:[.:](\d{1,3}))?\]")
INLINE_TIME_TAG = re.compile(r"<\d{1,2}:\d{2}(?:[.:]\d{1,3})?>")
META_TAG = re.compile(r"^\[(ti|ar|al|by|offset):(.+?)\]\s*$", re.IGNORECASE)
OFFSET_TAG = re.compile(r"^\[offset:([+-]?\d+)\]\s*$", re.IGNORECASE)

@dataclass
class LyricEvent:
    start_ms: int
    text: str
    
# Voor WSL specifiek:
def get_wsl_path(path: Path) -> str:
    """Converteer Windows pad naar WSL pad"""
    path_str = str(path.resolve())
    if '\\' in path_str:
        drive = path_str[0].lower()
        return f"/mnt/{drive}/{path_str[3:].replace('\\', '/')}"
    return str(path)


def parse_time_tag_to_ms(m: re.Match) -> int:
    minutes = int(m.group(1))
    seconds = int(m.group(2))
    frac = m.group(3)
    if frac is None:
        millis = 0
    else:
        # Pad to milliseconds precision
        frac_str = (frac + "000")[:3]
        millis = int(frac_str)
    return (minutes * 60 + seconds) * 1000 + millis


def parse_lrc(lrc_path: Path) -> Tuple[List[LyricEvent], int]:
    """Return (events, global_offset_ms). Supports multiple timestamps per line.
    Lines without a time tag are ignored. Empty lyric text becomes an em-dash.
    """
    events: List[LyricEvent] = []
    global_offset_ms = 0

    with lrc_path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip("\ufeff\n\r ")
            if not line:
                continue

            # Global offset tag
            off = OFFSET_TAG.match(line)
            if off:
                try:
                    global_offset_ms = int(off.group(1))
                except ValueError:
                    pass
                continue

            # Skip simple meta tags (title/artist/etc.)
            if META_TAG.match(line):
                continue

            # Extract all time tags at the start
            times = list(TIME_TAG.finditer(line))
            if not times:
                continue

            # Remove all leading time tags to get the pure lyric text
            text = TIME_TAG.sub("", line).strip()
            text = INLINE_TIME_TAG.sub("", text).strip()
            if not text:
                text = "—"

            for t in times:
                start_ms = parse_time_tag_to_ms(t)
                events.append(LyricEvent(start_ms=start_ms, text=text))

    # Sort by time in case LRC is unordered
    events.sort(key=lambda e: e.start_ms)
    return events, global_offset_ms


# -----------------------------
# SRT writing
# -----------------------------

def ms_to_srt_timestamp(ms: int) -> str:
    if ms < 0:
        ms = 0
    hours = ms // 3_600_000
    ms %= 3_600_000
    minutes = ms // 60_000
    ms %= 60_000
    seconds = ms // 1000
    millis = ms % 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"

"""
def build_srt(events: List[LyricEvent], *, offset_ms: int, fallback_duration_ms: int) -> str:
    Create SRT text from events. Each event ends at the next event's start - 100ms
    or uses fallback_duration_ms for the last event.
    
    lines = []
    n = len(events)
    for i, ev in enumerate(events, start=1):
        start = ev.start_ms + offset_ms
        if i < n:
            next_start = events[i].start_ms + offset_ms
            end = max(start + 400, next_start - 100)  # ensure min 400ms visibility
        else:
            end = start + fallback_duration_ms

        lines.append(str(i))
        lines.append(f"{ms_to_srt_timestamp(start)} --> {ms_to_srt_timestamp(end)}")
        # Escape potential SRT-problematic characters minimally; SRT is pretty lax.
        text = ev.text.replace("\u2028", " ").replace("\u2029", " ")
        lines.append(text)
        lines.append("")

    return "\n".join(lines)
"""

def convert_lrc_to_srt(lrc_content: str, offset_ms: int = 0, fallback_duration_ms: int = 4000) -> str:
    """
    Verbeterde LRC naar SRT conversie die naadloos integreert met je bestaande code.
    Handelt zowel [mm:ss.xx] als [mm:ss] formaten en ondersteunt Unicode volledig.
    """
    srt_lines = []
    events = []
    
    # Regex voor LRC-tijdstempels: [mm:ss.xx] of [mm:ss]
    time_pattern = re.compile(r'\[(\d+):(\d+)(?:\.(\d+))?\](.*)')
    
    for line in lrc_content.split('\n'):
        line = line.strip()
        if not line:
            continue
        
        match = time_pattern.match(line)
        if not match:
            continue  # Sla metadata en ongeldige regels over
        
        minutes, seconds, hundredths, text = match.groups()
        text = text.strip()
        
        # Converteer tijd naar milliseconden
        hundredths = hundredths or '00'
        hundredths = (hundredths + '00')[:2]  # Zorg voor 2 cijfers
        start_ms = (int(minutes) * 60 + int(seconds)) * 1000 + int(hundredths) * 10
        
        events.append((start_ms + offset_ms, text))
    
    # Sorteer op tijd (voor het geval de LRC niet gesorteerd is)
    events.sort()
    
    # Genereer SRT-regels
    for i, (start_ms, text) in enumerate(events, 1):
        end_ms = events[i][0] - 100 if i < len(events) else start_ms + fallback_duration_ms
        start_time = ms_to_srt_timestamp(start_ms)
        end_time = ms_to_srt_timestamp(end_ms)
        
        srt_lines.append(f"{i}\n{start_time} --> {end_time}\n{text}\n")
    
    return "\n".join(srt_lines)



def save_srt_to_file(srt_content: str, output_path: Path) -> None:
    """
    Sla SRT-inhoud op naar bestand met correcte Unicode-encoding.
    """
    try:
        output_path.write_text(srt_content, encoding='utf-8')
        print(f"✅ SRT-bestand opgeslagen: {output_path}")
    except Exception as e:
        print(f"❌ Fout bij opslaan SRT-bestand: {e}")
        raise

# -----------------------------
# ffmpeg helpers
# -----------------------------

def check_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        sys.exit("Error: ffmpeg not found on PATH. Please install ffmpeg and try again.")


def build_ffmpeg_cmd(
    
    video: Path,
    audio: Path,
    srt: Path,
    out: Path,
    *,
    font: str,
    font_size: int,
    alignment: str,
    primary_colour: str,
    outline_colour: str,
    outline: int,
    shadow: int,
    back_colour: str,
    back_alpha: int,
    video_codec: str,
    audio_codec: str,
    crf: int,
    preset: str,
    audio_bitrate: str,
    pix_fmt: str,
    shortest: bool,
) -> List[str]:
    # Convert hex colours to libass BGR with alpha handled via BackColour separately.
    def hex_to_bgr_hex(h: str) -> str:
        h = h.strip().lstrip('#')
        if len(h) != 6:
            raise ValueError("Colour must be 6 hex digits, e.g. FFFFFF")
        r = h[0:2]
        g = h[2:4]
        b = h[4:6]
        return f"{b}{g}{r}"  # BGR order for libass

    # libass colours are &HAABBGGRR (AA = alpha, BBGGRR = BGR)
    pri = hex_to_bgr_hex(primary_colour)
    outl = hex_to_bgr_hex(outline_colour)
    back = hex_to_bgr_hex(back_colour)

    # Alpha: 0 (opaque) .. 255 (fully transparent). We get 0..255 via back_alpha
    back_alpha = max(0, min(255, back_alpha))

    # Alignment mapping: bottom (2), middle (5), top (8)
    align_map = {
        'bottom': 2,
        'middle': 5,
        'center': 5,
        'top': 8,
    }
    align_val = align_map.get(alignment.lower(), 5)

    force_style = (
        f"FontName={font},FontSize={font_size},"
        f"PrimaryColour=&H00{pri}&,OutlineColour=&H00{outl}&,Outline={outline},Shadow={shadow},"
        f"BackColour=&H{back_alpha:02X}{back}&,BorderStyle=1,Alignment={align_val}"
    )

    # Gebruik raw string voor Windows/WSL pad conversie
    srt_path = get_wsl_path(srt)
    vf = f"subtitles='{srt_path}':force_style='{force_style}'"
    #vf = f"subtitles={shlex.quote(str(srt))}:force_style={shlex.quote(force_style)}"

    cmd = [
        "/usr/bin/ffmpeg",  # Gebruik volledige pad voor WSL compatibiliteit
        "-y",
        "-hide_banner", 
        "-loglevel", 
        "error",  # Minder ruis in output
        "-i", str(video),
        "-i", str(audio),
        "-vf", vf,
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", video_codec,
        "-pix_fmt", pix_fmt,
        "-preset", preset,
        "-crf", str(crf),
        "-c:a", audio_codec,
        "-b:a", audio_bitrate,
    ]
    if shortest:
        cmd += ["-shortest"]
    cmd += [str(out)]
    return cmd


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Create a karaoke video from video + audio + LRC lyrics")
    ap.add_argument("--video", required=True, type=Path, help="Path to background video (e.g., MP4)")
    ap.add_argument("--audio", required=True, type=Path, help="Path to song audio (e.g., MP3/WAV)")
    ap.add_argument("--lrc", required=True, type=Path, help="Path to LRC lyrics file")
    ap.add_argument("--out", required=True, type=Path, help="Output video path (e.g., MP4)")
    ap.add_argument("--outsrt", required=True, type=Path, help="Output srt path (e.g., SRT)")

    # Timing
    ap.add_argument("--offset-ms", type=int, default=0, help="Global timing offset in ms applied to all lyrics (can be negative)")
    ap.add_argument("--fallback-duration", type=float, default=4.0, help="Seconds for last-line duration when next timestamp is unknown")

    # Style (libass / ffmpeg subtitles)
    ap.add_argument("--font", default="Arial", help="Subtitle font name")
    ap.add_argument("--font-size", type=int, default=48, help="Subtitle font size")
    ap.add_argument("--alignment", choices=["top", "middle", "center", "bottom"], default="bottom", help="Subtitle vertical alignment")
    ap.add_argument("--primary-colour", default="FFFFFF", help="Hex text colour, e.g. FFFFFF")
    ap.add_argument("--outline-colour", default="000000", help="Hex outline colour, e.g. 000000")
    ap.add_argument("--outline", type=int, default=2, help="Outline thickness")
    ap.add_argument("--shadow", type=int, default=0, help="Drop shadow size (0 disables)")
    ap.add_argument("--back-colour", default="000000", help="Hex background box colour")
    ap.add_argument("--back-alpha", type=int, default=96, help="Background alpha 0..255 (0=opaque, 255=transparent)")

    # Encoding
    ap.add_argument("--video-codec", default="libx264", help="FFmpeg video codec (e.g., libx264, libx265)")
    ap.add_argument("--audio-codec", default="aac", help="FFmpeg audio codec (e.g., aac, libmp3lame)")
    ap.add_argument("--audio-bitrate", default="192k", help="Audio bitrate (e.g., 192k)")
    ap.add_argument("--crf", type=int, default=18, help="Constant Rate Factor for video quality (lower = better)")
    ap.add_argument("--preset", default="medium", help="FFmpeg x264/x265 preset (ultrafast..placebo)")
    ap.add_argument("--pix-fmt", default="yuv420p", help="Pixel format (yuv420p recommended for compatibility)")
    ap.add_argument("--no-shortest", action="store_true", help="Do not stop at the shortest stream; render full length of video")

    args = ap.parse_args()
    

    # Basic checks
    for p in (args.video, args.audio, args.lrc):
        if not p.exists():
            sys.exit(f"Error: File not found: {p}")
            

    check_ffmpeg()

    # Parse LRC
    events, lrc_offset = parse_lrc(args.lrc)
    if not events:
        sys.exit("Error: No timed lyric lines found in the LRC file.")

    # Compute effective offset: LRC's own [offset:] + user-provided offset
    effective_offset = lrc_offset + args.offset_ms

    # Build SRT text
    """
    srt_text = build_srt(
        events,
        offset_ms=effective_offset,
        fallback_duration_ms=int(args.fallback_duration * 1000),
    )
    """
    
    srt_text = convert_lrc_to_srt(
        args.lrc.read_text(encoding='utf-8'),
        offset_ms=effective_offset,
        fallback_duration_ms=int(args.fallback_duration * 1000)
    )

    # Opslaan naar schijf
    save_srt_to_file(srt_text, args.outsrt)

    # Temp SRT file
    with tempfile.TemporaryDirectory(prefix="karaoke_") as td:
        srt_path = Path(td) / "lyrics.srt"
        srt_path.write_text(srt_text, encoding="utf-8")

        # ffmpeg command
        cmd = build_ffmpeg_cmd(
            video=args.video,
            audio=args.audio,
            srt=srt_path,
            out=args.out,
            font=args.font,
            font_size=args.font_size,
            alignment=args.alignment,
            primary_colour=args.primary_colour,
            outline_colour=args.outline_colour,
            outline=args.outline,
            shadow=args.shadow,
            back_colour=args.back_colour,
            back_alpha=args.back_alpha,
            video_codec=args.video_codec,
            audio_codec=args.audio_codec,
            crf=args.crf,
            preset=args.preset,
            audio_bitrate=args.audio_bitrate,
            pix_fmt=args.pix_fmt,
            shortest=not args.no_shortest,
        )

        # Run ffmpeg
        print("Running:", " ".join(shlex.quote(c) for c in cmd))
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            sys.exit(f"ffmpeg failed with exit code {e.returncode}")

        print(f"✅ Done. Wrote: {args.out}")


if __name__ == "__main__":
    main()
