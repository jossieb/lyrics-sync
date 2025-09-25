#!/usr/bin/env python3
"""
Karaoke Videoclip Maker

Modes:
- line: gebruik .lrc → genereer .srt (line-subtitles)
- word: gebruik .json → genereer .ass (karaoke word highlighting)
"""

from __future__ import annotations
import argparse
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
import json

# -----------------------------
# Helpers
# -----------------------------
def get_wsl_path(path: Path) -> str:
    path_str = str(path.resolve())
    if '\\' in path_str:
        drive = path_str[0].lower()
        return f"/mnt/{drive}/{path_str[3:].replace('\\', '/')}"
    return str(path)

def ms_to_srt_timestamp(ms: int) -> str:
    if ms < 0: ms = 0
    h = ms // 3_600_000; ms %= 3_600_000
    m = ms // 60_000; ms %= 60_000
    s = ms // 1000; ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def ms_to_ass_timestamp(ms: int) -> str:
    if ms < 0: ms = 0
    h = ms // 3_600_000; ms %= 3_600_000
    m = ms // 60_000; ms %= 60_000
    s = ms // 1000; cs = (ms % 1000)//10
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

def check_ffmpeg():
    if shutil.which("ffmpeg") is None:
        sys.exit("Error: ffmpeg not found on PATH")

# -----------------------------
# Converters
# -----------------------------
def convert_lrc_to_srt(lrc_text: str, offset_ms: int=0, fallback_duration_ms: int=4000) -> str:
    events = []
    pat = re.compile(r'\[(\d+):(\d+)(?:\.(\d+))?\](.*)')
    for line in lrc_text.splitlines():
        m = pat.match(line.strip())
        if not m: continue
        minutes, seconds, hundredths, text = m.groups()
        hundredths = hundredths or '00'
        hundredths = (hundredths + '00')[:2]
        start = (int(minutes)*60+int(seconds))*1000 + int(hundredths)*10 + offset_ms
        events.append((start, text.strip()))
    events.sort()
    srt_lines=[]
    for i,(start,text) in enumerate(events,1):
        end = events[i][0]-100 if i<len(events) else start+fallback_duration_ms
        srt_lines.append(f"{i}\n{ms_to_srt_timestamp(start)} --> {ms_to_srt_timestamp(end)}\n{text}\n")
    return "\n".join(srt_lines)

def convert_json_to_ass(json_path: Path, offset_ms: int=0) -> str:
    words = json.loads(json_path.read_text(encoding="utf-8"))
    ass_lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        "Collisions: Normal",
        "PlayResX: 1280",
        "PlayResY: 720",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding",
        "Style: Default,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,2,0,2,10,10,30,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    if not words: return "\n".join(ass_lines)
    start_time = int(words[0]["start"]*1000)+offset_ms
    end_time = int(words[-1]["end"]*1000)+offset_ms
    dialogue_text=""
    for w in words:
        s=int(w["start"]*1000)+offset_ms
        e=int(w["end"]*1000)+offset_ms
        dur_cs=max(1,(e-s)//10)
        dialogue_text += f"{{\\k{dur_cs}}}{w['word']} "
    ass_lines.append(f"Dialogue: 0,{ms_to_ass_timestamp(start_time)},{ms_to_ass_timestamp(end_time)},Default,,0,0,0,,{dialogue_text.strip()}")
    return "\n".join(ass_lines)

# -----------------------------
# ffmpeg command
# -----------------------------
def build_ffmpeg_cmd(video: Path, audio: Path, subs: Path, out: Path, force_style: str,
                     video_codec="libx264", audio_codec="aac", crf=18,
                     preset="medium", audio_bitrate="192k", pix_fmt="yuv420p", shortest=True):
    vf = f"subtitles='{get_wsl_path(subs)}':force_style='{force_style}'"
    cmd = [
        "ffmpeg","-y","-hide_banner","-loglevel","error",
        "-i",str(video),"-i",str(audio),
        "-vf",vf,
        "-map","0:v:0","-map","1:a:0",
        "-c:v",video_codec,"-pix_fmt",pix_fmt,"-preset",preset,"-crf",str(crf),
        "-c:a",audio_codec,"-b:a",audio_bitrate
    ]
    if shortest: cmd+=["-shortest"]
    cmd.append(str(out))
    return cmd

# -----------------------------
# Main
# -----------------------------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--video",required=True,type=Path)
    ap.add_argument("--audio",required=True,type=Path)
    ap.add_argument("--lrc",required=True,type=Path,help="Input .lrc (line) or .json (word)")
    ap.add_argument("--out",required=True,type=Path)
    ap.add_argument("--outsrt",required=True,type=Path,help="Output .srt (line) or .ass (word)")
    ap.add_argument("--mode",choices=["line","word"],default="line")
    ap.add_argument("--offset-ms",type=int,default=0)
    ap.add_argument("--fallback-duration",type=float,default=4.0)
    args=ap.parse_args()

    check_ffmpeg()

    if args.mode=="line":
        srt_text=convert_lrc_to_srt(args.lrc.read_text(encoding="utf-8"),
                                    offset_ms=args.offset_ms,
                                    fallback_duration_ms=int(args.fallback_duration*1000))
        args.outsrt.write_text(srt_text,encoding="utf-8")
        subs_path=args.outsrt
    else:
        ass_text=convert_json_to_ass(args.lrc,offset_ms=args.offset_ms)
        args.outsrt.write_text(ass_text,encoding="utf-8")
        subs_path=args.outsrt

    force_style=f"FontName=Arial,FontSize=20,PrimaryColour=&H00FFFFFF&,OutlineColour=&H000000&,Outline=2,Shadow=0,BorderStyle=1,Alignment=2"
    cmd=build_ffmpeg_cmd(args.video,args.audio,subs_path,args.out,force_style,shortest=True)
    print("Running:", " ".join(shlex.quote(c) for c in cmd))
    subprocess.run(cmd,check=True)
    print(f"✅ Done. Wrote: {args.out}")

if __name__=="__main__":
    main()
