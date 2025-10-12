"""
Standaard video generatie instelling:
# Timing
--offset-ms", type=int, default=0, help="Global timing offset in ms applied to all lyrics (can be negative)
--fallback-duration", type=float, default=4.0, help="Seconds for last-line duration when next timestamp is unknown")

# Style (libass / ffmpeg subtitles)
--mode", choices=["word", "line"], default="word" , SET IN SCREEN
--primary-colour", default="FFFFFF", help="Hex text colour, e.g. FFFFFF")
--outline-colour", default="000000", help="Hex outline colour, e.g. 000000")
--outline", type=int, default=2, help="Outline thickness")
--shadow", type=int, default=0, help="Drop shadow size (0 disables)")
--back-colour", default="000000", help="Hex background box colour")
--back-alpha", type=int, default=96, help="Background alpha 0..255 (0=opaque, 255=transparent)")

# Encoding
--video-codec", default="libx264", help="FFmpeg video codec (e.g., libx264, libx265)")
--audio-codec", default="aac", help="FFmpeg audio codec (e.g., aac, libmp3lame)")
--audio-bitrate", default="192k", help="Audio bitrate (e.g., 192k)")
--crf", type=int, default=18, help="Constant Rate Factor for video quality (lower = better)")
--preset", default="medium", help="FFmpeg x264/x265 preset (ultrafast..placebo)")
--pix-fmt", default="yuv420p", help="Pixel format (yuv420p recommended for compatibility)")
--no-shortest", action="store_true", help="Do not stop at the shortest stream; render full length of video")

version: 1.1
reden: Toevoegen minimale duur per regel en check op overlappende regels
datum: 2025-09-26
auteur: JossieB

"""
import os
import subprocess
import time
from flask import Flask, request, render_template, send_from_directory, redirect, url_for, flash
from flask_wtf.csrf import CSRFProtect


# APP SETTINGS
# Configure application
app = Flask(__name__)

# CSRF protection
app.secret_key = b"_Gr1p0pMnKn1p2025;apl"
csrf = CSRFProtect()
csrf.init_app(app)

# UPLOAD and OUTPUT folders
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "output"
ALLOWED_AUDIO = {"mp3", "wav"}
ALLOWED_LYRICS = {"txt"}
ALLOWED_VIDEO = {"mp4"}

# Create folders if they do not exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Set configuration for Flask app
app.config["MAX_CONTENT_LENGTH"] = 150* 1024 * 1024
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER


def allowed_file(filename, allowed):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        start_time = time.time()   # <-- start timer
        
        # Check files
        audio = request.files.get("audio")
        lyrics = request.files.get("lyrics")
        video = request.files.get("video")
        print(f"Audio: {audio}, Lyrics: {lyrics}, Video: {video}")
        if not audio or not allowed_file(audio.filename, ALLOWED_AUDIO):
            flash("Upload een geldig audio-bestand (mp3/wav).")
            return redirect(request.url)
        if not lyrics or not allowed_file(lyrics.filename, ALLOWED_LYRICS):
            flash("Upload een geldig lyrics-bestand (.txt).")
            return redirect(request.url)
        if not video or not allowed_file(video.filename, ALLOWED_VIDEO):
            flash("Upload een geldig video-bestand (.mp4).")
            return redirect(request.url)

        # Save files
        audio_path = os.path.join(app.config["UPLOAD_FOLDER"], audio.filename)
        lyrics_path = os.path.join(app.config["UPLOAD_FOLDER"], lyrics.filename)
        video_path = os.path.join(app.config["UPLOAD_FOLDER"], video.filename)
        print(f"Audio path: {audio_path}, Lyrics path: {lyrics_path}, Video path: {video_path}")
        
        # Check size of video file
        size_bytes = os.path.getsize(video_path )
        if size_bytes > app.config["MAX_CONTENT_LENGTH"]:
            flash("Het video-bestand is te groot. Maximaal 150 MB toegestaan.")
            return redirect
        audio.save(audio_path)
        lyrics.save(lyrics_path)
        video.save(video_path)
        # Output path
        base_name = os.path.splitext(audio.filename)[0]
        lrc_name = base_name  + ".lrc"
        lrc_path = os.path.join(app.config["OUTPUT_FOLDER"], lrc_name)
        video_out_name = base_name + "_karaoke.mp4"
        video_out_path = os.path.join(app.config["OUTPUT_FOLDER"], video_out_name)
        print(f"LRC path: {lrc_path}, Video out path: {video_out_path}")
        
        mode = request.form.get("mode", "word")

        # 1. Run lrc_maker.py
        flash ("LRC aan het genereren...")
        print (f"LRC aan het genereren met audio: {audio_path}, lyrics: {lyrics_path}, output: {lrc_path}")
        result = subprocess.run([
            "python", "lrc_maker.py",
            "--audio", audio_path,
            "--lyrics", lyrics_path,
            "--out", lrc_path
        ], capture_output=True, text=True)
        if result.returncode != 0:
            flash("Fout bij genereren van de LRC: " + result.stderr)
            return redirect(request.url)
        
        # 2. Run karaoke_maker.py
        flash("LRC succesvol gegenereerd. Nu karaokevideo maken...")
        print("LRC succesvol gegenereerd. Nu karaokevideo maken...")
        print(f"Karaoke aan het genereren met Video: {video_path}, Audio: {audio_path}, LRC: {lrc_path}, Output: {video_out_path}, Mode: {mode}, Output SRT: {os.path.join(app.config['OUTPUT_FOLDER'], base_name + '.srt')}")
        result2 = subprocess.run([
            "python", "karaoke_maker.py",
            "--video", video_path,
            "--audio", audio_path,
            "--lrc", lrc_path,
            "--out", video_out_path,
            "--outsrt", os.path.join(app.config["OUTPUT_FOLDER"], base_name + ".srt")
        ], capture_output=True, text=True)

        if result2.returncode != 0:
            flash("Fout bij genereren van de karaokevideo: " + result2.stderr)
            return redirect(request.url)
        
        # Bereken en print doorlooptijd
        elapsed = time.time() - start_time
        hrs, rem = divmod(int(elapsed), 3600)
        mins, secs = divmod(rem, 60)
        elapsed_str = f"{hrs:02d}:{mins:02d}:{secs:02d}"
        print(f"Doorlooptijd totaal (hh:mm:ss): {elapsed_str}")
        flash(f"Karaokevideo succesvol gegenereerd: doorlooptijd: {elapsed_str}")   
        
        # Redirect to download
        return redirect(url_for("download", filename=video_out_name))
            
    flash("Welkom bij de Karaoke Maker!")

    return render_template("index.html")

@app.route("/download/<filename>")
def download(filename):
    flash(f"Bestand {filename} is aangemaakt in folder {app.config["OUTPUT_FOLDER"]}...")
    return send_from_directory(app.config["OUTPUT_FOLDER"], filename, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)