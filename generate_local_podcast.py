#!/usr/bin/env python3
"""
Podcast Generator — Synthesizes a two-voice podcast MP3 from a transcript.

Reads podcast_transcript.md, generates audio for each speaker segment using
the best available TTS engine, and combines into a single MP3.

Usage:
    python3 generate_local_podcast.py
"""

import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

# ============================================================================
# CONFIG — Edit these to customise output
# ============================================================================

TRANSCRIPT_FILE = "podcast_transcript.md"
OUTPUT_FILE = "podcast_architecture_of_control.mp3"

# Pause between speaker segments (seconds)
PAUSE_BETWEEN_SPEAKERS = 0.8

# Piper TTS voice models (Linux)
# See https://github.com/rhasspy/piper/blob/master/VOICES.md
PIPER_VOICES = {
    "CLAUDE": "en_GB-alan-medium",       # British male
    "DHARMAJ": "en_US-lessac-medium",     # American male
}

# Piper speech rate (words per minute approx — via length_scale)
# < 1.0 = faster, > 1.0 = slower
PIPER_LENGTH_SCALE = {
    "CLAUDE": 1.0,
    "DHARMAJ": 1.0,
}

# macOS 'say' voices (used if running on macOS)
MAC_VOICES = {
    "CLAUDE": "Daniel",
    "DHARMAJ": "Rishi",
}

MAC_SPEECH_RATE = {
    "CLAUDE": 175,
    "DHARMAJ": 175,
}

# espeak-ng voices (Linux fallback)
ESPEAK_VOICES = {
    "CLAUDE": "en-gb",
    "DHARMAJ": "en-us",
}

ESPEAK_SPEED = {
    "CLAUDE": 160,
    "DHARMAJ": 165,
}

# Audio settings
SAMPLE_RATE = 22050
SAMPLE_WIDTH = 2  # 16-bit
CHANNELS = 1

# ============================================================================
# END CONFIG
# ============================================================================


def parse_transcript(filepath: str) -> list[tuple[str, str]]:
    """Parse transcript markdown into (speaker, text) segments."""
    text = Path(filepath).read_text(encoding="utf-8")

    segments = []
    pattern = re.compile(r'\*\*(\w+):\*\*\s*(.*?)(?=\n\n\*\*\w+:\*\*|\n---\n|\Z)', re.DOTALL)

    for match in pattern.finditer(text):
        speaker = match.group(1).upper()
        content = match.group(2).strip()
        # Clean up markdown artifacts
        content = re.sub(r'\*+', '', content)
        content = content.replace('\n', ' ').strip()
        if content and speaker in ("CLAUDE", "DHARMAJ"):
            segments.append((speaker, content))

    return segments


def detect_platform() -> str:
    """Detect the current platform."""
    system = platform.system().lower()
    if system == "darwin":
        return "macos"
    elif system == "linux":
        return "linux"
    elif system == "windows":
        return "windows"
    return system


def ensure_ffmpeg():
    """Ensure ffmpeg is available."""
    if shutil.which("ffmpeg"):
        return
    print("ffmpeg not found. Installing...")
    if detect_platform() == "linux":
        subprocess.run(["sudo", "apt-get", "update", "-qq"], check=False)
        subprocess.run(["sudo", "apt-get", "install", "-y", "-qq", "ffmpeg"], check=True)
    elif detect_platform() == "macos":
        subprocess.run(["brew", "install", "ffmpeg"], check=True)
    else:
        print("ERROR: Please install ffmpeg manually.")
        sys.exit(1)


def ensure_piper():
    """Ensure piper-tts is installed."""
    try:
        import piper  # noqa: F401
        return True
    except ImportError:
        pass
    print("Installing piper-tts...")
    subprocess.run([sys.executable, "-m", "pip", "install", "piper-tts"], check=True)
    try:
        import piper  # noqa: F401
        return True
    except ImportError:
        print("WARNING: piper-tts installation failed.")
        return False


def ensure_espeak():
    """Ensure espeak-ng is available."""
    if shutil.which("espeak-ng"):
        return True
    print("espeak-ng not found. Installing...")
    try:
        subprocess.run(["sudo", "apt-get", "update", "-qq"], check=False)
        subprocess.run(["sudo", "apt-get", "install", "-y", "-qq", "espeak-ng"], check=True)
        return True
    except Exception:
        print("WARNING: espeak-ng installation failed.")
        return False


def generate_silence(duration_sec: float, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Generate silence as raw PCM bytes."""
    num_samples = int(sample_rate * duration_sec)
    return struct.pack(f"<{num_samples}h", *([0] * num_samples))


def synthesize_piper(text: str, speaker: str, output_wav: str):
    """Generate speech using piper-tts."""
    from piper import PiperVoice

    voice_name = PIPER_VOICES.get(speaker, "en_US-lessac-medium")
    length_scale = PIPER_LENGTH_SCALE.get(speaker, 1.0)

    # piper downloads models to a data dir
    data_dir = Path.home() / ".local" / "share" / "piper-voices"
    data_dir.mkdir(parents=True, exist_ok=True)

    model_path = data_dir / f"{voice_name}.onnx"
    config_path = data_dir / f"{voice_name}.onnx.json"

    # Download model if not present
    if not model_path.exists() or not config_path.exists():
        print(f"  Downloading voice model: {voice_name}...")
        base_url = f"https://huggingface.co/rhasspy/piper-voices/resolve/main"
        # Voice path structure: <lang>/<lang>-<name>/<quality>/<lang>-<name>-<quality>.onnx
        parts = voice_name.split("-")
        lang = parts[0] + "_" + parts[1]  # e.g., en_GB
        quality = parts[-1]  # e.g., medium
        name = "-".join(parts[:-1])  # e.g., en_GB-alan
        voice_path = f"{lang}/{name}/{quality}/{voice_name}"

        for suffix in [".onnx", ".onnx.json"]:
            url = f"{base_url}/{voice_path}{suffix}"
            dest = data_dir / f"{voice_name}{suffix}"
            print(f"    Fetching {url}")
            subprocess.run([
                "python3", "-c",
                f"import urllib.request; urllib.request.urlretrieve('{url}', '{dest}')"
            ], check=True)

    voice = PiperVoice.load(str(model_path), config_path=str(config_path))

    with wave.open(output_wav, "wb") as wav_file:
        voice.synthesize(text, wav_file, length_scale=length_scale)


def synthesize_espeak(text: str, speaker: str, output_wav: str):
    """Generate speech using espeak-ng."""
    voice = ESPEAK_VOICES.get(speaker, "en")
    speed = ESPEAK_SPEED.get(speaker, 160)
    subprocess.run([
        "espeak-ng",
        "-v", voice,
        "-s", str(speed),
        "-w", output_wav,
        text
    ], check=True, capture_output=True)


def synthesize_mac_say(text: str, speaker: str, output_wav: str):
    """Generate speech using macOS 'say' command."""
    voice = MAC_VOICES.get(speaker, "Daniel")
    rate = MAC_SPEECH_RATE.get(speaker, 175)
    aiff_file = output_wav.replace(".wav", ".aiff")
    subprocess.run([
        "say", "-v", voice, "-r", str(rate), "-o", aiff_file, text
    ], check=True)
    # Convert AIFF to WAV
    subprocess.run([
        "ffmpeg", "-y", "-i", aiff_file, "-ar", str(SAMPLE_RATE),
        "-ac", str(CHANNELS), "-sample_fmt", "s16", output_wav
    ], check=True, capture_output=True)
    os.unlink(aiff_file)


def normalize_wav(input_wav: str, output_wav: str):
    """Normalize WAV to consistent format using ffmpeg."""
    subprocess.run([
        "ffmpeg", "-y", "-i", input_wav,
        "-ar", str(SAMPLE_RATE),
        "-ac", str(CHANNELS),
        "-sample_fmt", "s16",
        output_wav
    ], check=True, capture_output=True)


def concatenate_wavs(wav_files: list[str], output_wav: str, pause_sec: float):
    """Concatenate WAV files with silence between them."""
    silence = generate_silence(pause_sec)

    with wave.open(output_wav, "wb") as out:
        out.setnchannels(CHANNELS)
        out.setsampwidth(SAMPLE_WIDTH)
        out.setframerate(SAMPLE_RATE)

        for i, wav_path in enumerate(wav_files):
            with wave.open(wav_path, "rb") as inp:
                out.writeframes(inp.readframes(inp.getnframes()))
            if i < len(wav_files) - 1:
                out.writeframes(silence)


def wav_to_mp3(wav_path: str, mp3_path: str):
    """Convert WAV to MP3 using ffmpeg."""
    subprocess.run([
        "ffmpeg", "-y", "-i", wav_path,
        "-codec:a", "libmp3lame", "-qscale:a", "2",
        mp3_path
    ], check=True, capture_output=True)


def get_wav_duration(wav_path: str) -> float:
    """Get duration of a WAV file in seconds."""
    with wave.open(wav_path, "rb") as w:
        return w.getnframes() / w.getframerate()


def main():
    print("=" * 60)
    print("Podcast Generator")
    print("=" * 60)

    # Check transcript exists
    if not Path(TRANSCRIPT_FILE).exists():
        print(f"ERROR: {TRANSCRIPT_FILE} not found.")
        print("Place the transcript file in the same directory as this script.")
        sys.exit(1)

    # Parse transcript
    print(f"\nParsing {TRANSCRIPT_FILE}...")
    segments = parse_transcript(TRANSCRIPT_FILE)
    print(f"Found {len(segments)} segments.")

    if not segments:
        print("ERROR: No speaker segments found in transcript.")
        sys.exit(1)

    # Detect platform
    plat = detect_platform()
    print(f"Platform: {plat}")

    # Ensure ffmpeg
    ensure_ffmpeg()

    # Select TTS engine
    synthesize_fn = None
    engine_name = ""

    if plat == "macos":
        engine_name = "macOS say"
        synthesize_fn = synthesize_mac_say
        print(f"Using TTS engine: {engine_name}")
        # Check for premium voices
        result = subprocess.run(["say", "-v", "?"], capture_output=True, text=True)
        available = result.stdout
        for speaker, voice in MAC_VOICES.items():
            if voice.lower() in available.lower():
                print(f"  {speaker}: {voice} (found)")
            else:
                print(f"  {speaker}: {voice} (NOT FOUND — using default)")

    elif plat == "linux":
        # Try piper first, then espeak
        if ensure_piper():
            engine_name = "piper-tts"
            synthesize_fn = synthesize_piper
            print(f"Using TTS engine: {engine_name}")
            for speaker, voice in PIPER_VOICES.items():
                print(f"  {speaker}: {voice}")
        elif ensure_espeak():
            engine_name = "espeak-ng"
            synthesize_fn = synthesize_espeak
            print(f"Using TTS engine: {engine_name} (fallback)")
        else:
            print("ERROR: No TTS engine available. Install piper-tts or espeak-ng.")
            sys.exit(1)
    else:
        print(f"ERROR: Unsupported platform: {plat}")
        sys.exit(1)

    # Generate audio
    with tempfile.TemporaryDirectory(prefix="podcast_") as tmpdir:
        wav_files = []
        total_segments = len(segments)

        print(f"\nGenerating audio for {total_segments} segments...")
        for i, (speaker, text) in enumerate(segments):
            raw_wav = os.path.join(tmpdir, f"seg_{i:04d}_raw.wav")
            norm_wav = os.path.join(tmpdir, f"seg_{i:04d}.wav")

            progress = f"[{i + 1}/{total_segments}]"
            preview = text[:60] + "..." if len(text) > 60 else text
            print(f"  {progress} {speaker}: {preview}")

            synthesize_fn(text, speaker, raw_wav)

            # Normalize to consistent format
            normalize_wav(raw_wav, norm_wav)
            wav_files.append(norm_wav)

        # Concatenate
        print("\nCombining segments...")
        combined_wav = os.path.join(tmpdir, "combined.wav")
        concatenate_wavs(wav_files, combined_wav, PAUSE_BETWEEN_SPEAKERS)

        duration = get_wav_duration(combined_wav)
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        print(f"Total duration: {minutes}m {seconds}s")

        # Convert to MP3
        print(f"\nConverting to MP3: {OUTPUT_FILE}")
        wav_to_mp3(combined_wav, OUTPUT_FILE)

    file_size = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    print(f"\nDone! Output: {OUTPUT_FILE} ({file_size:.1f} MB)")
    print("=" * 60)


if __name__ == "__main__":
    main()
