#!/usr/bin/env python3
"""
Podcast Generator — Synthesizes a two-voice podcast MP3 from a transcript.

Reads podcast_transcript.md, generates audio for each speaker segment using
the best available TTS engine, and combines into a single MP3.

Usage:
    python3 generate_local_podcast.py

Dependencies (auto-installed if missing):
    pip install piper-tts imageio-ffmpeg pathvalidate
"""

import ctypes
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

# espeak-ng voices (used via piper's bundled espeak-ng on Linux)
# See: espeak-ng --voices for full list
ESPEAK_VOICES = {
    "CLAUDE": "en-gb",      # British English male — for Claude
    "DHARMAJ": "en-us",     # American English male — for Dharmaj
}

# espeak-ng speech rate (words per minute)
ESPEAK_SPEED = {
    "CLAUDE": 160,
    "DHARMAJ": 165,
}

# espeak-ng pitch (0-99, default ~50)
ESPEAK_PITCH = {
    "CLAUDE": 45,
    "DHARMAJ": 55,
}

# Piper TTS voice models (used if ONNX models are available)
# See https://github.com/rhasspy/piper/blob/master/VOICES.md
PIPER_VOICES = {
    "CLAUDE": "en_GB-alan-medium",       # British male
    "DHARMAJ": "en_US-lessac-medium",    # American male
}

PIPER_LENGTH_SCALE = {
    "CLAUDE": 1.0,
    "DHARMAJ": 1.0,
}

# macOS 'say' voices
MAC_VOICES = {
    "CLAUDE": "Daniel",
    "DHARMAJ": "Rishi",
}

MAC_SPEECH_RATE = {
    "CLAUDE": 175,
    "DHARMAJ": 175,
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
    pattern = re.compile(
        r'\*\*(\w+):\*\*\s*(.*?)(?=\n\n\*\*\w+:\*\*|\n---\n|\Z)', re.DOTALL
    )

    for match in pattern.finditer(text):
        speaker = match.group(1).upper()
        content = match.group(2).strip()
        # Clean up markdown formatting
        content = re.sub(r'\*+', '', content)
        content = content.replace('\n', ' ').strip()
        # Normalize whitespace
        content = re.sub(r'\s+', ' ', content)
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


def find_ffmpeg() -> str:
    """Find ffmpeg binary — system PATH or pip-installed."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
    return ""


def ensure_ffmpeg() -> str:
    """Ensure ffmpeg is available, install if needed. Returns path."""
    path = find_ffmpeg()
    if path:
        return path
    print("ffmpeg not found. Installing via pip...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "imageio-ffmpeg"],
        check=True,
    )
    path = find_ffmpeg()
    if path:
        return path
    print("ERROR: Could not install ffmpeg. Install manually:")
    print("  pip install imageio-ffmpeg")
    print("  # or: sudo apt install ffmpeg")
    sys.exit(1)


# ---------------------------------------------------------------------------
# TTS Engine: Bundled espeak-ng (via piper-tts shared library)
# ---------------------------------------------------------------------------

_espeak_lib = None
_espeak_sample_rate = None
_espeak_audio_chunks: list[bytes] = []


@ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.POINTER(ctypes.c_short), ctypes.c_int, ctypes.c_void_p
)
def _espeak_callback(wav, numsamples, events):
    """Callback invoked by espeak-ng to deliver synthesized audio chunks."""
    if wav is not None and numsamples > 0:
        arr = ctypes.cast(
            wav, ctypes.POINTER(ctypes.c_short * numsamples)
        ).contents
        _espeak_audio_chunks.append(bytes(arr))
    return 0


def _init_bundled_espeak() -> bool:
    """Initialize espeak-ng from piper-tts bundled shared library."""
    global _espeak_lib, _espeak_sample_rate

    if _espeak_lib is not None:
        return True

    try:
        import piper  # noqa: F401
    except ImportError:
        return False

    piper_dir = os.path.dirname(piper.__file__)
    lib_path = os.path.join(piper_dir, "espeakbridge.so")
    data_dir = os.path.join(piper_dir, "espeak-ng-data")

    if not os.path.exists(lib_path):
        return False

    try:
        _espeak_lib = ctypes.CDLL(lib_path)
        # AUDIO_OUTPUT_SYNCHRONOUS = 2
        _espeak_sample_rate = _espeak_lib.espeak_Initialize(
            2, SAMPLE_RATE, data_dir.encode(), 0
        )
        _espeak_lib.espeak_SetSynthCallback(_espeak_callback)
        return _espeak_sample_rate > 0
    except Exception as e:
        print(f"  WARNING: Failed to init bundled espeak: {e}")
        _espeak_lib = None
        return False


def synthesize_bundled_espeak(text: str, speaker: str, output_wav: str):
    """Generate speech using piper's bundled espeak-ng library."""
    voice = ESPEAK_VOICES.get(speaker, "en")
    speed = ESPEAK_SPEED.get(speaker, 160)
    pitch = ESPEAK_PITCH.get(speaker, 50)

    _espeak_audio_chunks.clear()

    _espeak_lib.espeak_SetVoiceByName(voice.encode())
    _espeak_lib.espeak_SetParameter(1, speed, 0)   # espeakRATE
    _espeak_lib.espeak_SetParameter(2, 100, 0)     # espeakVOLUME
    _espeak_lib.espeak_SetParameter(4, pitch, 0)   # espeakPITCH

    text_bytes = text.encode("utf-8")
    # 0x1000 = espeakCHARS_AUTO, 0x01 = espeakENDPAUSE
    _espeak_lib.espeak_Synth(
        text_bytes, len(text_bytes) + 1, 0, 0, 0, 0x1001, None, None
    )
    _espeak_lib.espeak_Synchronize()

    audio = b"".join(_espeak_audio_chunks)

    with wave.open(output_wav, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(_espeak_sample_rate)
        wf.writeframes(audio)


# ---------------------------------------------------------------------------
# TTS Engine: System espeak-ng
# ---------------------------------------------------------------------------

def _has_system_espeak() -> bool:
    return shutil.which("espeak-ng") is not None


def synthesize_system_espeak(text: str, speaker: str, output_wav: str):
    """Generate speech using system espeak-ng binary."""
    voice = ESPEAK_VOICES.get(speaker, "en")
    speed = ESPEAK_SPEED.get(speaker, 160)
    pitch = ESPEAK_PITCH.get(speaker, 50)
    subprocess.run(
        ["espeak-ng", "-v", voice, "-s", str(speed), "-p", str(pitch),
         "-w", output_wav, text],
        check=True, capture_output=True,
    )


# ---------------------------------------------------------------------------
# TTS Engine: Piper neural TTS
# ---------------------------------------------------------------------------

def _has_piper_models() -> bool:
    """Check if piper voice ONNX models are downloaded."""
    data_dir = Path.home() / ".local" / "share" / "piper-voices"
    for voice_name in PIPER_VOICES.values():
        model = data_dir / f"{voice_name}.onnx"
        config = data_dir / f"{voice_name}.onnx.json"
        if not model.exists() or not config.exists():
            return False
    return True


def synthesize_piper(text: str, speaker: str, output_wav: str):
    """Generate speech using piper-tts neural voices."""
    from piper import PiperVoice

    voice_name = PIPER_VOICES.get(speaker, "en_US-lessac-medium")
    length_scale = PIPER_LENGTH_SCALE.get(speaker, 1.0)
    data_dir = Path.home() / ".local" / "share" / "piper-voices"

    model_path = str(data_dir / f"{voice_name}.onnx")
    config_path = str(data_dir / f"{voice_name}.onnx.json")

    voice = PiperVoice.load(model_path, config_path=config_path)
    with wave.open(output_wav, "wb") as wav_file:
        voice.synthesize(text, wav_file, length_scale=length_scale)


# ---------------------------------------------------------------------------
# TTS Engine: macOS say
# ---------------------------------------------------------------------------

def synthesize_mac_say(text: str, speaker: str, output_wav: str):
    """Generate speech using macOS 'say' command."""
    voice = MAC_VOICES.get(speaker, "Daniel")
    rate = MAC_SPEECH_RATE.get(speaker, 175)
    aiff_file = output_wav.replace(".wav", ".aiff")
    subprocess.run(
        ["say", "-v", voice, "-r", str(rate), "-o", aiff_file, text],
        check=True,
    )
    ffmpeg_bin = find_ffmpeg() or "ffmpeg"
    subprocess.run(
        [ffmpeg_bin, "-y", "-i", aiff_file, "-ar", str(SAMPLE_RATE),
         "-ac", str(CHANNELS), "-sample_fmt", "s16", output_wav],
        check=True, capture_output=True,
    )
    os.unlink(aiff_file)


# ---------------------------------------------------------------------------
# Audio utilities
# ---------------------------------------------------------------------------

def generate_silence(duration_sec: float, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Generate silence as raw PCM bytes."""
    num_samples = int(sample_rate * duration_sec)
    return struct.pack(f"<{num_samples}h", *([0] * num_samples))


def normalize_wav(input_wav: str, output_wav: str, ffmpeg_bin: str = "ffmpeg"):
    """Normalize WAV to consistent format using ffmpeg."""
    subprocess.run(
        [ffmpeg_bin, "-y", "-i", input_wav,
         "-ar", str(SAMPLE_RATE), "-ac", str(CHANNELS),
         "-sample_fmt", "s16", output_wav],
        check=True, capture_output=True,
    )


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


def wav_to_mp3(wav_path: str, mp3_path: str, ffmpeg_bin: str = "ffmpeg"):
    """Convert WAV to MP3 using ffmpeg."""
    subprocess.run(
        [ffmpeg_bin, "-y", "-i", wav_path,
         "-codec:a", "libmp3lame", "-qscale:a", "2", mp3_path],
        check=True, capture_output=True,
    )


def get_wav_duration(wav_path: str) -> float:
    """Get duration of a WAV file in seconds."""
    with wave.open(wav_path, "rb") as w:
        return w.getnframes() / w.getframerate()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
    ffmpeg_bin = ensure_ffmpeg()
    print(f"ffmpeg: {ffmpeg_bin}")

    # Select TTS engine (priority order)
    synthesize_fn = None
    engine_name = ""

    if plat == "macos":
        engine_name = "macOS say"
        synthesize_fn = synthesize_mac_say
        print(f"TTS engine: {engine_name}")
        result = subprocess.run(["say", "-v", "?"], capture_output=True, text=True)
        available = result.stdout
        for speaker, voice in MAC_VOICES.items():
            found = voice.lower() in available.lower()
            print(f"  {speaker}: {voice} ({'found' if found else 'NOT FOUND'})")

    elif plat == "linux":
        # 1. Try piper with pre-downloaded neural voice models
        try:
            import piper  # noqa: F401
            if _has_piper_models():
                engine_name = "piper-tts (neural)"
                synthesize_fn = synthesize_piper
                print(f"TTS engine: {engine_name}")
                for speaker, voice in PIPER_VOICES.items():
                    print(f"  {speaker}: {voice}")
        except ImportError:
            pass

        # 2. Try bundled espeak-ng via piper's shared library
        if synthesize_fn is None and _init_bundled_espeak():
            engine_name = "espeak-ng (bundled via piper-tts)"
            synthesize_fn = synthesize_bundled_espeak
            print(f"TTS engine: {engine_name}")
            for speaker, voice in ESPEAK_VOICES.items():
                print(f"  {speaker}: {voice} @ {ESPEAK_SPEED[speaker]} wpm")

        # 3. Try system espeak-ng
        if synthesize_fn is None and _has_system_espeak():
            engine_name = "espeak-ng (system)"
            synthesize_fn = synthesize_system_espeak
            print(f"TTS engine: {engine_name}")

        if synthesize_fn is None:
            print("ERROR: No TTS engine available.")
            print("Install: pip install piper-tts")
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

            # Normalize to consistent sample rate/format
            normalize_wav(raw_wav, norm_wav, ffmpeg_bin)
            wav_files.append(norm_wav)

        # Concatenate all segments with pauses
        print("\nCombining segments...")
        combined_wav = os.path.join(tmpdir, "combined.wav")
        concatenate_wavs(wav_files, combined_wav, PAUSE_BETWEEN_SPEAKERS)

        duration = get_wav_duration(combined_wav)
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        print(f"Total duration: {minutes}m {seconds}s")

        # Convert to MP3
        print(f"\nConverting to MP3: {OUTPUT_FILE}")
        wav_to_mp3(combined_wav, OUTPUT_FILE, ffmpeg_bin)

    file_size = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    print(f"\nDone! Output: {OUTPUT_FILE} ({file_size:.1f} MB)")
    print("=" * 60)


if __name__ == "__main__":
    main()
