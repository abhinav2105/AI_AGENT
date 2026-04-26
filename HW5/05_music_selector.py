"""
05_music_selector.py
Algorithmic music selection driven by per-cluster colour analysis.

Algorithm
─────────
1. For each cluster, compute mean HSV statistics across all member images.
2. Derive mood parameters from colour:
     energy   = 0.5·saturation + 0.5·brightness
     warmth   = proximity of hue to warm (orange/yellow) tones
3. Map mood → musical parameters:
     tempo    = 60 + energy×60   (60–120 BPM)
     mode     = major if warm+bright > 1.0 else minor
     root_hz  = 110 × 2^(brightness×2)   (110–440 Hz)
4. Synthesise a WAV track:
     - Sustained pad drone (root + fifth)
     - Pentatonic arpeggio cycling through the chosen scale
     - Kick drum on downbeat of every measure
5. Attach the WAV to the matching silent video → final MP4.

All randomness uses seeds derived from the cluster's colour stats, so the
same visual content always produces identical audio.
"""

import json
import wave
import numpy as np
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
CLUSTERS_DIR = ROOT / "output" / "clusters"
VIDEOS_DIR = ROOT / "output" / "videos"
AUDIO_DIR = ROOT / "output" / "audio"

SAMPLE_RATE = 44100

# Semitone intervals for scale arpeggios
MAJOR_PENTATONIC = [0, 2, 4, 7, 9, 12, 14, 16]
MINOR_PENTATONIC = [0, 3, 5, 7, 10, 12, 15, 17]


# ── Colour analysis ────────────────────────────────────────────────────────────

def rgb_to_hsv_stats(arr: np.ndarray) -> dict:
    """arr: H×W×3 uint8 → mean hue/saturation/brightness in [0,1]."""
    f = arr.astype(np.float32) / 255.0
    r, g, b = f[:, :, 0], f[:, :, 1], f[:, :, 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    delta = maxc - minc

    v = maxc
    s = np.where(maxc > 1e-6, delta / maxc, 0.0)

    with np.errstate(invalid="ignore", divide="ignore"):
        h = np.where(
            delta < 1e-6, 0.0,
            np.where(maxc == r, ((g - b) / delta) % 6.0,
            np.where(maxc == g, (b - r) / delta + 2.0,
                                (r - g) / delta + 4.0))
        )
    h = (h / 6.0) % 1.0

    return {
        "hue":        float(h.mean()),
        "saturation": float(s.mean()),
        "brightness": float(v.mean()),
    }


def cluster_colour_profile(image_paths: list) -> dict:
    hues, sats, bris = [], [], []
    for p in image_paths:
        try:
            arr = np.array(Image.open(p).convert("RGB"))
            st = rgb_to_hsv_stats(arr)
            hues.append(st["hue"])
            sats.append(st["saturation"])
            bris.append(st["brightness"])
        except Exception:
            continue

    hues = np.array(hues)
    sats = np.array(sats)
    bris = np.array(bris)

    avg_hue = float(hues.mean())
    avg_sat = float(sats.mean())
    avg_bri = float(bris.mean())

    # Warmth: hues near 0–0.17 or 0.9–1.0 (red/orange/yellow) are warm
    dist_warm = min(avg_hue, abs(avg_hue - 1.0), abs(avg_hue - 0.08))
    warmth = float(np.clip(1.0 - dist_warm / 0.4, 0.0, 1.0))
    energy = 0.5 * avg_sat + 0.5 * avg_bri

    return {
        "avg_hue":        avg_hue,
        "avg_saturation": avg_sat,
        "avg_brightness": avg_bri,
        "warmth":         warmth,
        "energy":         energy,
    }


# ── Music parameter mapping ────────────────────────────────────────────────────

def colour_to_music_params(profile: dict) -> dict:
    energy = profile["energy"]
    warmth = profile["warmth"]
    bright = profile["avg_brightness"]
    sat    = profile["avg_saturation"]

    tempo_bpm    = 60.0 + energy * 60.0            # 60–120 BPM
    mode         = "major" if (warmth + bright) > 1.0 else "minor"
    root_hz      = 110.0 * (2 ** (bright * 2.0))  # 110–440 Hz
    amplitude    = 0.25 + sat * 0.35               # 0.25–0.60

    # Deterministic melody seed from colour stats (not random)
    melody_seed  = int((profile["avg_hue"] * 1000 + profile["avg_saturation"] * 100) % 2**31)

    return {
        "tempo_bpm":    round(tempo_bpm, 1),
        "beat_sec":     60.0 / tempo_bpm,
        "mode":         mode,
        "root_hz":      round(root_hz, 2),
        "amplitude":    round(amplitude, 3),
        "melody_seed":  melody_seed,
    }


# ── Audio synthesis ────────────────────────────────────────────────────────────

def _hz(root: float, semitones: int) -> float:
    return root * (2.0 ** (semitones / 12.0))


def _adsr(n: int, sr: int, attack=0.03, decay=0.08, sustain=0.65, release=0.15) -> np.ndarray:
    env = np.ones(n) * sustain
    a = int(sr * attack)
    d = int(sr * decay)
    r = int(sr * release)
    if a: env[:min(a, n)] = np.linspace(0, 1, min(a, n))
    if d and a < n:
        d_end = min(a + d, n)
        env[a:d_end] = np.linspace(1.0, sustain, d_end - a)
    if r and n > r:
        env[-r:] = np.linspace(sustain, 0.0, r)
    return env


def synthesize_track(params: dict, total_seconds: float) -> np.ndarray:
    sr          = SAMPLE_RATE
    root        = params["root_hz"]
    amp         = params["amplitude"]
    mode        = params["mode"]
    beat_sec    = params["beat_sec"]
    seed        = params["melody_seed"]
    scale       = MAJOR_PENTATONIC if mode == "major" else MINOR_PENTATONIC

    n = int(sr * total_seconds)
    track = np.zeros(n, dtype=np.float64)
    t_full = np.linspace(0, total_seconds, n, endpoint=False)

    # Layer 1: pad drone — root + perfect fifth + octave
    for semi, gain in [(0, 0.30), (7, 0.18), (12, 0.10)]:
        freq = _hz(root, semi)
        lfo  = 1.0 + 0.04 * np.sin(2 * np.pi * 0.18 * t_full)  # gentle tremolo
        track += amp * gain * lfo * np.sin(2 * np.pi * freq * t_full)

    # Layer 2: pentatonic arpeggio (one note per beat, deterministic order)
    rng = np.random.RandomState(seed)
    note_order = rng.permutation(len(scale)).tolist()  # fixed permutation per cluster
    beat_n = int(sr * beat_sec)
    for beat_i in range(int(np.ceil(total_seconds / beat_sec))):
        pos = beat_i * beat_n
        if pos >= n:
            break
        semi   = scale[note_order[beat_i % len(note_order)]]
        octave = 12 * (1 + (beat_i // len(scale)) % 2)  # alternate octaves
        freq   = _hz(root, semi + octave)
        note_n = min(beat_n, n - pos)
        t_note = np.linspace(0, beat_sec, note_n, endpoint=False)
        env    = _adsr(note_n, sr)
        # Richer tone: fundamental + 2nd harmonic
        note   = amp * 0.18 * env * (
            0.7 * np.sin(2 * np.pi * freq * t_note) +
            0.3 * np.sin(2 * np.pi * freq * 2 * t_note)
        )
        track[pos : pos + note_n] += note

    # Layer 3: kick drum on beat 1 of every 4-beat measure
    measure_n = beat_n * 4
    kick_len  = int(sr * 0.18)
    kt        = np.linspace(0, 0.18, kick_len, endpoint=False)
    kick_env  = np.exp(-28 * kt)
    kick_freq = 90 * np.exp(-18 * kt)          # pitch drops from 90 Hz
    kick      = amp * 0.55 * kick_env * np.sin(2 * np.pi * kick_freq * kt)
    for pos in range(0, n, measure_n):
        end = min(pos + kick_len, n)
        track[pos:end] += kick[: end - pos]

    # Normalize with headroom
    peak = np.abs(track).max()
    if peak > 1e-6:
        track = track / peak * 0.88

    return track.astype(np.float32)


def save_wav(audio: np.ndarray, path: Path) -> None:
    pcm = (audio * 32767).astype(np.int16)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())


# ── Video + audio combination ──────────────────────────────────────────────────

def combine(video_path: Path, audio_path: Path, out_path: Path) -> None:
    from moviepy import VideoFileClip, AudioFileClip, concatenate_audioclips

    video = VideoFileClip(str(video_path))
    audio = AudioFileClip(str(audio_path))

    if audio.duration < video.duration:
        loops = int(np.ceil(video.duration / audio.duration))
        audio = concatenate_audioclips([audio] * loops).subclipped(0, video.duration)
    else:
        audio = audio.subclipped(0, video.duration)

    final = video.with_audio(audio)
    final.write_videofile(
        str(out_path), codec="libx264", audio_codec="aac", logger=None
    )
    video.close()
    audio.close()
    print(f"  Final video  → {out_path.relative_to(ROOT)}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

    cluster_meta = json.loads((CLUSTERS_DIR / "cluster_metadata.json").read_text())
    report = {}

    for cid, info in cluster_meta.items():
        dominant = info["dominant_category"]
        print(f"\n── Cluster {cid}  ({dominant}, {info['size']} images) ──")

        profile = cluster_colour_profile(info["image_paths"])
        params  = colour_to_music_params(profile)

        print(f"  Colour  brightness={profile['avg_brightness']:.2f}  "
              f"saturation={profile['avg_saturation']:.2f}  "
              f"warmth={profile['warmth']:.2f}")
        print(f"  Music   mode={params['mode']}  "
              f"tempo={params['tempo_bpm']} BPM  "
              f"root={params['root_hz']} Hz  "
              f"amp={params['amplitude']}")

        silent_path = VIDEOS_DIR / f"cluster_{cid}_silent.mp4"
        if silent_path.exists():
            from moviepy import VideoFileClip
            with VideoFileClip(str(silent_path)) as v:
                video_dur = v.duration
        else:
            print("  Silent video not found; generating 60 s audio track.")
            video_dur = 60.0

        audio = synthesize_track(params, video_dur)
        audio_path = AUDIO_DIR / f"cluster_{cid}_music.wav"
        save_wav(audio, audio_path)
        print(f"  Audio   → {audio_path.relative_to(ROOT)}")

        if silent_path.exists():
            final_path = VIDEOS_DIR / f"cluster_{cid}_final.mp4"
            combine(silent_path, audio_path, final_path)

        report[cid] = {
            "dominant_category": dominant,
            "colour_profile":    profile,
            "music_params":      {k: v for k, v in params.items() if k != "melody_seed"},
        }

    out = ROOT / "output" / "music_selection_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nMusic selection report → {out.relative_to(ROOT)}")
    return report


if __name__ == "__main__":
    main()
