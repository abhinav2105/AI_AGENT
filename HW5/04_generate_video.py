"""
04_generate_video.py
Generates a silent slideshow MP4 for every cluster.
Music is added in 05_music_selector.py.

Output: output/videos/cluster_<id>_silent.mp4
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# MoviePy 2.x uses from moviepy import ... (not moviepy.editor)
CLUSTERS_DIR = ROOT / "output" / "clusters"
VIDEOS_DIR = ROOT / "output" / "videos"

IMG_DURATION = 3.0      # seconds each image is shown
FADE_DURATION = 0.4     # crossfade between images
VIDEO_SIZE = (640, 480)
FPS = 24
MAX_IMAGES = 30         # cap per video to keep file sizes manageable


def make_slideshow(image_paths: list, cluster_id: int):
    from moviepy import ImageClip, concatenate_videoclips
    import moviepy.video.fx as vfx

    clips = []
    for p in image_paths:
        try:
            clip = (
                ImageClip(p)
                .with_duration(IMG_DURATION)
                .resized(VIDEO_SIZE)
                .with_effects([vfx.CrossFadeIn(FADE_DURATION)])
            )
            clips.append(clip)
        except Exception as exc:
            print(f"  Skipping {Path(p).name}: {exc}")

    if not clips:
        raise RuntimeError(f"No valid images for cluster {cluster_id}")

    video = concatenate_videoclips(clips, method="compose", padding=-FADE_DURATION)
    return video


def main():
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

    meta = json.loads((CLUSTERS_DIR / "cluster_metadata.json").read_text())
    sorted_clusters = sorted(meta.items(), key=lambda x: x[1]["size"], reverse=True)

    for cid, info in sorted_clusters:
        out_path = VIDEOS_DIR / f"cluster_{cid}_silent.mp4"
        if out_path.exists():
            print(f"Cluster {cid}: silent video already exists, skipping.")
            continue

        paths = info["image_paths"][:MAX_IMAGES]
        dominant = info["dominant_category"]
        print(f"\nCluster {cid} ({dominant}, {info['size']} images) "
              f"→ rendering {len(paths)} frames …")

        try:
            video = make_slideshow(paths, int(cid))
            video.write_videofile(
                str(out_path),
                fps=FPS,
                codec="libx264",
                audio=False,
                logger=None,
            )
            video.close()
            print(f"  Saved → {out_path.relative_to(ROOT)}")
        except Exception as exc:
            print(f"  ERROR cluster {cid}: {exc}")

    print("\nVideo generation complete.")


if __name__ == "__main__":
    main()
