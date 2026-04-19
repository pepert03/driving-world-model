"""Generate a composite image per preset: mean100 reward graph + 4 video frames."""

import cv2
import glob
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

REPO_ROOT = Path(__file__).resolve().parent.parent

PRESETS = [
    "car_racing",
    "car_racing2",
    "walker_walk",
]

# Presets that only get frames (no graph)
# FRAMES_ONLY = {"rainbow_pixel_humanoid", "pixel_hopper", "pixel_walker2d"}


def extract_frames(video_path: Path, n_frames: int = 4) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < n_frames:
        raise ValueError(f"{video_path}: only {total} frames, need {n_frames}")
    indices = [int(i * (total - 1) / (n_frames - 1)) for i in range(n_frames)]
    indices[-1] = (
        total * (n_frames * 10 - 1) // (n_frames * 10)
    )  # Ensure last frame is near the end
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"{video_path}: failed to read frame {idx}")
        frames.append(frame)
    cap.release()
    return frames


def load_mean100(preset: str) -> tuple[list[int], list[float]]:
    """Load mean100 reward from TensorBoard events."""
    tb_dir = REPO_ROOT / "runs" / preset / "tensorboard"
    pattern = str(tb_dir / "events.out.*")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No TensorBoard events found in {tb_dir}")

    ea = EventAccumulator(files[0], size_guidance={"scalars": 0})
    ea.Reload()

    scalars = ea.Scalars("reward/mean_100")
    episodes = [s.step for s in scalars]
    values = [s.value for s in scalars]
    return episodes, values


def render_graph(episodes: list[int], values: list[float], target_h: int) -> np.ndarray:
    """Render a mean100 reward graph and return it as a BGR numpy array."""
    dpi = 100
    # Make graph width roughly equal to target height for a square-ish plot
    fig_w = target_h / dpi
    fig_h = target_h / dpi

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.plot(episodes, values, color="#1f77b4", linewidth=0.8)
    ax.set_xlabel("Episode", fontsize=8)
    ax.set_ylabel("Mean 100 Reward", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    # Render to numpy array
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)
    plt.close(fig)

    # RGBA -> BGR for OpenCV
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    # Resize to exact target height
    h, w = img_bgr.shape[:2]
    if h != target_h:
        scale = target_h / h
        new_w = int(w * scale)
        img_bgr = cv2.resize(img_bgr, (new_w, target_h), interpolation=cv2.INTER_AREA)

    return img_bgr


def main():
    for preset in PRESETS:
        video = REPO_ROOT / "runs" / preset / "eval" / "best.mp4"
        if not video.exists():
            print(f"SKIP  {preset}: {video} not found")
            continue

        # Extract 4 frames and concatenate horizontally
        frames = extract_frames(video)
        frames_strip = np.concatenate(frames, axis=1)
        strip_h = frames_strip.shape[0]

        # if preset in FRAMES_ONLY:
        #     out = video.parent / "frames.png"
        #     cv2.imwrite(str(out), frames_strip)
        #     print(f"OK    {preset} (frames only) -> {out}")
        #     continue

        # Load mean100 data from TensorBoard
        try:
            episodes, values = load_mean100(preset)
        except Exception as e:
            print(f"SKIP  {preset} graph: {e}")
            # Save frames-only as fallback
            out = video.parent / "frames.png"
            cv2.imwrite(str(out), frames_strip)
            print(f"OK    {preset} (frames only) -> {out}")
            continue

        # Render graph with same height as frames strip
        graph_img = render_graph(episodes, values, strip_h)

        # Combine: graph on the left, frames on the right
        composite = np.concatenate([graph_img, frames_strip], axis=1)

        out = video.parent / "frames.png"
        cv2.imwrite(str(out), composite)
        print(f"OK    {preset} -> {out}")


if __name__ == "__main__":
    main()
