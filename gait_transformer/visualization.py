"""
Duplicates some of the methods from pose_pipeline to keep this self contained
"""

import os
import cv2
import tempfile
import shutil
import subprocess
import numpy as np
from typing import Callable, Tuple
from tqdm import tqdm


def video_overlay(
    video: str,
    output_name: str,
    callback: Callable[[np.ndarray, int], np.ndarray],
    downsample: int = 4,
    codec: str = "MP4V",
    compress: bool = True,
    bitrate: str = "1M",
    max_frames: None | int = None,
):
    """Process a video and create overlay image

    Args:
        video (str): filename for source
        output_name (str): output filename
        callback (fn(im, idx) -> im): method to overlay frame
    """

    cap = cv2.VideoCapture(video)

    # get info
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # configure output
    output_size = (int(w / downsample), int(h / downsample))

    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_name, fourcc, fps, output_size)

    if max_frames:
        total_frames = max_frames

    for idx in tqdm(range(total_frames)):

        ret, frame = cap.read()
        if not ret or frame is None:
            break

        # process image in RGB format
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out_frame = callback(frame, idx)

        # move back to BGR format and write to movie
        out_frame = cv2.cvtColor(out_frame, cv2.COLOR_RGB2BGR)
        out_frame = cv2.resize(out_frame, output_size)
        out.write(out_frame)

    out.release()
    cap.release()

    if compress:
        fd, temp = tempfile.mkstemp(suffix=".mp4")
        subprocess.run(["ffmpeg", "-y", "-i", output_name, "-hide_banner", "-loglevel", "error", "-c:v", "libx264", "-b:v", bitrate, temp])
        os.close(fd)
        shutil.move(temp, output_name)


def draw_keypoints(
    image: np.ndarray,
    keypoints: np.ndarray,
    radius: float = 10,
    threshold: float = 0.2,
    color: Tuple = (255, 255, 255),
    border_color: Tuple = (0, 0, 0),
):
    """Draw the keypoints on an image"""
    image = image.copy()
    keypoints = keypoints.copy()
    keypoints[..., 0] = np.clip(keypoints[..., 0], 0, image.shape[1])
    keypoints[..., 1] = np.clip(keypoints[..., 1], 0, image.shape[0])
    for i in range(keypoints.shape[0]):
        if keypoints[i, -1] > threshold:
            cv2.circle(
                image,
                (int(keypoints[i, 0]), int(keypoints[i, 1])),
                radius,
                border_color,
                -1,
            )
            if radius > 2:
                cv2.circle(
                    image,
                    (int(keypoints[i, 0]), int(keypoints[i, 1])),
                    radius - 2,
                    color,
                    -1,
                )
    return image


def get_trace_overlay_fn(phases, stride, walking_prob=None):

    phases = np.reshape(phases, [-1, 4, 2])[:, :, 0]
    # phases = np.arctan2(phases[:, :, 1], phases[:, :, 0])

    def plot_trace(image, x, y, color=[255, 255, 255], width=3):
        a = np.stack([x, y], axis=-1).astype(np.int32)

        image = image.copy()
        for index, item in enumerate(a[1:]):
            cv2.line(image, item, a[index], color, width)

        return image

    right_color = [255, 40, 40]
    left_color = [40, 40, 255]

    def plot_traces(image, idx):
        height, width, _ = image.shape

        # rescaling code as original version was designed for 1920x1080
        def scale_y(y, max_y):
            return int(y * (height / max_y))

        def scale_x(x):
            return int(x * (width / 1080))

        if idx > 50 and idx < len(phases) - 50:
            x = np.linspace(scale_x(750), scale_x(1050), 100)

            idx_range = np.array(range(idx - 50, idx + 50))

            # foot pos
            y = scale_y(100, 1300) - stride[idx_range, 0] * scale_y(100, 1300)
            image = plot_trace(image, x, y, right_color)
            y = scale_y(100, 1300) - stride[idx_range, 1] * scale_y(100, 1300)
            image = plot_trace(image, x, y, left_color)

            cv2.putText(image, "Foot pos", (scale_x(500), scale_y(100, 1300)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # foot vel
            y = scale_y(300, 1300) - stride[idx_range, 3] * scale_y(20, 1300)
            image = plot_trace(image, x, y, right_color)
            y = scale_y(300, 1300) - stride[idx_range, 4] * scale_y(20, 1300)
            image = plot_trace(image, x, y, left_color)

            cv2.putText(image, "Foot vel", (scale_x(500), scale_y(300, 1300)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # pelvis vel
            y = scale_y(500, 1300) - stride[idx_range, 2] * scale_y(100, 1300)
            image = plot_trace(image, x, y, [0, 128, 0])
            image = plot_trace(image, x, scale_y(500, 1300) - y * 0, [0, 0, 0])

            cv2.putText(image, "Pelvis vel", (scale_x(500), scale_y(500, 1300)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # hip
            y = scale_y(700, 1300) - stride[idx_range, 5] * scale_y(100, 1300)
            image = plot_trace(image, x, y, right_color)
            y = scale_y(700, 1300) - stride[idx_range, 6] * scale_y(100, 1300)
            image = plot_trace(image, x, y, left_color)
            image = plot_trace(image, x, scale_y(700, 1300) - y * 0, [0, 0, 0])

            cv2.putText(image, "Hip", (scale_x(550), scale_y(700, 1300)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # knee
            y = scale_y(900, 1300) - stride[idx_range, 7] * scale_y(100, 1300)
            image = plot_trace(image, x, y, right_color)
            y = scale_y(900, 1300) - stride[idx_range, 8] * scale_y(100, 1300)
            image = plot_trace(image, x, y, left_color)
            image = plot_trace(image, x, scale_y(900, 1300) + y * 0, [0, 0, 0])

            cv2.putText(image, "Knee", (scale_x(550), scale_y(900, 1300)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # phases
            y = scale_y(1100, 1300) - phases[idx_range, 0] * scale_y(100, 1300)
            image = plot_trace(image, x, y, left_color)
            y = scale_y(1100, 1300) - phases[idx_range, 1] * scale_y(100, 1300)
            image = plot_trace(image, x, y, right_color)
            y = scale_y(1100, 1300) - phases[idx_range, 2] * scale_y(100, 1300)
            image = plot_trace(image, x, y, [c // 2 for c in left_color])
            y = scale_y(1100, 1300) - phases[idx_range, 3] * scale_y(100, 1300)
            image = plot_trace(image, x, y, [c // 2 for c in right_color])

            cv2.putText(image, "Phases", (scale_x(550), scale_y(1100, 1300)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            if walking_prob is not None:
                # walking prob
                y = scale_y(1300, 1300) - walking_prob[idx_range] * scale_y(50, 1300)
                image = plot_trace(image, x, y, [0, 128, 0])
                image = plot_trace(image, x, scale_y(1300, 1300) - 0 * y, [0, 0, 0])
                cv2.putText(image, "Prob", (scale_x(550), scale_y(1300, 1300)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.line(image, (int(np.mean(x)), scale_y(1300, 1300)), (int(np.mean(x)), 0), (255, 255, 255), 4)

        return image

    return plot_traces


def make_overlay(video: str, phases: np.array, stride: np.array, keypoints: np.array, outname=None):
    """
    Make a gait transformer overlay video

    Args:
        video: video file name
        phases: gait phases quadrature encoded
        stride: stride features
        keypoints: keypoints (time x 17 x 3)
        outname: output file name (optional, otherwise returns temp file)
    """

    import tempfile
    import os

    plot_traces = get_trace_overlay_fn(phases, stride)

    phases = np.reshape(phases, [-1, 4, 2])
    phases = np.arctan2(phases[:, :, 1], phases[:, :, 0])
    left_down = phases[:, 0] < phases[:, 2]
    right_down = phases[:, 1] < phases[:, 3]

    ankle_idx = np.array([13, 10])

    def overlay_fn(image, idx):
        image = draw_keypoints(image, keypoints[idx], color=(255, 255, 255))

        down = [left_down[idx], right_down[idx]]

        if down[0]:
            image = draw_keypoints(image, keypoints[idx, ankle_idx[0] : ankle_idx[0] + 1], radius=15, color=(0, 255, 0))
        else:
            image = draw_keypoints(image, keypoints[idx, ankle_idx[0] : ankle_idx[0] + 1], radius=15, color=(255, 0, 0))

        if down[1]:
            image = draw_keypoints(image, keypoints[idx, ankle_idx[1] : ankle_idx[1] + 1], radius=15, color=(0, 255, 0))
        else:
            image = draw_keypoints(image, keypoints[idx, ankle_idx[1] : ankle_idx[1] + 1], radius=15, color=(255, 0, 0))

        image = plot_traces(image, idx)

        return image

    fd, out_file_name = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    video_overlay(video, out_file_name, overlay_fn, downsample=1)

    if outname is not None:
        import shutil

        shutil.move(out_file_name, outname)
        return outname

    return out_file_name


def jupyter_embed_video(video_filename: str, height: int | None = None):

    from IPython.display import HTML
    from base64 import b64encode

    video = open(video_filename, "rb").read()
    video_encoded = b64encode(video).decode("ascii")
    if height is None:
        video_tag = f'<video controls src="data:video/mp4;base64,{video_encoded}">'
    else:
        video_tag = f'<video controls height="{height}" src="data:video/mp4;base64,{video_encoded}">'

    return HTML(video_tag)
