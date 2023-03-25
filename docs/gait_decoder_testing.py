import datajoint as dj
import numpy as np

import pose_pipeline
from pose_pipeline import *

from gait_analysis.fetch_training_dataset import get_trials, get_dataset, field_types
from gait_analysis.gait_decoder_training import ModelTraining, ModelTrainingParam

## The code below duplicates code from sensor_fusion.gait_dj

gait_schema = dj.schema("gait_transformer_training")


@gait_schema
class GaitPhaseStrideTransformerSelect(dj.Manual):
    definition = """
    -> ModelTraining
    -> LiftingPerson
    window_len        :  int
    """


@gait_schema
class GaitPhaseStrideTransformer(dj.Computed):
    definition = """
    -> GaitPhaseStrideTransformerSelect
    ---
    phase      : longblob
    stride     : longblob
    """

    def make(self, key):

        from gait_analysis.gait_phase_transformer import gait_phase_stride_inference

        regressor = (ModelTraining & key).get_model()
        keypoints3d = (LiftingPerson & key).fetch1("keypoints_3d")

        # for projects using the emgimu schema we manually can add these entries for
        # subjects
        from sensor_fusion.emgimu_session import Height, FirebaseSession, DualRecording

        if len(FirebaseSession.AppVideo & key) > 0:
            height = (Height & (FirebaseSession.AppVideo & key)).fetch1("height_mm")
        elif len(DualRecording.DualRecordingVideo & key) > 0:
            height = (Height & (DualRecording.DualRecordingVideo & key)).fetch1("height_mm")
        else:
            raise Exception(f"Could not find Firebase.AppVideo or DualRecording.DualRecordingVideo match for {key}")

        L = key["window_len"]
        phase, stride = gait_phase_stride_inference(keypoints3d, height, regressor, L)

        key["phase"] = phase
        key["stride"] = stride
        self.insert1(key)


@gait_schema
class GaitPhaseStrideKalman(dj.Computed):
    definition = """
    -> GaitPhaseStrideTransformer
    ---
    states        : longblob
    predictions   : longblob
    errors        : longblob
    kalman_phases : longblob
    left_down     : longblob
    left_up       : longblob
    right_down    : longblob
    right_up      : longblob
    """

    def make(self, key):
        import gait_analysis.gait_phase_kalman

        timestamps = (pose_pipeline.VideoInfo & key).fetch1("timestamps")
        timestamps = np.array([(t - timestamps[0]).total_seconds() for t in timestamps])

        phase = (GaitPhaseStrideTransformer & key).fetch1("phase")
        phase = np.take(phase, [0, 4, 1, 5, 2, 6, 3, 7], axis=-1)

        states, predictions, errors = gait_analysis.gait_phase_kalman.gait_kalman_smoother(phase)
        kalman_phases = gait_analysis.gait_phase_kalman.compute_phases(states)
        detected_events = gait_analysis.gait_phase_kalman.get_event_times(states, timestamps)

        key["states"] = np.array(states)
        key["predictions"] = np.array(predictions)
        key["errors"] = np.array(errors)
        key["kalman_phases"] = np.array(kalman_phases)
        key["left_down"] = np.array(detected_events["left_down"])
        key["left_up"] = np.array(detected_events["left_up"])
        key["right_down"] = np.array(detected_events["right_down"])
        key["right_up"] = np.array(detected_events["right_up"])

        self.insert1(key)


def detect_walking(key):

    from scipy.signal import medfilt
    from pose_pipeline import VideoInfo, PersonBbox, TopDownPerson

    states, predictions, errors = (GaitPhaseStrideKalman & key).fetch1("states", "predictions", "errors")
    kp2d = (TopDownPerson & key).fetch1("keypoints")
    height, width = (VideoInfo & key).fetch1("height", "width")
    bbox, present = (PersonBbox & key).fetch1("bbox", "present")

    def hyst(x, th_lo, th_hi, initial=False):
        hi = x >= th_hi
        lo_or_hi = (x <= th_lo) | hi
        ind = np.nonzero(lo_or_hi)[0]
        if not ind.size:  # prevent index error if ind is empty
            return np.zeros_like(x, dtype=bool) | initial
        cnt = np.cumsum(lo_or_hi)  # from 0 to len(x)
        return np.where(cnt, hi[ind[cnt - 1]], initial)

    def deglitch(x):
        return medfilt(x, 11)

    joint_names = TopDownPerson.joint_names()
    joint_idx = np.array([joint_names.index(j) for j in ["Right Ankle", "Left Ankle"]])

    prob_err = np.exp(-errors * 2)
    prob_speed1 = 1 / (1 + np.exp(-2 * (states[:, 1] - np.pi / 2)))
    prob_speed2 = np.exp(-0.1 * np.abs(states[:, 1] - medfilt(states[:, 1], 31)))
    prob_speed = prob_speed1 * prob_speed2
    prob_keypoints = np.min(kp2d[:, joint_idx, 2], axis=-1)

    bbox_tlhw = bbox.copy()
    bbox_tlhw[:, 2:] = bbox_tlhw[:, :2] + bbox_tlhw[:, 2:]
    present.astype(np.float32)

    margin = (bbox_tlhw - np.array([[0, 0, width, height]])) * np.array([[1.0, 1.0, -1.0, -1.0]])
    margin = (1 + np.tanh(margin * 10)) * 0.5  # need 10-20 pixels before fully confident
    prob_bbox = np.prod(margin, axis=-1) * present.astype(np.float32)

    prob_keypoints = np.exp(-2 * np.abs((0.9 - prob_keypoints)))
    walking_prob = prob_err * prob_speed  # * prob_keypoints * prob_bbox # Disable bbox detection for now

    thresh = hyst(deglitch(walking_prob), 0.65, 0.35)

    keep_idx = np.nonzero(thresh)[0]
    thresh_no_short = thresh.copy() * 0
    segments = []
    if len(keep_idx) > 0:
        breaks = np.where(np.diff(keep_idx) != 1)[0]
        breaks = np.array([-1, *breaks, -1])
        segments = zip(keep_idx[breaks[:-1] + 1], keep_idx[breaks[1:]])
        segments = [(s1, s2) for s1, s2 in segments if (s2 - s1) > 100]

        for s in segments:
            thresh_no_short[s[0] : s[1]] = 1

    return {
        "walking": thresh_no_short,
        "walking_prob": walking_prob,
        "prob_err": prob_err,
        "prob_speed": prob_speed,
        "prob_keypoints": prob_keypoints,
        "prob_bbox": prob_bbox,
        "segments": segments,
    }


@gait_schema
class GaitPhaseWalking(dj.Computed):
    definition = """
    -> GaitPhaseStrideKalman
    ---
    walking         : longblob
    walking_prob    : longblob
    prob_err        : longblob
    prob_speed      : longblob
    prob_keypoints  : longblob
    prob_bbox       : longblob
    segments        : longblob
    """

    class WalkingSegment(dj.Part):
        definition = """
        -> GaitPhaseWalking
        segment_start        : int
        segment_end          : int
        ---
        frames         : int
        cadence        : float
        velocity       : float
        stance_left    : float
        stance_right   : float
        ss_left        : float
        ss_right       : float
        dst            : float
        length_left    : float
        length_right   : float
        """

    def make(self, key):
        from gait_analysis.fetch_training_dataset import get_dataset, mocap_keep
        import gait_analysis.mocap_sync

        walking = detect_walking(key)
        walking.update(key)
        self.insert1(walking)

        print(key)

        dt = 1.0 / (pose_pipeline.VideoInfo & key).fetch1("fps")
        timestamps = (pose_pipeline.VideoInfo & key).fetch_timestamps()

        fetch = GaitPhaseStrideKalman * GaitPhaseStrideTransformer & key
        states, stride, phase, kalman_phases, left_down, right_down = fetch.fetch1(
            "states", "stride", "phase", "kalman_phases", "left_down", "right_down"
        )

        def parse_frame(f):
            cadence = (states[f[1], 0] - states[f[0], 0]) / (2 * np.pi) / ((f[1] - f[0]) * dt) * 60 * 2

            fields = [f for f in (ModelTrainingParam & key).get_fields() if f != "Phase"]

            pelvis_idx = fields.index("vPELO")
            velocity = np.mean(stride[f[0] : f[1], pelvis_idx])

            avg_phase = 100 - np.mean(states[f[0] : f[1], 2:], axis=0) * 100 / (2 * np.pi)

            stance_left = avg_phase[1]
            stance_right = 100 - (avg_phase[0] - avg_phase[2])
            ss_left = avg_phase[0] - avg_phase[2]
            ss_right = 100 - avg_phase[1]
            dst = 100 - ss_left - ss_right

            right_idx = fields.index("RTOE")  # gait_analysis.mocap_sync.default_coco_map['Right Ankle'])
            left_idx = fields.index("LTOE")  # gait_analysis.mocap_sync.default_coco_map['Left Ankle'])

            print(len(timestamps), len(stride))
            if f[1] == len(timestamps):
                f = (f[0], f[1] - 1)

            left_length = np.interp(
                left_down[np.logical_and(left_down >= timestamps[f[0]], left_down < timestamps[f[1]])],
                timestamps,
                stride[:, left_idx] - stride[:, right_idx],
            )
            left_length = np.mean(left_length)

            right_length = np.interp(
                right_down[np.logical_and(right_down >= timestamps[f[0]], right_down < timestamps[f[1]])],
                timestamps,
                stride[:, right_idx] - stride[:, left_idx],
            )
            right_length = np.mean(right_length)

            if np.isnan(left_length):
                left_length = 0
            if np.isnan(right_length):
                right_length = 0

            return {
                "cadence": float(cadence),
                "velocity": velocity,
                "frames": f[1] - f[0],
                "stance_left": stance_left,
                "stance_right": stance_right,
                "ss_left": ss_left,
                "ss_right": ss_right,
                "dst": dst,
                "length_left": left_length,
                "length_right": right_length,
            }

        for seg in walking["segments"]:
            stats = parse_frame(seg)
            stats.update(key)
            stats["segment_start"] = seg[0]
            stats["segment_end"] = seg[1]
            GaitPhaseWalking.WalkingSegment.insert1(stats)


@gait_schema
class Steps(dj.Computed):
    definition = """
    -> GaitPhaseWalking.WalkingSegment
    step_time   : decimal(10,5)
    ---
    side        : enum('Left', 'Right')
    velocity    : float
    cadence     : float
    ss_left     : float
    ss_right    : float
    dst         : float
    length      : float
    """

    def make(self, key):

        import scipy.signal
        from gait_analysis.fetch_training_dataset import get_dataset, mocap_keep

        dt = 1.0 / (pose_pipeline.VideoInfo & key).fetch1("fps")
        timestamps = (pose_pipeline.VideoInfo & key).fetch_timestamps()
        states, stride, left_down, right_down = (GaitPhaseStrideKalman * GaitPhaseStrideTransformer & key).fetch1(
            "states", "stride", "left_down", "right_down"
        )

        def parse_events(times, side="Left"):

            times = times[
                np.logical_and(times >= timestamps[key["segment_start"]], times < timestamps[key["segment_end"]])
            ]

            cadence = np.interp(times, timestamps, states[:, 1]) * 120 / (2 * np.pi)

            filt = 1 + int(np.median(cadence) / 120 / dt / 2) * 3
            filt = 1 + int(filt / 2) * 2

            fields = [f for f in (ModelTrainingParam & key).get_fields() if f != "Phase"]

            pelvis_idx = fields.index("vPELO")

            velocity = np.interp(
                times, timestamps, scipy.signal.medfilt(stride[:, pelvis_idx], filt)
            )  # probably should average over cycle

            phase = 100 - states[:, 2:] * 100 / (2 * np.pi)
            phase_time = lambda x: np.interp(times, timestamps, phase[:, x])

            stance_left = phase_time(1)
            ss_left = phase_time(0) - phase_time(2)
            ss_right = 100 - stance_left
            dst = 100 - ss_left - ss_right

            right_idx = fields.index("RTOE")
            left_idx = fields.index("LTOE")

            if side == "Left":
                length = np.interp(times, timestamps, stride[:, left_idx] - stride[:, right_idx])
            else:
                length = np.interp(times, timestamps, stride[:, right_idx] - stride[:, left_idx])

            entries = []
            for i, x in enumerate(zip(times, velocity, cadence, ss_left, ss_right, dst, length)):
                entry = key.copy()
                entry["side"] = side
                entry["step_time"] = x[0]
                entry["velocity"] = x[1]
                entry["cadence"] = x[2]
                entry["ss_left"] = x[3]
                entry["ss_right"] = x[4]
                entry["dst"] = x[5]
                entry["length"] = x[6]
                entries.append(entry)
            self.insert(entries)

        parse_events(left_down)
        parse_events(right_down, "Right")


def get_trace_overlay_fn(phases, stride, walking, walking_prob, model_fields):

    import cv2

    # phases = np.reshape(phases, [-1, 4, 2])[:, :, 0]
    # phases = np.arctan2(phases[:, :, 1], phases[:, :, 0])

    def plot_trace(image, x, y, color=[255, 255, 255]):
        width = int(image.shape[0] * 4 / 1080)
        a = np.stack([x, y], axis=-1).astype(np.int32)

        image = image.copy()
        for index, item in enumerate(a[1:]):
            cv2.line(image, item, a[index], color, width)

        return image

    def plot_traces(image, idx):

        height, width, _ = image.shape
        spacing = 0.14 * height
        scale = 0.065 * height
        label_x = int(0.5 * width)

        def get_y(i):
            return spacing * 0.5 + spacing * i

        def text(image, label, coord, font, fontScale, color, thickness):
            fontScale = fontScale * 3
            thickness = thickness * 3
            (label_width, _), _ = cv2.getTextSize(label, font, fontScale, thickness)
            cv2.putText(image, label, (coord[0] - label_width, coord[1]), font, fontScale, color, thickness)

        window = 30

        right_color = [255, 40, 40]
        left_color = [40, 40, 255]

        if idx > window and idx < len(walking) - window:
            x = np.linspace(label_x, 0.99 * width, window * 2)

            idx = np.array(range(idx - window, idx + window))

            # foot pos
            ch = 0
            y = get_y(ch) - stride[idx, model_fields.index("RTOE")] * scale
            image = plot_trace(image, x, y, right_color)
            y = get_y(ch) - stride[idx, model_fields.index("LTOE")] * scale
            image = plot_trace(image, x, y, left_color)

            text(image, "Foot pos", [label_x, int(get_y(ch))], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # foot vel
            ch = 1
            y = get_y(ch) - stride[idx, model_fields.index("vRTOE")] * scale / 5
            image = plot_trace(image, x, y, right_color)
            y = get_y(ch) - stride[idx, model_fields.index("vLTOE")] * scale / 5
            image = plot_trace(image, x, y, left_color)

            text(image, "Foot vel", [label_x, int(get_y(ch))], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # pelvis vel
            ch = 2
            y = get_y(ch) - stride[idx, model_fields.index("vPELO")] * scale
            image = plot_trace(image, x, y, [0, 128, 0])
            image = plot_trace(image, x, get_y(ch) - y * 0, [0, 0, 0])

            text(image, "Pelvis vel", [label_x, int(get_y(ch))], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # hip
            ch = 3
            y = get_y(ch) - stride[idx, model_fields.index("RHipAngles")] * scale  # / np.pi
            image = plot_trace(image, x, y, right_color)
            y = get_y(ch) - stride[idx, model_fields.index("LHipAngles")] * scale  # / np.pi
            image = plot_trace(image, x, y, left_color)
            image = plot_trace(image, x, get_y(ch) - y * 0, [0, 0, 0])

            text(image, "Hip", [label_x, int(get_y(ch))], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # knee
            ch = 4
            y = get_y(ch) - stride[idx, model_fields.index("RKneeAngles")] * scale  # / np.pi
            image = plot_trace(image, x, y, right_color)
            y = get_y(ch) - stride[idx, model_fields.index("LKneeAngles")] * scale  # / np.pi
            image = plot_trace(image, x, y, left_color)
            image = plot_trace(image, x, get_y(ch) + y * 0, [0, 0, 0])

            text(image, "Knee", [label_x, int(get_y(ch))], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # phases  [left_down_phase, right_down_phase, left_up_phase, right_up_phase]
            ch = 5
            y = get_y(ch) - phases[idx, 0] * scale
            image = plot_trace(image, x, y, left_color)
            y = get_y(ch) - phases[idx, 1] * scale
            image = plot_trace(image, x, y, right_color)
            y = get_y(ch) - phases[idx, 2] * scale
            image = plot_trace(image, x, y, [c // 2 for c in left_color])
            y = get_y(ch) - phases[idx, 3] * scale
            image = plot_trace(image, x, y, [c // 2 for c in right_color])

            text(image, "Phases", [label_x, int(get_y(ch))], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # walking prob
            ch = 6
            y = get_y(ch) - walking_prob[idx] * scale / 2
            image = plot_trace(image, x, y, [0, 128, 0])
            image = plot_trace(image, x, get_y(ch) - 0 * y, [0, 0, 0])
            text(image, "Prob", [label_x, int(get_y(ch))], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.line(image, [int(np.mean(x)), int(get_y(ch))], [int(np.mean(x)), 0], (255, 255, 255), 4)
        return image

    return plot_traces


@gait_schema
class GaitPhaseWalkingVideo(dj.Computed):
    definition = """
    -> GaitPhaseWalking
    -> BlurredVideo
    ---
    output_video      : attach@localattach    # datajoint managed video file
    """

    def make(self, key):

        from pose_pipeline.utils.visualization import video_overlay, draw_keypoints
        import tempfile
        import os

        # height, width, timestamps = (VideoInfo & key).fetch1('height', 'width', 'timestamps')
        coco_keypoints = (TopDownPerson & key).fetch1("keypoints")
        phases, walking, walking_prob, stride = (GaitPhaseWalking * GaitPhaseStrideTransformer & key).fetch1(
            "phase", "walking", "walking_prob", "stride"
        )

        model_fields = [f for f in (ModelTrainingParam & key).get_fields() if f != "Phase"]

        walking_prob = walking_prob.copy()
        walking_prob[np.isnan(walking_prob)] = 0

        bbox_fn = PersonBbox.get_overlay_fn(key)

        plot_traces = get_trace_overlay_fn(phases, stride, walking, walking_prob, model_fields)

        kalman_phases = (GaitPhaseStrideKalman & key).fetch1("kalman_phases")
        left_down = kalman_phases[:, 0] > kalman_phases[:, 2]
        right_down = kalman_phases[:, 1] > kalman_phases[:, 3]

        def frame_phase(idx):
            phase = phases[idx]
            down = [left_down[idx], right_down[idx]]

            return walking[idx], phase, down

        def overlay_fn(image, idx):
            walking, phase, down = frame_phase(idx)

            ankle_idx = [TopDownPerson.joint_names().index(j) for j in ["Left Ankle", "Right Ankle"]]

            image = draw_keypoints(image, coco_keypoints[idx], color=(0, 0, 255) if walking else (255, 255, 255))
            image = bbox_fn(image, idx)

            if walking:
                if down[0]:
                    image = draw_keypoints(
                        image, coco_keypoints[idx, ankle_idx[0] : ankle_idx[0] + 1], radius=15, color=(0, 255, 0)
                    )
                else:
                    image = draw_keypoints(
                        image, coco_keypoints[idx, ankle_idx[0] : ankle_idx[0] + 1], radius=15, color=(255, 0, 0)
                    )

                if down[1]:
                    image = draw_keypoints(
                        image, coco_keypoints[idx, ankle_idx[1] : ankle_idx[1] + 1], radius=15, color=(0, 255, 0)
                    )
                else:
                    image = draw_keypoints(
                        image, coco_keypoints[idx, ankle_idx[1] : ankle_idx[1] + 1], radius=15, color=(255, 0, 0)
                    )

            image = plot_traces(image, idx)

            return image

        video = (BlurredVideo & key).fetch1("output_video")

        _, out_file_name = tempfile.mkstemp(suffix=".mp4")
        video_overlay(video, out_file_name, overlay_fn, downsample=1)
        key["output_video"] = out_file_name

        self.insert1(key)

        os.remove(out_file_name)
        os.remove(video)
