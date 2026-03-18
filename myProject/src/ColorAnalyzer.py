import cv2
import numpy as np


class ColorAnalyzer:
    def __init__(self, video_path, scenes, sample_step=0.5, resize_to=(150, 150)):
        self.video_path = video_path
        self.scenes = scenes
        self.sample_step = sample_step
        self.resize_to = resize_to

    def analyze(self):
        cap = cv2.VideoCapture(self.video_path)

        fps = cap.get(cv2.CAP_PROP_FPS)

        brightness = []
        saturation = []
        dom_colors = []
        scene_color_strips = []

        for start, end in self.scenes:
            sats = []
            bris = []
            frame_colors = []

            for t in np.arange(start, end, self.sample_step):
                frame_idx = int(t * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

                ok, frame = cap.read()
                if not ok:
                    continue

                frame = cv2.resize(frame, self.resize_to)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)

                sats.append(np.mean(hsv[:, :, 1] / 255.0))
                bris.append(np.mean(hsv[:, :, 2] / 255.0))

                frame_colors.append(np.mean(frame_rgb.reshape(-1, 3), axis=0))

            frame_colors = np.array(frame_colors)

            dom_colors.append(np.mean(frame_colors, axis=0))
            saturation.append(np.mean(sats))
            brightness.append(np.mean(bris))
            scene_color_strips.append(frame_colors)

        cap.release()

        return (
            np.array(brightness),
            np.array(saturation),
            np.array(dom_colors),
            scene_color_strips
        )