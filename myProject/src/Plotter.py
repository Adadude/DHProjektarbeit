import matplotlib.pyplot as plt
import numpy as np


class Plotter:
    def __init__(self, title="Color and Emotion Analysis", jitter_emotion=0.02, jitter_visual=0.01, random_state=42):
        self.title = title
        self.jitter_emotion = jitter_emotion
        self.jitter_visual = jitter_visual
        self.rng = np.random.default_rng(random_state)

    def _jitter(self, values, std):
        values = np.asarray(values, dtype=float)
        return values + self.rng.normal(0, std, len(values))

    def _build_scene_barcode(self, scene_color_strips, height=20):
        scene_blocks = []

        for strip in scene_color_strips:
            strip = np.asarray(strip, dtype=float)

            if strip.ndim == 1:
                strip = strip.reshape(1, 3)

            strip = np.clip(strip / 255.0, 0, 1)
            block = np.tile(strip[np.newaxis, :, :], (height, 1, 1))
            scene_blocks.append(block)

        if len(scene_blocks) == 0:
            return np.zeros((height, 1, 3))

        return np.concatenate(scene_blocks, axis=1)

    def _build_gray_scene_blocks(self, values, widths, height=20):
        values = np.asarray(values, dtype=float)
        blocks = []

        for v, w in zip(values, widths):
            block = np.full((height, w), v, dtype=float)
            blocks.append(block)

        if len(blocks) == 0:
            return np.zeros((height, 1), dtype=float)

        return np.concatenate(blocks, axis=1)

    def plot_all(self, intensity, happiness, brightness, saturation, dom_colors, scene_color_strips, scene_ids=None):
        intensity = np.asarray(intensity, dtype=float)
        happiness = np.asarray(happiness, dtype=float)
        brightness = np.asarray(brightness, dtype=float)
        saturation = np.asarray(saturation, dtype=float)
        dom_colors = np.asarray(dom_colors, dtype=float)

        n = min(len(intensity), len(happiness), len(brightness), len(saturation), len(dom_colors), len(scene_color_strips))

        intensity = intensity[:n]
        happiness = happiness[:n]
        brightness = brightness[:n]
        saturation = saturation[:n]
        dom_colors = dom_colors[:n]
        scene_color_strips = scene_color_strips[:n]

        if scene_ids is None:
            scene_ids = list(range(1, n + 1))
        else:
            scene_ids = scene_ids[:n]

        widths = [len(strip) if np.asarray(strip).ndim > 1 else 1 for strip in scene_color_strips]

        color_barcode = self._build_scene_barcode(scene_color_strips, height=20)
        brightness_barcode = self._build_gray_scene_blocks(brightness, widths, height=20)
        saturation_barcode = self._build_gray_scene_blocks(saturation, widths, height=20)

        intensity_j = self._jitter(intensity, self.jitter_emotion)
        happiness_j = self._jitter(happiness, self.jitter_emotion)
        brightness_j = self._jitter(brightness, self.jitter_visual)
        saturation_j = self._jitter(saturation, self.jitter_visual)

        fig, axes = plt.subplots(
            2, 3,
            figsize=(18, 9),
            gridspec_kw={"height_ratios": [3, 1]}
        )

        # TOP LEFT
        for i in range(n):
            if np.isnan(intensity[i]) or np.isnan(happiness[i]):
                continue

            color = np.clip(dom_colors[i] / 255.0, 0, 1)

            axes[0, 0].scatter(
                intensity_j[i], happiness_j[i],
                color=[color], s=220,
                edgecolors="black", linewidths=0.8, alpha=0.9
            )
            axes[0, 0].text(
                intensity_j[i] + 0.02,
                happiness_j[i] + 0.02,
                str(scene_ids[i]),
                fontsize=8
            )

        axes[0, 0].set_xlabel("Intensity")
        axes[0, 0].set_ylabel("Happiness")
        axes[0, 0].set_title("Emotion vs Scene Color")
        axes[0, 0].axhline(0, color="gray")
        axes[0, 0].axvline(0, color="gray")
        axes[0, 0].set_xlim(-1.1, 1.1)
        axes[0, 0].set_ylim(-1.1, 1.1)
        axes[0, 0].grid(True, alpha=0.4)

        # TOP MIDDLE
        for i in range(n):
            if np.isnan(brightness[i]) or np.isnan(happiness[i]):
                continue

            color = np.clip(dom_colors[i] / 255.0, 0, 1)

            axes[0, 1].scatter(
                brightness_j[i], happiness_j[i],
                color=[color], s=220,
                edgecolors="black", linewidths=0.8, alpha=0.9
            )
            axes[0, 1].text(
                brightness_j[i] + 0.005,
                happiness_j[i] + 0.02,
                str(scene_ids[i]),
                fontsize=8
            )

        axes[0, 1].set_xlabel("Brightness")
        axes[0, 1].set_ylabel("Happiness")
        axes[0, 1].set_title("Brightness vs Happiness")
        axes[0, 1].set_xlim(0, 1)
        axes[0, 1].set_ylim(-1.1, 1.1)
        axes[0, 1].grid(True, alpha=0.4)

        # TOP RIGHT
        for i in range(n):
            if np.isnan(saturation[i]) or np.isnan(intensity[i]):
                continue

            color = np.clip(dom_colors[i] / 255.0, 0, 1)

            axes[0, 2].scatter(
                saturation_j[i], intensity_j[i],
                color=[color], s=220,
                edgecolors="black", linewidths=0.8, alpha=0.9
            )
            axes[0, 2].text(
                saturation_j[i] + 0.005,
                intensity_j[i] + 0.02,
                str(scene_ids[i]),
                fontsize=8
            )

        axes[0, 2].set_xlabel("Saturation")
        axes[0, 2].set_ylabel("Intensity")
        axes[0, 2].set_title("Saturation vs Intensity")
        axes[0, 2].set_xlim(0, 1)
        axes[0, 2].set_ylim(-1.1, 1.1)
        axes[0, 2].grid(True, alpha=0.4)

        # BOTTOM LEFT
        axes[1, 0].imshow(color_barcode, aspect="auto")
        axes[1, 0].set_title("Scene-Aligned Color Barcode")
        axes[1, 0].set_yticks([])
        axes[1, 0].set_xlabel("Scene")

        centers = []
        pos = 0
        for w in widths:
            centers.append(pos + w / 2 - 0.5)
            pos += w

        axes[1, 0].set_xticks(centers)
        axes[1, 0].set_xticklabels(scene_ids, fontsize=8)

        pos = 0
        for w in widths[:-1]:
            pos += w
            axes[1, 0].axvline(pos - 0.5, color="white", linewidth=1)

        # BOTTOM MIDDLE
        axes[1, 1].imshow(brightness_barcode, cmap="gray", aspect="auto", vmin=0, vmax=1)
        axes[1, 1].set_title("Brightness Timeline")
        axes[1, 1].set_yticks([])
        axes[1, 1].set_xticks(centers)
        axes[1, 1].set_xticklabels(scene_ids, fontsize=8)
        axes[1, 1].set_xlabel("Scene")

        pos = 0
        for w in widths[:-1]:
            pos += w
            axes[1, 1].axvline(pos - 0.5, color="white", linewidth=1)

        # BOTTOM RIGHT
        axes[1, 2].imshow(saturation_barcode, cmap="gray", aspect="auto", vmin=0, vmax=1)
        axes[1, 2].set_title("Saturation Timeline")
        axes[1, 2].set_yticks([])
        axes[1, 2].set_xticks(centers)
        axes[1, 2].set_xticklabels(scene_ids, fontsize=8)
        axes[1, 2].set_xlabel("Scene")

        pos = 0
        for w in widths[:-1]:
            pos += w
            axes[1, 2].axvline(pos - 0.5, color="white", linewidth=1)

        plt.suptitle(self.title, fontsize=16)
        plt.tight_layout()
        plt.show()
        