
from EmotionAnalyzer import EmotionAnalyzer
from ColorAnalyzer import ColorAnalyzer
from Plotter import Plotter
import numpy as np


csv_path = "src/test/ToyStory2Responses.csv"
video_path = "src/test/toyStory2Use.mp4"

scenes = [
      (0,10),
(10,14),
(14,17),
(18,23),
(23,25),
(25,30),
(30,33),
(33,36),
(36,52),
(52,56),
(56,60),
(60,61),
(61,70),
(70,72),
(73,77),
(77,80),
(81,93),
(94,98),
(99,103)

]


#Analyzes Emotion

emotion_analyzer = EmotionAnalyzer(csv_path)
intensity_idx, feeling_idx = emotion_analyzer.analyze()

#Analyzes Color

color_analyzer = ColorAnalyzer(video_path, scenes)
brightness, saturation, dom_colors, scene_color_strips = color_analyzer.analyze()

#Debug Prints

print("intensity_idx =", intensity_idx)
print("feeling_idx =", feeling_idx)
print("brightness =", brightness)
print("saturation =", saturation)
print("dom_colors =", dom_colors)

#lenth Checker

if len(scenes) != len(intensity_idx):
    print(f"Warning: scenes in video = {len(scenes)}, scenes in survey = {len(intensity_idx)}")

n = min(len(intensity_idx), len(feeling_idx), len(brightness), len(saturation), len(dom_colors))
print("Using n =", n)

intensity_idx = intensity_idx[:n]
feeling_idx = feeling_idx[:n]
brightness = brightness[:n]
saturation = saturation[:n]
dom_colors = dom_colors[:n]
scene_color_strips = scene_color_strips[:n]

scene_ids = list(range(1, n + 1))

#Correlations

corr_brightness_happiness = np.corrcoef(brightness, feeling_idx)[0, 1]
corr_saturation_intensity = np.corrcoef(saturation, intensity_idx)[0, 1]
corr_intensity_happiness = np.corrcoef(intensity_idx, feeling_idx)[0, 1]

print("\n--- CORRELATIONS ---")
print("Brightness vs Happiness:", corr_brightness_happiness)
print("Saturation vs Intensity:", corr_saturation_intensity)
print("Intensity vs Happiness:", corr_intensity_happiness)

# Plot

plotter = Plotter(
    title="Toy Story 5, Color and Emotion Correlation",
    jitter_emotion=0.03,
    jitter_visual=0.015,
    random_state=42
)

plotter.plot_all(
    intensity=intensity_idx,
    happiness=feeling_idx,
    brightness=brightness,
    saturation=saturation,
    dom_colors=dom_colors,
    scene_color_strips=scene_color_strips,
    scene_ids=scene_ids
)

