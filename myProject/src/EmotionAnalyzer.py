import pandas as pd
import numpy as np

class EmotionAnalyzer:
    def __init__(self, csv_path):
        self.csv_path = csv_path

    def analyze(self):
        A = pd.read_csv(self.csv_path)

        scene_names = [
            "first","second","third","fourth","fifth",
            "sixth","seventh","eighth","ninth","tenth",
            "eleventh","twelfth","thirteenth","fourteenth","fifteenth",
            "sixteenth","seventeenth","eighteenth","nineteenth","twentieth",
            "twenty-first","twenty-second","twenty-third"
        ]

        n_scenes = (len(A.columns) - 1) // 2

        intensity_idx = np.zeros(n_scenes)
        feeling_idx = np.zeros(n_scenes)

        for i in range(n_scenes):
            intensity_col = f"is the {scene_names[i]} scene intense or calm?"
            feeling_col = f"is the {scene_names[i]} scene happy or sad?"

            intensity_vals = pd.to_numeric(A[intensity_col], errors="coerce").to_numpy()
            feeling_vals = pd.to_numeric(A[feeling_col], errors="coerce").to_numpy()

            intensity_idx[i] = np.nanmean((intensity_vals - 5) / 5)
            feeling_idx[i] = np.nanmean((feeling_vals - 5) / 5)

        return intensity_idx, feeling_idx