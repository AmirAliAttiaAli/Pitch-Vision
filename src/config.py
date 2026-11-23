import os
from pathlib import Path
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GLOBAL_CONF = 0.10

PLAYER_CONF = 0.20
REFEREE_CONF = 0.40
BALL_CONF = 0.10
GOALKEEPER_CONF = 0.20

ROOT_DIR = Path(__file__).resolve().parent.parent

# Model weights
MODEL_BEST_PATH = ROOT_DIR / "Model" / "football_detector_best.pt"
MODEL_FALLBACK_PATH = ROOT_DIR / "Model" / "best.pt"
MODEL_PATH = MODEL_BEST_PATH if MODEL_BEST_PATH.exists() else MODEL_FALLBACK_PATH

# Test Data Directory
TEST_DATA_DIR = ROOT_DIR / "Test Data"
INPUT_VIDEO_PATH = TEST_DATA_DIR / "test_video.mp4"
OUTPUT_VIDEO_PATH = TEST_DATA_DIR / "possession_output.mp4"

# Sampling settings for team clustering
SAMPLE_MAX_FRAMES = 180
SAMPLE_FRAME_STEP = 3

# Ball possession settings
POSSESSION_DIST_THRESHOLD = 80.0  # Max pixel distance between ball and player to count as possession

# Class mapping (YOLO output)
CLASS_BALL = 0
CLASS_PLAYER = 1
CLASS_REFEREE = 2
CLASS_GOALKEEPER = 3

CLASS_NAMES = {
    0: "ball",
    1: "player",
    2: "referee",
    3: "goalkeeper"
}
