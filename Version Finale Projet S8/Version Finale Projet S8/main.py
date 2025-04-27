import os
import sys
from PyQt6.QtWidgets import QApplication
from ui import VideoAnalysisUI

BASE_DIR = os.getcwd()
BANK_DIR = os.path.join(BASE_DIR, "Bank")
os.makedirs(BANK_DIR, exist_ok=True)
FRAMES_DIR = os.path.join(BANK_DIR, "Frames")
os.makedirs(BANK_DIR, exist_ok=True)
ANALYSIS_DIR = os.path.join(BASE_DIR, "Analysis")
os.makedirs(ANALYSIS_DIR, exist_ok=True)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = VideoAnalysisUI(BASE_DIR, BANK_DIR, ANALYSIS_DIR)
    window.show()
    sys.exit(app.exec())