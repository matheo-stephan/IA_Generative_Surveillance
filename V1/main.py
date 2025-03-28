import os
import sys
from PyQt6.QtWidgets import QApplication
from ui import VideoAnalysisUI

# Définition des constantes globales
BASE_DIR = os.getcwd()
VIDEOS_DIR = os.path.join(BASE_DIR, "Analysis")
os.makedirs(VIDEOS_DIR, exist_ok=True)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = VideoAnalysisUI(BASE_DIR)
    window.show()
    sys.exit(app.exec())
