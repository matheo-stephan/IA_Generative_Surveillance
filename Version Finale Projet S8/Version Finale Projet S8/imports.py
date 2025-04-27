import os
import shutil
import zipfile
import json
import time, datetime
import cv2
import faiss
import torch
import numpy as np
from AI_Models.threads import UploadThread, ClipAnalysisThread 
from AI_Models.Clip_Analysis import EmbeddingComparator
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QTreeWidget, QTreeWidgetItem, QPushButton, QStackedWidget, 
    QSplitter, QTextEdit, QLabel, QComboBox, QHBoxLayout, QSizePolicy, QFileDialog, 
    QTreeWidgetItem, QMessageBox, QSlider, QInputDialog, QLineEdit, QProgressBar, 
    QTabWidget, QFormLayout, QCheckBox, QGroupBox, QTextBrowser
)
from PyQt6.QtMultimedia import QMediaPlayer
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtCore import QUrl, Qt, QTimer, QThread, pyqtSignal, QSize, QRect, QEvent
from PyQt6.QtGui import QIcon, QPixmap, QMovie
