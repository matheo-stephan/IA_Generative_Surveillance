from PyQt6.QtCore import QThread, pyqtSignal
from Yolo11_Analysis import Yolo11VideoAnalysis
from Yolo8_Analysis import Yolo8VideoAnalysis
from Florence2_Analysis import Florence2Analysis
from Clip_Analysis import ClipAnalysis
import os

class Yolo11AnalysisThread(QThread):
    analysis_finished = pyqtSignal(str)

    def __init__(self, video_path, model_name, target_folder, target_fps=15, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.model_name = model_name
        self.target_folder = target_folder
        self.target_fps = target_fps

    def run(self):
        """Exécute l'analyse dans un thread séparé."""
        video_name = os.path.splitext(os.path.basename(self.video_path))[0]

        # Initialiser l'analyse
        analysis = Yolo11VideoAnalysis(self.target_folder)
        video_folder, video_name = analysis.extract_and_process_frames(
            self.video_path, target_fps=self.target_fps, model_name=self.model_name, existing_folder=self.target_folder
        )
        if video_folder:
            analysis.create_video_from_frames(
                os.path.join(video_folder, "Analysed_Frames"), video_name, video_folder, fps=self.target_fps
            )
            self.analysis_finished.emit(os.path.join(video_folder, f"{video_name}_Analysed.mp4"))


class Yolo8AnalysisThread(QThread):
    analysis_finished = pyqtSignal(str)

    def __init__(self, analysis_folder, video_path, target_fps=15, parent=None):
        super().__init__(parent)
        self.analysis_folder = analysis_folder
        self.video_path = video_path
        self.target_fps = target_fps

    def run(self):
            """Exécute l'analyse avec YOLOv8 dans un thread séparé."""
            try:
                print(f"📂 Dossier d'analyse : {self.analysis_folder}")
                print(f"📄 Chemin de la vidéo : {self.video_path}")
                analyzer = Yolo8VideoAnalysis(self.analysis_folder)

                # Lancer l'analyse
                result_folder = analyzer.extract_and_process_frames(self.video_path, self.target_fps, "Yolo8")
                if result_folder:
                    # Générer la vidéo annotée
                    analysed_dir = os.path.join(result_folder, "Analysed_Frames")
                    video_name = os.path.splitext(os.path.basename(self.video_path))[0]
                    analyzer.create_video_from_frames(analysed_dir, video_name, result_folder, fps=self.target_fps)

                    # Émettre le signal avec le chemin de la vidéo générée
                    self.analysis_finished.emit(os.path.join(result_folder, f"{video_name}_Analysed.mp4"))
            except Exception as e:
                print(f"❌ Erreur dans le thread Yolo8AnalysisThread : {e}")
            

class Florence2AnalysisThread(QThread):
    analysis_finished = pyqtSignal(str)

    def __init__(self, analysis_folder, video_path, target_fps=1):
        super().__init__()
        self.analysis_folder = analysis_folder
        self.video_path = video_path
        self.target_fps = target_fps

    def run(self):
        """Exécute l'analyse dans un thread séparé."""
        analyzer = Florence2Analysis(self.analysis_folder)
        result_folder = analyzer.analyze_video(self.video_path, self.target_fps)
        if result_folder:
            self.analysis_finished.emit(result_folder)


class ClipAnalysisThread(QThread):
    analysis_finished = pyqtSignal(str)

    def __init__(self, analysis_folder, video_path, target_fps=1, parent=None):
        super().__init__(parent)
        self.analysis_folder = analysis_folder
        self.video_path = video_path
        self.target_fps = target_fps

    def run(self):
        """Exécute l'analyse avec le modèle CLIP dans un thread séparé."""
        analyzer = ClipAnalysis(self.analysis_folder)
        result_folder = analyzer.analyze_video(self.video_path, self.target_fps)
        if result_folder:
            self.analysis_finished.emit(result_folder)