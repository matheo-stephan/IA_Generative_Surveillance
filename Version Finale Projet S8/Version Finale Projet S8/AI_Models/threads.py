from PyQt6.QtCore import QThread, pyqtSignal

class UploadThread(QThread):
    upload_finished = pyqtSignal(str)
    upload_failed = pyqtSignal(str)

    def __init__(self, video_path, bank_path, client, target_fps=2):
        super().__init__()
        self.video_path = video_path
        self.bank_path = bank_path
        self.client = client
        self.target_fps = target_fps

    def run(self):
        try:
            from upload import process_upload
            result_message = process_upload(self.video_path, self.bank_path, self.client, self.target_fps)
            self.upload_finished.emit(result_message)
        except Exception as e:
            self.upload_failed.emit(f"❌ Error during upload: {e}")


class ClipAnalysisThread(QThread):
    analysis_finished = pyqtSignal(str, str)
    analysis_failed = pyqtSignal(str)

    def __init__(self, analysis_folder, video_path, target_fps=2, prompt=""):
        super().__init__()
        self.analysis_folder = analysis_folder
        self.video_path = video_path
        self.target_fps = target_fps
        self.prompt = prompt
        self.model_name = "Clip"

    def run(self):
        try:
            from AI_Models.Clip_Analysis import analyse_prompt
            result_folder = analyse_prompt(self.analysis_folder, self.prompt)
            if result_folder:
                self.analysis_finished.emit(result_folder, self.model_name)
            else:
                self.analysis_failed.emit("❌ Analysis failed: No results generated.")
        except Exception as e:
            self.analysis_failed.emit(f"❌ Error during analysis: {e}")