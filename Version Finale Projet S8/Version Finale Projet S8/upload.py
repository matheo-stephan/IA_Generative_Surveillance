from imports import *
from utils import load_hierarchy
from AI_Models.Clip_Analysis import ClipAnalysis
from AI_Models.threads import UploadThread
from AI_Models.faiss_instance import faiss_client

import gc

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Root folder
LOGO_PATH = os.path.join(BASE_DIR, "assets/logo.png")  # Logo path

def process_upload(video_path, bank_path, client, target_fps=2):
    """
    Gère l'upload d'une vidéo : extraction des frames, génération des embeddings, ajout à FAISS.
    """
    try:
        print(f"🔍 Vérification de l'existence de la vidéo : {video_path}")
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"❌ Video file not found: {video_path}")

        print(f"📂 Création du dossier des frames pour la vidéo : {video_path}")
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        frames_dir = os.path.join(bank_path, "Frames", f"{video_name}_frames")
        os.makedirs(frames_dir, exist_ok=True)

        print(f"🎥 Début de l'extraction des frames et de la génération des embeddings.")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"❌ Unable to open video: {video_path}")

        # Charger l'index FAISS existant (s'il existe)
        faiss_index_path = os.path.join(BASE_DIR, "data.faiss")
        if os.path.exists(faiss_index_path):
            print(f"🔄 Chargement de l'index FAISS existant depuis '{faiss_index_path}'...")
            client.load(faiss_index_path)
        else:
            print(f"⚠️ Aucun index FAISS existant trouvé. Un nouvel index sera créé.")

        comparator = ClipAnalysis(bank_path).comparator

        frame_count = 0
        ids = []
        embeddings = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Sauvegarder la frame sur le disque
            frame_file_name = f"frame_{frame_count:06d}.jpg"
            frame_path = os.path.join(frames_dir, frame_file_name)
            cv2.imwrite(frame_path, frame)
            print(f"✅ Frame sauvegardée : {frame_file_name}")

            # Générer l'embedding pour la frame
            embedding = comparator.encode_image(frame_path)
            embeddings.append(embedding[0])
            ids.append(frame_file_name)
            frame_count += 1

        # Ajouter les embeddings à la collection FAISS
        client.add_to_collection(video_name, ids, embeddings)
        client.save(faiss_index_path)  # Sauvegarder l'index FAISS mis à jour
        print(f"✅ Sauvegarde de l'index FAISS terminée.")

        cap.release()
        gc.collect()
        print(f"✅ Extraction et génération des embeddings terminées. Total frames : {frame_count}")
        return f"✅ Upload terminé pour la vidéo : {video_path}"

    except Exception as e:
        print(f"❌ Error in process_upload: {e}")
        raise RuntimeError(f"❌ Error in process_upload: {e}")

upload_threads = []  # Liste globale pour conserver les références aux threads

def add_item(bank_path, client, extractionLabel, extractionGifLabel, extractionMovie, bankNav):
    """
    Ouvre une fenêtre pour sélectionner une vidéo et l'ajouter à la Bank.
    """
    file_path, _ = QFileDialog.getOpenFileName(None, "Select a file to add to Bank", "", "Video Files (*.mp4 *.avi *.mov *.mkv)")
    if not file_path:
        return

    target_path = os.path.join(bank_path, os.path.basename(file_path))
    if os.path.exists(target_path):
        QMessageBox.information(None, "File Exists", f"The file '{os.path.basename(file_path)}' already exists in the Bank folder.")
        return

    try:
        shutil.copy(file_path, target_path)

        # Mettre à jour l'interface utilisateur
        extractionLabel.setText(f"Uploading...")
        extractionLabel.setVisible(True)
        extractionGifLabel.setVisible(True)
        extractionMovie.start()

        # Lancer le thread d'upload
        thread = UploadThread(target_path, bank_path, client, target_fps=2)
        upload_threads.append(thread)  # Conserver une référence au thread
        print(f"🔍 Thread ajouté : {thread}")

        thread.upload_finished.connect(lambda msg: QMessageBox.information(None, "Success", msg))
        thread.upload_finished.connect(lambda: hide_extraction_status(extractionLabel, extractionGifLabel, extractionMovie))
        thread.upload_finished.connect(lambda: upload_threads.remove(thread))  # Nettoyer après la fin
        thread.upload_finished.connect(lambda msg: on_extraction_finished(msg, bankNav, bank_path))
        thread.upload_failed.connect(lambda msg: QMessageBox.critical(None, "Error", msg))
        thread.upload_failed.connect(lambda: hide_extraction_status(extractionLabel, extractionGifLabel, extractionMovie))
        thread.upload_failed.connect(lambda: upload_threads.remove(thread))  # Nettoyer après la fin
        thread.start()
        
    except Exception as e:
        print(f"❌ Error copying file: {e}")
        QMessageBox.critical(None, "Error", f"Failed to copy file: {e}")

def hide_extraction_status(extractionLabel, extractionGifLabel, extractionMovie):
    """Hide the extraction message and GIF."""
    extractionLabel.setVisible(False)
    extractionGifLabel.setVisible(False)
    extractionMovie.stop()

def on_extraction_finished(msg, tree_widget, directory):
    """Handle the end of frame extraction."""
    QMessageBox.information(None, "Success", msg)
    load_hierarchy(tree_widget, directory)