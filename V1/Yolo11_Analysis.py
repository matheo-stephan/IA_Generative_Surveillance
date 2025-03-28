import os
import json
import datetime
import cv2
import concurrent.futures
import numpy as np
import faiss
import shutil
from ultralytics import YOLO
from sentence_transformers import SentenceTransformer

class Yolo11VideoAnalysis:
    def __init__(self, analysis_folder):
        self.analysis_folder = analysis_folder
        self.model = YOLO("yolo11n.pt")
        self.embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def create_video_folders(self, video_folder):
        """Crée les dossiers nécessaires pour stocker les frames extraites et analysées."""
        # Ajouter le modèle au nom du dossier
        extracted_frames_dir = os.path.join( "Extracted_Frames")
        analysed_frames_dir = os.path.join( "Analysed_Frames")
        
        os.makedirs(extracted_frames_dir, exist_ok=True)
        os.makedirs(analysed_frames_dir, exist_ok=True)
        
        return extracted_frames_dir, analysed_frames_dir

    def extract_and_process_frames(self, video_path, target_fps, model_name, num_threads=4, existing_folder=None):
        """Extrait les frames d'une vidéo et les analyse avec YOLO."""
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        start_time = datetime.datetime.now()
        # Utiliser le dossier existant ou en créer un nouveau
        if existing_folder:
            # Vérifier si existing_folder est déjà le bon dossier
            if os.path.basename(existing_folder) == f"{video_name}_{model_name}":
                video_folder = existing_folder
            else:
                # Créer un sous-dossier nommé video_name_modelname dans existing_folder
                video_folder = os.path.join(existing_folder, f"{video_name}_{model_name}")
                os.makedirs(video_folder, exist_ok=True)
        else:
            # Créer le dossier directement dans self.analysis_folder
            video_folder = os.path.join(self.analysis_folder, f"{video_name}_{model_name}")
            os.makedirs(video_folder, exist_ok=True)

        # Copier la vidéo dans le dossier d'analyse
        video_copy_path = os.path.join(video_folder, f"{video_name}.mp4")
        if not os.path.exists(video_copy_path):
            shutil.copy(video_path, video_copy_path)
            print(f"✅ Vidéo copiée dans : {video_copy_path}")

        # Créer les sous-dossiers pour les frames
        extracted_dir = os.path.join(video_folder, "Extracted_Frames")
        analysed_dir = os.path.join(video_folder, "Analysed_Frames")
        os.makedirs(extracted_dir, exist_ok=True)
        os.makedirs(analysed_dir, exist_ok=True)

        print(f"📂 Dossiers créés : {extracted_dir}, {analysed_dir}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Impossible d'ouvrir la vidéo : {video_path}")
            return None

        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / original_fps
        cap.release()

        print(f"🎥 Vidéo : {video_name}, FPS : {original_fps}, Total Frames : {total_frames}, Durée : {duration:.2f}s")

        target_timestamps = [i / target_fps for i in range(int(duration * target_fps))]
        detections_data = []
        embeddings_list = []

        # Analyse des frames en parallèle
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(
                    self.process_frame,
                    video_path, timestamp, frame_number, video_name, extracted_dir, analysed_dir, detections_data, embeddings_list
                )
                for frame_number, timestamp in enumerate(target_timestamps)
            ]
            concurrent.futures.wait(futures)

        end_time = datetime.datetime.now()
        print(f"🔍 Analyse terminée pour toutes les frames.")

        # Enregistrer les résultats
        self.save_analysis_results(detections_data, embeddings_list, video_folder)

        # Générer le fichier info.json
        self.generate_info_file(video_name, video_folder, duration, total_frames, original_fps, start_time, end_time)

        return video_folder, video_name

    def process_frame(self, video_path, timestamp, frame_number, video_name, extracted_dir, analysed_dir, detections_data, embeddings_list):
        """Traite une frame spécifique : extraction, analyse et annotation."""
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        success, frame = cap.read()
        cap.release()

        if success:
            # Sauvegarder la frame extraite
            frame_filename = os.path.join(extracted_dir, f"{video_name}_frame{frame_number:06d}.png")
            cv2.imwrite(frame_filename, frame)
            print(f"✅ Frame extraite et sauvegardée : {frame_filename}")

            # Analyse avec YOLO
            results = self.model(frame_filename)
            if results:
                annotated_filename = os.path.join(analysed_dir, f"{video_name}_Analysed_frame{frame_number:06d}.png")
                results[0].save(filename=annotated_filename)
                print(f"✅ Frame annotée sauvegardée : {annotated_filename}")

                # Collecter les données des détections
                for result in results:
                    boxes = result.boxes.xywh.cpu().numpy()
                    confs = result.boxes.conf.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy()
                    class_names = [result.names[int(cls)] for cls in classes]

                    for i in range(len(boxes)):
                        detection = {
                            "image": frame_filename,
                            "class": class_names[i],
                            "x": float(boxes[i][0]),
                            "y": float(boxes[i][1]),
                            "width": float(boxes[i][2]),
                            "height": float(boxes[i][3]),
                            "confidence": float(confs[i]),
                        }
                        detections_data.append(detection)

                        # Générer des embeddings pour chaque détection
                        text_data = f"{frame_filename} {detection['class']} {detection['x']} {detection['y']} {detection['width']} {detection['height']} {detection['confidence']}"
                        embeddings_list.append(self.embedder.encode(text_data))
            else:
                print(f"❌ Aucune détection trouvée pour la frame : {frame_filename}")
        else:
            print(f"❌ Échec de l'extraction de la frame au timestamp {timestamp}s")

    def save_analysis_results(self, detections_data, embeddings_list, video_folder):
        """Enregistre les résultats d'analyse dans des fichiers JSON et FAISS."""
        detections_path = os.path.join(video_folder, "detections.json")
        vdb_path = os.path.join(video_folder, "vector_database.faiss")

        # Vérifier si embeddings_list est vide
        if not embeddings_list:
            print("❌ Aucun embedding généré. Les résultats ne peuvent pas être enregistrés.")
            return

        embeddings_np = np.array(embeddings_list).astype('float32')
        index = faiss.IndexFlatL2(embeddings_np.shape[1])
        index.add(embeddings_np)

        with open(detections_path, "w") as f:
            json.dump(detections_data, f, indent=4)

        faiss.write_index(index, vdb_path)
        print(f"✅ Résultats enregistrés dans '{detections_path}' et '{vdb_path}'")

    def create_video_from_frames(self, analysed_dir, video_name, video_folder, fps):
        """Génère une vidéo annotée à partir des frames analysées."""
        output_video_path = os.path.join(video_folder, f"{video_name}_Analysed.mp4")
        frames = sorted([f for f in os.listdir(analysed_dir) if f.endswith(".png")])

        if not frames:
            print(f"Erreur: Aucune frame annotée trouvée dans '{analysed_dir}'")
            return

        first_frame = cv2.imread(os.path.join(analysed_dir, frames[0]))
        height, width, _ = first_frame.shape
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        for frame_filename in frames:
            frame = cv2.imread(os.path.join(analysed_dir, frame_filename))
            out.write(frame)

        out.release()
        print(f"✅ Vidéo générée à '{output_video_path}'")

    def log_analysis(self, video_name, video_folder, log_message):
        """Ajoute un message de log dans le fichier log_videoname.txt."""
        log_file_path = os.path.join(video_folder, f"log_{video_name}.txt")
        with open(log_file_path, "a") as f:  # Mode "append" pour ajouter au fichier existant
            f.write(log_message + "\n")
        print(f"Log ajouté : {log_message}")

    def generate_info_file(self, video_name, video_folder, duration, total_frames, fps, start_time, end_time):
        """Génère un fichier JSON contenant les informations de la vidéo et de l'analyse."""
        # Calculer le temps total de traitement
        processing_time = (end_time - start_time).total_seconds()
        avg_time_per_frame = processing_time / total_frames if total_frames > 0 else 0

        # Générer les données pour le fichier JSON
        info_file_path = os.path.join(video_folder, f"info_{video_name}.json")
        info_data = {
            "date_start": start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "date_end": end_time.strftime("%Y-%m-%d %H:%M:%S"),
            "processing_time_seconds": round(processing_time, 2),
            "average_time_per_frame_seconds": round(avg_time_per_frame, 4),
            "model": "Yolov11",
            "video_name": video_name,
            "video_duration_seconds": round(duration, 2),
            "total_frames": total_frames,
            "fps": round(fps, 2)
        }
        with open(info_file_path, "w") as f:
            json.dump(info_data, f, indent=4)
        print(f"✅ Fichier d'informations généré : {info_file_path}")
