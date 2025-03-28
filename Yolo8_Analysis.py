import os
import json
import faiss
import cv2
import shutil
import datetime
import concurrent.futures
import numpy as np
from ultralytics import YOLO
from sentence_transformers import SentenceTransformer

class Yolo8VideoAnalysis:
    def __init__(self, analysis_folder):
        self.analysis_folder = analysis_folder
        self.model = YOLO("yolov8n.pt")  # Modèle YOLOv8 avec segmentation
        self.embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def create_video_folders(self, video_folder):
        """Crée les dossiers nécessaires pour stocker les frames extraites et analysées."""
        extracted_frames_dir = os.path.join(video_folder, "Extracted_Frames")
        analysed_frames_dir = os.path.join(video_folder, "Analysed_Frames")
        os.makedirs(extracted_frames_dir, exist_ok=True)
        os.makedirs(analysed_frames_dir, exist_ok=True)
        return extracted_frames_dir, analysed_frames_dir

    def extract_and_process_frames(self, video_path, target_fps, model_name, num_threads=4, existing_folder=None):
        """Extrait les frames d'une vidéo et les analyse avec YOLOv8."""
        # Extraire uniquement le nom du fichier vidéo (sans chemin complet ni extension)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        start_time = datetime.datetime.now()

        # Utiliser le dossier existant ou en créer un nouveau
        if existing_folder:
            # Vérifier si existing_folder est déjà le bon dossier
            if os.path.basename(existing_folder) == f"{video_name}_{model_name}":
                video_folder = existing_folder
            else:
                # Créer un sous-dossier nommé video_name_modelname dans existing_folder
                video_folder = os.path.join(self.analysis_folder, f"{video_name}_{model_name}")
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
        extracted_dir, analysed_dir = self.create_video_folders(video_folder)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Impossible d'ouvrir la vidéo : {video_path}")
            return None

        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / original_fps
        cap.release()

        print(f"🎥 Vidéo : {video_name}, FPS : {original_fps}, Total Frames : {total_frames}, Durée : {duration:.2f}s")

        # Calculer les timestamps des frames à extraire
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

        return video_folder
    
    def process_frame(self, video_path, timestamp, frame_number, video_name, extracted_dir, analysed_dir, detections_data, embeddings_list):
        """Traite une frame spécifique : extraction, analyse et annotation."""
        # Ouvrir la vidéo et se positionner au timestamp
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)  # Positionner à la milliseconde
        success, frame = cap.read()
        cap.release()

        if success:
            # Sauvegarder la frame extraite
            frame_filename = os.path.join(extracted_dir, f"{video_name}_frame{frame_number:06d}.jpg")
            cv2.imwrite(frame_filename, frame)
            print(f"✅ Frame extraite et sauvegardée : {frame_filename}")

            # Analyse avec YOLOv8
            results = self.model(frame_filename)
            if results:
                # Sauvegarder l'image annotée
                annotated_filename = os.path.join(analysed_dir, f"{video_name}_Analysed_frame{frame_number:06d}.jpg")
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

    def create_video_from_frames(self, analysed_dir, video_name, video_folder, fps):
        """Génère une vidéo annotée à partir des frames analysées."""
        output_video_path = os.path.join(video_folder, f"{video_name}_Analysed.mp4")
        
        # Filtrer les fichiers avec l'extension correcte et les trier
        frames = sorted(
            [f for f in os.listdir(analysed_dir) if f.lower().endswith((".jpg", ".png"))]
        )

        if not frames:
            print(f"Erreur: Aucune frame annotée trouvée dans '{analysed_dir}'")
            return

        # Charger la première frame pour obtenir les dimensions de la vidéo
        first_frame_path = os.path.join(analysed_dir, frames[0])
        first_frame = cv2.imread(first_frame_path)
        if first_frame is None:
            print(f"Erreur: Impossible de lire la première frame : {first_frame_path}")
            return

        height, width, _ = first_frame.shape
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        # Ajouter chaque frame à la vidéo
        for frame_filename in frames:
            frame_path = os.path.join(analysed_dir, frame_filename)
            frame = cv2.imread(frame_path)
            if frame is None:
                print(f"Erreur: Impossible de lire la frame : {frame_path}")
                continue
            out.write(frame)

        out.release()
        print(f"✅ Vidéo générée à '{output_video_path}'")

    def save_analysis_results(self, detections_data, embeddings_list, video_folder):
        """Enregistre les résultats d'analyse dans des fichiers JSON et un index FAISS."""
        detections_path = os.path.join(video_folder, "detections.json")
        faiss_index_path = os.path.join(video_folder, "embeddings.faiss")

        # Sauvegarder les détections dans un fichier JSON
        with open(detections_path, "w") as f:
            json.dump(detections_data, f, indent=4)

        # Convertir les embeddings en un tableau numpy
        embeddings_np = np.array(embeddings_list).astype('float32')

        # Créer un index FAISS basé sur la distance L2 (euclidienne)
        dimension = embeddings_np.shape[1]  # Taille des vecteurs d'embedding
        index = faiss.IndexFlatL2(dimension)  # Index FAISS pour la recherche L2
        index.add(embeddings_np)  # Ajouter les embeddings à l'index

        # Sauvegarder l'index FAISS sur disque
        faiss.write_index(index, faiss_index_path)

        print(f"✅ Résultats enregistrés dans : {detections_path}")
        print(f"✅ Index FAISS enregistré dans : {faiss_index_path}")

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
            "model": "Yolov8",
            "video_name": video_name,
            "video_duration_seconds": round(duration, 2),
            "total_frames": total_frames,
            "fps": round(fps, 2)
        }
        with open(info_file_path, "w") as f:
            json.dump(info_data, f, indent=4)
        print(f"✅ Fichier d'informations généré : {info_file_path}")