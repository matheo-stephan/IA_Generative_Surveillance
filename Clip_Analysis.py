import os
import torch
import clip
import cv2
import json
import datetime
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

class ClipAnalysis:
    def __init__(self, analysis_folder):
        self.analysis_folder = analysis_folder
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.text_embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def analyze_video(self, video_path, target_fps=1):
        """Analyse une vidéo en extrayant des frames et en utilisant CLIP pour les analyser."""
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_folder = os.path.join(self.analysis_folder, f"{video_name}_Clip")
        os.makedirs(video_folder, exist_ok=True)

        extracted_frames_dir = os.path.join(video_folder, "Extracted_Frames")
        analysed_frames_dir = os.path.join(video_folder, "Analysed_Frames")
        os.makedirs(extracted_frames_dir, exist_ok=True)
        os.makedirs(analysed_frames_dir, exist_ok=True)

        # Capturer le timestamp de début
        start_time = datetime.datetime.now()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Impossible d'ouvrir la vidéo : {video_path}")
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        frame_interval = int(fps / target_fps)
        frame_count = 0
        detections = []
        embeddings_list = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                frame_path = os.path.join(extracted_frames_dir, f"frame_{frame_count:06d}.jpg")
                cv2.imwrite(frame_path, frame)
                print(f"✅ Frame extraite : {frame_path}")

                # Analyse de la frame avec CLIP
                image = self.preprocess(Image.open(frame_path)).unsqueeze(0).to(self.device)
                text_prompts = ["a person", "a car", "a tree", "a dog", "a cat"]
                text = clip.tokenize(text_prompts).to(self.device)

                with torch.no_grad():
                    image_features = self.model.encode_image(image)
                    text_features = self.model.encode_text(text)
                    logits_per_image, _ = self.model(image, text)
                    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

                # Sauvegarder les résultats
                detection = {
                    "frame": frame_path,
                    "detections": [
                        {"label": text_prompts[i], "probability": float(probs[0][i])}
                        for i in range(len(text_prompts))
                    ]
                }
                detections.append(detection)
                detection_path = os.path.join(analysed_frames_dir, f"detection_{frame_count:06d}.json")
                with open(detection_path, "w") as f:
                    json.dump(detection, f, indent=4)
                print(f"✅ Résultats sauvegardés : {detection_path}")

                # Générer un embedding pour les annotations textuelles
                embedding = self.text_embedder.encode(json.dumps(detection), convert_to_tensor=False)
                embeddings_list.append(embedding)

            frame_count += 1

        cap.release()

        # Capturer le timestamp de fin
        end_time = datetime.datetime.now()

        # Sauvegarder toutes les détections dans un fichier global
        detections_path = os.path.join(video_folder, "detections.json")
        with open(detections_path, "w") as f:
            json.dump(detections, f, indent=4)
        print(f"✅ Analyse terminée. Résultats sauvegardés dans : {detections_path}")

        # Sauvegarder les embeddings dans un fichier .npy
        embeddings_np = np.array(embeddings_list).astype('float32')
        embeddings_path = os.path.join(video_folder, "embeddings.npy")
        np.save(embeddings_path, embeddings_np)
        print(f"✅ Embeddings sauvegardés dans : {embeddings_path}")

        # Générer le fichier info.json
        self.generate_info_file(video_name, video_folder, duration, total_frames, fps, start_time, end_time)

        return video_folder

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
            "model": "CLIP",
            "video_name": video_name,
            "video_duration_seconds": round(duration, 2),
            "total_frames": total_frames,
            "fps": round(fps, 2)
        }

        # Sauvegarder les données dans un fichier JSON
        with open(info_file_path, "w") as f:
            json.dump(info_data, f, indent=4)
        print(f"✅ Fichier d'informations généré : {info_file_path}")