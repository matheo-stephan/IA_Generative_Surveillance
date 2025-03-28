import os
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import cv2
import json
import numpy as np
import faiss
import datetime

class Florence2Analysis:
    def __init__(self, analysis_folder):
        self.analysis_folder = analysis_folder
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Florence-2-large", 
            torch_dtype=self.torch_dtype, 
            trust_remote_code=True
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)
        self.text_embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')  # Encodeur pour les embeddings

    def analyze_video(self, video_path, target_fps=1):
        """Analyse une vidéo en extrayant des frames et en utilisant Florence-2 pour les analyser."""
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_folder = os.path.join(self.analysis_folder, f"{video_name}_Florence-2")
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

                # Analyse de la frame avec Florence-2
                image = Image.open(frame_path)
                prompt = "<OD>"
                inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device, self.torch_dtype)
                generated_ids = self.model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=4096,
                    num_beams=3,
                    do_sample=False
                )
                generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
                parsed_answer = self.processor.post_process_generation(
                    generated_text, task="<OD>", image_size=(image.width, image.height)
                )

                # Sauvegarder les résultats
                detection_path = os.path.join(analysed_frames_dir, f"detection_{frame_count:06d}.json")
                with open(detection_path, "w") as f:
                    json.dump(parsed_answer, f, indent=4)
                detections.append(parsed_answer)
                print(f"✅ Résultats sauvegardés : {detection_path}")

                # Générer un embedding pour l'annotation textuelle
                embedding = self.text_embedder.encode(generated_text, convert_to_tensor=False)
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

        # Sauvegarder les embeddings dans un fichier .faiss
        self.save_embeddings_to_faiss(embeddings_list, video_folder)

        # Générer le fichier info.json
        self.generate_info_file(video_name, video_folder, duration, total_frames, fps, start_time, end_time)

        return video_folder

    def save_embeddings_to_faiss(self, embeddings_list, video_folder):
        """Enregistre les embeddings dans un fichier .faiss."""
        if not embeddings_list:
            print("❌ Aucun embedding généré. Les résultats ne peuvent pas être enregistrés.")
            return

        embeddings_np = np.array(embeddings_list).astype('float32')
        index = faiss.IndexFlatL2(embeddings_np.shape[1])  # Index FAISS pour la recherche L2
        index.add(embeddings_np)

        faiss_path = os.path.join(video_folder, "vector_database.faiss")
        faiss.write_index(index, faiss_path)
        print(f"✅ Embeddings sauvegardés dans : {faiss_path}")

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
            "model": "Florence2",
            "video_name": video_name,
            "video_duration_seconds": round(duration, 2),
            "total_frames": total_frames,
            "fps": round(fps, 2)
        }

        # Sauvegarder les données dans un fichier JSON
        with open(info_file_path, "w") as f:
            json.dump(info_data, f, indent=4)
        print(f"✅ Fichier d'informations généré : {info_file_path}")
