import cv2
import os
from ultralytics import YOLO
from sentence_transformers import SentenceTransformer
from database import add_detection
import numpy as np
import random

# Charger les modèles
model = YOLO("yolo11n.pt")
embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def random_color():
    """Génère une couleur aléatoire."""
    return [random.randint(0, 255) for _ in range(3)]

def analyze_frames(frames_dir):
    """Analyse les frames avec YOLO et génère des embeddings."""
    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    image_files = [f for f in os.listdir(frames_dir) if f.lower().endswith(image_extensions)]
    counter = 0

    if not image_files:
        print("Aucune image trouvée pour l'analyse.")
        return []

    analysed_folder = os.path.join(frames_dir, "analysed_frames")
    os.makedirs(analysed_folder, exist_ok=True)
    
    embeddings_list = []
    
    for image_name in image_files:
        image_path = os.path.join(frames_dir, image_name)
        print(f"📷 Analyse de : {image_name}")
        results = model(image_path)

        if results:
            results[0].save(filename=os.path.join(analysed_folder, image_name))
            print(f"✅ Image annotée sauvegardée : {analysed_folder}/{image_name}")
        
            for result in results:
                boxes = result.boxes.xywh.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                class_names = [result.names[int(cls)] for cls in classes]
                
                for i in range(len(boxes)):
                    detection = {
                        "image": image_name,
                        "class": class_names[i],
                        "x": str(float(boxes[i][0])),
                        "y": str(float(boxes[i][1])),
                        "width": str(float(boxes[i][2])),
                        "height": str(float(boxes[i][3])),
                        "confidence": str(float(confs[i])),
                    }
                    counter += 1
                    text_data = f"{detection['class']} {detection['confidence']}"
                    embedding = embedder.encode(text_data).tolist()
                    embeddings_list.append(embedding)
                    add_detection(str(counter), embedding, detection)
                    print(f"🔹 Objet détecté : {detection['class']} | Confiance : {float(detection['confidence']):.2f}")
        else:
            print(f"🚨 Aucune détection sur {image_name}")
    return embeddings_list

def analyze_frames_filtered(frames_dir, target_objects):
    """Analyse et annote les frames pour des objets spécifiques."""
    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    image_files = [f for f in os.listdir(frames_dir) if f.lower().endswith(image_extensions)]
    if not image_files:
        print("Aucune image trouvée pour l'analyse.")
        return

    analysed_folder = os.path.join(frames_dir, "analysed_frames_filtered")
    os.makedirs(analysed_folder, exist_ok=True)
    class_colors = {}

    for image_name in image_files:
        image_path = os.path.join(frames_dir, image_name)
        print(f"📷 Analyse de : {image_name}")
        results = model(image_path)
        image = cv2.imread(image_path)

        if results:
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()

                for i, cls in enumerate(classes):
                    class_name = result.names[int(cls)]
                    if class_name not in class_colors:
                        class_colors[class_name] = random_color()
                    if class_name in target_objects:
                        color = class_colors[class_name]
                        x1, y1, x2, y2 = map(int, boxes[i])
                        confidence = confs[i]
                        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                        label = f"{class_name} {confidence:.2f}"
                        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.imwrite(os.path.join(analysed_folder, image_name), image)
            print(f"✅ Image annotée sauvegardée : {analysed_folder}/{image_name}")
        else:
            print(f"🚨 Aucune détection sur {image_name}")
    return analysed_folder