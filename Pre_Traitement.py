import cv2
import os
import datetime
from ultralytics import YOLO
from sentence_transformers import SentenceTransformer
import shutil
import pickle

FPS_TARGET = 24  # FPS cible
TARGET_SIZE = (640, 360 )  # Nouvelle taille des frames

# Charger le modèle YOLO
model = YOLO("yolo11n.pt")

# Clear cache and re-download the SentenceTransformer model
cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)

# Charger le modèle d'embeddings
embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def create_unique_folder(base_dir, prefix="frames"):
    timestamp = datetime.datetime.now().strftime("%d%m_%H%M%S")
    folder_path = os.path.join(base_dir, f"{prefix}_{timestamp}")
    os.makedirs(folder_path, exist_ok=True)
    return folder_path

def resize_frame(frame, size):
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)

def extract_and_resize_frames(video_path, output_dir, target_fps, target_size, start_frame=0, custom_output_dir=None):
    if not os.path.exists(video_path):
        print(f"Erreur: Le fichier vidéo '{video_path}' n'existe pas.")
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Erreur: Impossible d'ouvrir la vidéo '{video_path}'")
        return None

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames_original = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames_target = int((total_frames_original / original_fps) * target_fps)

    print(f"Total frames original: {total_frames_original}")
    print(f"Total frames cible: {total_frames_target}")

    # Vérification et ajustement de start_frame
    if start_frame > total_frames_target:
        print(f"Erreur: La frame de départ ({start_frame}) est supérieure au nombre total de frames cibles ({total_frames_target}).")
        start_frame = total_frames_target  # Ajuster start_frame pour ne pas dépasser total_frames_target
        print(f"start_frame ajusté à {start_frame}")

    # Produit en croix pour ajuster la frame de départ
    start_time = (start_frame * total_frames_original) / total_frames_target
    adjusted_start_frame = round(start_time)
    print(f"Start frame ajusté: {adjusted_start_frame}")

    # Utilisation du dossier personnalisé ou création d'un nouveau dossier
    frames_dir = custom_output_dir if custom_output_dir else create_unique_folder(output_dir, "frames_resized")
    os.makedirs(frames_dir, exist_ok=True)

    # Extraire et redimensionner les frames à partir de la frame ajustée
    extracted_count = start_frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, adjusted_start_frame)  # Placer le lecteur vidéo à la frame ajustée

    while True:
        success, frame = cap.read()
        if not success:
            break

        resized_frame = resize_frame(frame, target_size)
        frame_filename = os.path.join(frames_dir, f"frame_{extracted_count:06d}.png")
        cv2.imwrite(frame_filename, resized_frame)
        extracted_count += 1

    cap.release()
    print(f"Extraction et redimensionnement terminés. {extracted_count - adjusted_start_frame} images enregistrées dans '{frames_dir}'")
    return frames_dir


def create_video_from_frames(frames_dir, output_path, fps):
    frames = sorted(f for f in os.listdir(frames_dir) if f.endswith(".png"))
    if not frames:
        print(f"Erreur: Aucune frame trouvée dans '{frames_dir}'")
        return
    
    first_frame = cv2.imread(os.path.join(frames_dir, frames[0]))
    height, width, _ = first_frame.shape
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame_filename in frames:
        frame = cv2.imread(os.path.join(frames_dir, frame_filename))
        out.write(frame)
    
    out.release()
    print(f"Vidéo générée à '{output_path}'")

def analyze_frames(frames_dir):
    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    image_files = [f for f in os.listdir(frames_dir) if f.lower().endswith(image_extensions)]
    if not image_files:
        print("Aucune image trouvée pour l'analyse.")
        return

    analysed_folder = os.path.join(frames_dir, "analysed_frames")
    os.makedirs(analysed_folder, exist_ok=True)
    
    detections_data = []
    embeddings_list = []
    
    for image_name in image_files:
        image_path = os.path.join(frames_dir, image_name)
        print(f"📷 Analyse de : {image_name}")
        results = model(image_path)

        if results:  # Vérifie si des résultats ont été trouvés
            analysed_image_path = os.path.join(analysed_folder, image_name)
            results[0].save(filename=analysed_image_path)
            print(f"✅ Image annotée sauvegardée : {analysed_image_path}")
        
            for result in results:
                boxes = result.boxes.xywh.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                class_names = [result.names[int(cls)] for cls in classes]
                
                for i in range(len(boxes)):
                    detection = {
                        "image": image_name,
                        "class": class_names[i],
                        "x": float(boxes[i][0]),
                        "y": float(boxes[i][1]),
                        "width": float(boxes[i][2]),
                        "height": float(boxes[i][3]),
                        "confidence": float(confs[i]),
                    }
                    detections_data.append(detection)
                    text_data = f"{detection['class']} {detection['confidence']}"
                    embedding = embedder.encode(text_data)
                    embeddings_list.append(embedding)
                    print(f"🔹 Objet détecté : {detection['class']} | Confiance : {detection['confidence']:.2f}")
            # Sauvegarde des embeddings
            with open("embeddings.pkl", "wb") as f:
                pickle.dump((embeddings_list), f)        
        else:
            print(f"🚨 Aucune détection sur {image_name}")
    return embeddings_list

# ============================
#           EXÉCUTION
# ============================

video_path = "C:/Users/mstep/OneDrive/Bureau/Projet_IA/Pre-traitement/Vidéo/Video_2.mp4"
frames_base_dir = "C:/Users/mstep/OneDrive/Bureau/Projet_IA/Pre-traitement/Frames"
timestamp = datetime.datetime.now().strftime("%d%m_%H%M%S")
output_video_path = os.path.join("C:/Users/mstep/OneDrive/Bureau/Projet_IA/Pre-traitement", f"video_output_{timestamp}.mp4")

custom_folder = ""
frames_dir = extract_and_resize_frames(video_path, frames_base_dir, FPS_TARGET, TARGET_SIZE, start_frame=0, custom_output_dir=custom_folder)

if frames_dir:
    analyze_frames(frames_dir)

    # Correction ici : on s'assure que le dossier d'analyse est bien dans custom_folder
    analysed_folder = os.path.join(frames_dir, "analysed_frames")
    os.makedirs(analysed_folder, exist_ok=True)

    create_video_from_frames(analysed_folder, output_video_path, FPS_TARGET)