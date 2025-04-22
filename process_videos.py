import os
import json
import cv2
from PIL import Image
import numpy as np

def extract_frames(video_path, fps=1):
    """Extrait des frames d'une vidéo à un intervalle donné (fps frames par seconde)."""
    print(f"Extraction des frames de la vidéo : {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    # Vérifier si la vidéo est ouverte correctement
    if not cap.isOpened():
        print(f"Erreur : Impossible d'ouvrir la vidéo {video_path}")
        return [], []

    video_fps = cap.get(cv2.CAP_PROP_FPS)  # Frames par seconde de la vidéo
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Vérifier si video_fps est valide
    if video_fps <= 0:
        print(f"Erreur : FPS invalide ({video_fps}) pour la vidéo {video_path}. Vérifiez si la vidéo est corrompue.")
        cap.release()
        return [], []

    # Vérifier si la vidéo a des frames
    if total_frames <= 0:
        print(f"Erreur : Aucune frame trouvée dans la vidéo {video_path}. Vérifiez si la vidéo est vide.")
        cap.release()
        return [], []

    duration = total_frames / video_fps  # Durée de la vidéo en secondes
    frame_interval = int(video_fps / fps)  # Intervalle pour extraire une frame par seconde

    # Créer un dossier temporaire pour les frames
    temp_dir = "temp_frames"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
        print(f"Dossier temporaire créé : {temp_dir}")

    frames = []
    frame_paths = []
    frame_idx = 0
    count = 0

    # Extraire le nom de fichier de la vidéo (sans chemin ni extension)
    video_name = os.path.basename(video_path).split('.')[0]

    while frame_idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = cap.read()
        if not success:
            print(f"Avertissement : Impossible de lire la frame {frame_idx} de la vidéo {video_path}")
            break
        # Convertir en image PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        # Sauvegarder la frame temporairement dans le dossier temp_frames
        frame_path = os.path.join(temp_dir, f"temp_frame_{video_name}_frame_{count}.png")
        pil_image.save(frame_path)
        print(f"Frame {count} extraite et sauvegardée à : {frame_path}")
        frames.append(pil_image)
        frame_paths.append(frame_path)
        frame_idx += frame_interval
        count += 1

    cap.release()
    return frames, frame_paths

def load_kinetics_annotations(annotation_file, video_dir, max_videos=100):
    """Charge les annotations de Kinetics-700 et extrait les frames pour chaque vidéo."""
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    
    all_frame_paths = []  # Liste de listes : chaque sous-liste contient les chemins des frames d'une vidéo
    labels = []
    video_ids = []
    count = 0
    
    for video_id, info in annotations.items():
        if count >= max_videos:
            break
        video_path = os.path.join(video_dir, info['path'])
        label = info['label']
        # Extraire toutes les frames (1 par seconde)
        _, frame_paths = extract_frames(video_path, fps=1)
        if not frame_paths:
            print(f"Aucune frame extraite pour {video_path}. Passage à la vidéo suivante.")
            continue
        all_frame_paths.append(frame_paths)
        labels.append(label)
        video_ids.append(video_id)
        count += 1
    
    return all_frame_paths, labels, video_ids