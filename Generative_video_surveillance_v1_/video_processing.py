import cv2
import os
import datetime
import shutil
from config import FPS_TARGET, TARGET_SIZE

def create_unique_folder(base_dir, prefix="frames"):
    """Crée un dossier unique avec timestamp."""
    timestamp = datetime.datetime.now().strftime("%d%m_%H%M%S")
    folder_path = os.path.join(base_dir, f"{prefix}_{timestamp}")
    os.makedirs(folder_path, exist_ok=True)
    return folder_path

def resize_frame(frame, size):
    """Redimensionne une frame."""
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)

def extract_and_resize_frames(video_path, output_dir, target_fps, target_size, start_frame=0, custom_output_dir=None):
    """Extrait et redimensionne les frames d'une vidéo."""
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

    if start_frame > total_frames_target:
        start_frame = total_frames_target
        print(f"start_frame ajusté à {start_frame}")

    start_time = (start_frame * total_frames_original) / total_frames_target
    adjusted_start_frame = round(start_time)
    print(f"Start frame ajusté: {adjusted_start_frame}")

    frames_dir = custom_output_dir if custom_output_dir else create_unique_folder(output_dir, "frames_resized")
    os.makedirs(frames_dir, exist_ok=True)

    extracted_count = start_frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, adjusted_start_frame)

    while True:
        success, frame = cap.read()
        if not success:
            break
        resized_frame = resize_frame(frame, target_size)
        frame_filename = os.path.join(frames_dir, f"frame_{extracted_count:06d}.png")
        cv2.imwrite(frame_filename, resized_frame)
        extracted_count += 1

    cap.release()
    print(f"Extraction terminée. {extracted_count - adjusted_start_frame} images dans '{frames_dir}'")
    return frames_dir

def create_video_from_frames(frames_dir, output_path, fps):
    """Crée une vidéo à partir de frames."""
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