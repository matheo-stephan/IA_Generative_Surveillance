import cv2
import os
import datetime

FPS_TARGET = 15  # FPS cible
TARGET_SIZE = (1920, 1080)  # Nouvelle taille des frames

def create_unique_folder(base_dir, prefix="frames"):
    """
    Crée un dossier unique pour stocker les frames redimensionnées en utilisant un horodatage.
    """
    timestamp = datetime.datetime.now().strftime("%d%m_%H%M%S")
    folder_path = os.path.join(base_dir, f"{prefix}_{timestamp}")
    os.makedirs(folder_path, exist_ok=True)
    return folder_path

def resize_frame(frame, size):
    """Redimensionne une image à la taille spécifiée."""
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)

def extract_and_resize_frames(video_path, output_dir, target_fps, target_size):
    """
    Extrait les frames d'une vidéo, les redimensionne et les enregistre.
    """
    if not os.path.exists(video_path):
        print(f"Erreur: Le fichier vidéo '{video_path}' n'existe pas.")
        return None
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Erreur: Impossible d'ouvrir la vidéo '{video_path}'")
        return None
    
    original_fps = cap.get(cv2.CAP_PROP_FPS)  
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  
    duration = total_frames / original_fps  

    frames_dir = create_unique_folder(output_dir, "frames_resized")
    target_timestamps = [i / target_fps for i in range(int(duration * target_fps))]

    extracted_count = 0
    for timestamp in target_timestamps:
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)  
        success, frame = cap.read()
        if not success:
            break  
        
        resized_frame = resize_frame(frame, target_size)
        frame_filename = os.path.join(frames_dir, f"frame_{extracted_count:06d}.png")
        cv2.imwrite(frame_filename, resized_frame)
        extracted_count += 1
    
    cap.release()
    print(f"Extraction et redimensionnement terminés. {extracted_count} images enregistrées dans '{frames_dir}'")
    return frames_dir

def create_video_from_frames(frames_dir, output_path, fps):
    """
    Crée une vidéo à partir des frames extraites.
    """
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

def create_video_info(video_path, output_path, fps):
    """
    Crée un fichier texte contenant les caractéristiques de la vidéo.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Erreur: Impossible d'ouvrir la vidéo '{video_path}'")
        return
    
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / original_fps

    with open(output_path, "w") as f:
        f.write(f"Vidéo: {video_path}\n")
        f.write(f"Nombre de frames: {total_frames}\n")
        f.write(f"Durée: {duration:.2f} secondes\n")
        f.write(f"FPS original: {original_fps}\n")
        f.write(f"FPS cible: {fps}\n")
    
    print(f"Fichier d'informations généré à '{output_path}'")

# ============================
#           EXÉCUTION
# ============================

# Récupérer le répertoire du script
current_dir = os.getcwd()

# Définir les dossiers pour la vidéo source et la vidéo finale
video_dir = os.path.join(current_dir, "video")
output_dir = os.path.join(current_dir, "video_finale")

# Créer les dossiers s'ils n'existent pas
os.makedirs(video_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

video_path = os.path.join(video_dir, "videos.mp4")  # Vidéo source dans "video"
frames_base_dir = os.path.join(current_dir, "Frames")  # Dossier où seront stockées les frames
timestamp = datetime.datetime.now().strftime("%d%m_%H%M%S")
output_video_path = os.path.join(output_dir, f"video_output_{timestamp}.mp4") # Vidéo finale dans "video_finale"

# Étape 1 : Extraire, redimensionner et sauvegarder les frames
frames_dir = extract_and_resize_frames(video_path, frames_base_dir, FPS_TARGET, TARGET_SIZE)
video_info_path = os.path.join(frames_dir, "video_info.txt")

# Étape 2 : Générer la vidéo à partir des frames redimensionnées
if frames_dir:
    create_video_from_frames(frames_dir, output_video_path, FPS_TARGET)

# Étape 3 : Générer les infos de la vidéo
create_video_info(video_path, video_info_path, FPS_TARGET)