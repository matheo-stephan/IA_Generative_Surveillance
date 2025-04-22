import os
import json
import yt_dlp
import cv2

def download_kinetics_subset(kinetics_subset, video_dir, annotation_file):
    """Télécharge un sous-ensemble de vidéos Kinetics et génère un fichier d'annotations."""
    os.makedirs(video_dir, exist_ok=True)
    annotations = {}
    video_ids = []
    count = 0
    
    for video in kinetics_subset:
        youtube_id = video['youtube_id']
        start_time = video['start_time']
        end_time = video['end_time']
        label = video['label']
        
        print(f"Téléchargement de la vidéo {youtube_id} ({label})...")
        video_path = os.path.join(video_dir, f"video_{count+1}.mp4")
        
        # Options pour yt-dlp
        ydl_opts = {
            'format': 'bestvideo[height<=480]+bestaudio/best[height<=480]',  # Limiter la résolution pour économiser de la bande passante
            'outtmpl': video_path,
            'quiet': True,
            'merge_output_format': 'mp4',
            'download_ranges': lambda _, __: [{'start_time': start_time, 'end_time': end_time}],
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([f"https://www.youtube.com/watch?v={youtube_id}"])
            
            # Vérifier si la vidéo est lisible
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Erreur : La vidéo téléchargée {video_path} n'est pas lisible.")
                os.remove(video_path)
                continue
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if video_fps <= 0 or total_frames <= 0:
                print(f"Erreur : La vidéo {video_path} a des métadonnées invalides (FPS={video_fps}, frames={total_frames}).")
                os.remove(video_path)
                cap.release()
                continue
            cap.release()
            
            video_id = f"video_{count+1}"
            annotations[video_id] = {
                'path': f"video_{count+1}.mp4",
                'label': label
            }
            video_ids.append(video_id)
            count += 1
        except Exception as e:
            print(f"Erreur lors du téléchargement de {youtube_id} : {e}")
            continue
    
    # Sauvegarder les annotations
    with open(annotation_file, 'w') as f:
        json.dump(annotations, f, indent=4)
    
    print(f"Téléchargement terminé. Annotations sauvegardées dans {annotation_file}.")
    print(f"Vidéos téléchargées : {video_ids}")