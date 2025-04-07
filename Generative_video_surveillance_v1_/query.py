from scipy.spatial.distance import cosine
import numpy as np
from database import detection
import datetime
import os
import shutil
from video_processing import create_unique_folder, create_video_from_frames
from config import FPS_TARGET

def ask_query(collection, query_vector, target_classes=None, max_dist=0.5):
    """Recherche par proximité de vecteurs avec filtrage par classes."""
    all_data = collection.get(include=["embeddings", "metadatas"])
    num_documents = len(all_data["ids"])
    
    cosine_dist = [cosine(query_vector, np.squeeze(np.array(embedding))) for embedding in all_data["embeddings"]]
    results = {"ids": [], "embeddings": [], "metadatas": [], "distances": []}

    for i in range(num_documents):
        metadata = all_data["metadatas"][i]
        class_name = metadata.get("class", "")
        if (cosine_dist[i] <= max_dist) and (target_classes is None or class_name in target_classes):
            results["ids"].append(all_data["ids"][i])
            results["embeddings"].append(all_data["embeddings"][i])
            results["metadatas"].append(metadata)
            results["distances"].append(cosine_dist[i])

    return results, cosine_dist

def extract_frames_from_query(results, input_dir, output_dir):
    """Extrait les frames correspondant aux résultats de la requête."""
    frames_selected_dir = create_unique_folder(output_dir, "frames_selected")
    os.makedirs(frames_selected_dir, exist_ok=True)

    images_id = [metadata["image"] for metadata in results["metadatas"]]
    for image_id in images_id:
        image_path = os.path.join(input_dir, image_id)
        if os.path.exists(image_path):
            shutil.copy(image_path, frames_selected_dir)
        else:
            print(f"Avertissement : {image_path} n'existe pas, ignoré.")

    output_video_path = os.path.join(output_dir, f"video_query_{datetime.datetime.now().strftime('%d%m_%H%M%S')}.mp4")
    create_video_from_frames(frames_selected_dir, output_video_path, FPS_TARGET)
    return frames_selected_dir