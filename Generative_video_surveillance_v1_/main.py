import os
import datetime
from database import detection, generate_test_data
from video_processing import extract_and_resize_frames, create_video_from_frames
from frame_analysis import analyze_frames, analyze_frames_filtered
from query import ask_query, extract_frames_from_query
from llm_integration import extract_objects, embed_objects
from config import VIDEO_PATH, FRAMES_BASE_DIR, OUTPUT_VIDEO_DIR, FPS_TARGET, TARGET_SIZE
import numpy as np

def main():
    # Étape 1 : Extraction des frames
    frames_dir = extract_and_resize_frames(VIDEO_PATH, FRAMES_BASE_DIR, FPS_TARGET, TARGET_SIZE)
    if not frames_dir:
        return

    # Étape 2 : Analyse des frames
    vectors = analyze_frames(frames_dir)

    # Étape 3 : Générer une vidéo annotée (toutes les détections)
    timestamp = datetime.datetime.now().strftime("%d%m_%H%M%S")
    output_video_path = os.path.join(OUTPUT_VIDEO_DIR, f"video_output_{timestamp}.mp4")
    create_video_from_frames(os.path.join(frames_dir, "analysed_frames"), output_video_path, FPS_TARGET)

    # Étape 4 : Recherche avec une requête LLM
    query_text = "a car"
    objects = extract_objects(query_text)
    embeddings_dict = embed_objects(objects)
    print(f"Objets : {objects}")
    print(f"Embeddings : {embeddings_dict}")

    if embeddings_dict:
        embeddings_list = list(embeddings_dict.values())
        query_vector = np.mean(embeddings_list, axis=0).tolist()
        results, dist = ask_query(detection, query_vector, target_classes=objects, max_dist=0.5)

        print(f"Distance cosinus : {dist}")
        print(f"Résultats de la requête : {len(results['ids'])} détections trouvées")
        for i, metadata in enumerate(results["metadatas"]):
            print(f"Frame {i}: {metadata}")

        # Étape 5 : Extraire les frames de la requête
        frames_selected_dir = extract_frames_from_query(results, os.path.join(frames_dir, "analysed_frames"), OUTPUT_VIDEO_DIR)
        
        # Étape 6 : Filtrer les frames et générer la vidéo query
        filtered_frames_dir = analyze_frames_filtered(frames_selected_dir, objects)
        query_video_path = os.path.join(OUTPUT_VIDEO_DIR, f"video_query_{timestamp}.mp4")
        create_video_from_frames(filtered_frames_dir, query_video_path, FPS_TARGET)

    # Étape 7 : Afficher toutes les détections
    all_data = detection.get(include=["embeddings", "metadatas"])
    for i in range(len(all_data["ids"])):
        print(f"🔹 ID: {all_data['ids'][i]}")
        if all_data['embeddings'] is not None and len(all_data['embeddings']) > 0:
            print(f"🧠 Embedding: {all_data['embeddings'][i]}")
        print(f"📌 Métadonnées: {all_data['metadatas'][i]}")
        print("-" * 40)

if __name__ == "__main__":
    main()