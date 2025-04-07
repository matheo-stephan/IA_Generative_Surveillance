from query import ask_query, extract_frames_from_query
from database import detection, generate_test_data
import os

print("Testing query functions...")

# Simuler des métadonnées réalistes comme dans frame_analysis.py
metadatas = [
    {"image": "frame_000001.png", "class": "cat", "confidence": "0.95"},
    {"image": "frame_000002.png", "class": "dog", "confidence": "0.87"},
    {"image": "frame_000003.png", "class": "bird", "confidence": "0.91"}
]
vectors, _ = generate_test_data(3, 384)  # Ignorer les métadonnées par défaut
for i, (vec, meta) in enumerate(zip(vectors, metadatas)):
    detection.add(ids=str(i), embeddings=[vec], metadatas=[meta])

# Créer un dossier fictif pour simuler les frames
input_dir = "test_input_dir"
os.makedirs(input_dir, exist_ok=True)
for meta in metadatas:
    with open(os.path.join(input_dir, meta["image"]), "w") as f:
        f.write("Fake frame content")

results, dist = ask_query(detection, vectors[0], max_dist=0.8)
print("Résultats de la requête :", results)
frames_dir = extract_frames_from_query(results, input_dir, "test_output_dir")
print("Dossier des frames extraites :", frames_dir)