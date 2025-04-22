import pandas as pd
import random
import json

# Charger les annotations
train_csv = "annotations/k700_2020_train.csv"
df = pd.read_csv(train_csv)

# Obtenir toutes les classes uniques
all_classes = df['label'].unique().tolist()
print(f"Nombre total de classes : {len(all_classes)}")

# Sélectionner aléatoirement 50 classes
random.seed(42)  # Pour reproductibilité
selected_classes = random.sample(all_classes, 50)
print(f"Classes sélectionnées : {selected_classes}")

# Sélectionner 5 vidéos par classe
kinetics_subset = []
for label in selected_classes:
    class_videos = df[df['label'] == label]
    # Sélectionner aléatoirement 5 vidéos (ou moins si indisponible)
    selected_videos = class_videos.sample(n=min(5, len(class_videos)), random_state=42)
    for _, row in selected_videos.iterrows():
        kinetics_subset.append({
            "youtube_id": row['youtube_id'],
            "label": row['label'],
            "start_time": row['time_start'],
            "end_time": row['time_end']
        })

# Sauvegarder le sous-ensemble
with open("annotations/selected_subset.json", "w") as f:
    json.dump(kinetics_subset, f, indent=4)
print(f"Sous-ensemble sauvegardé : {len(kinetics_subset)} vidéos")