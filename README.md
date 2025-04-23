# 🎥 Projet de Traitement Vidéo avec YOLO & Embeddings NLP

## 📌 Contexte

Ce projet a pour but d'extraire des images à partir de vidéos, d'y détecter automatiquement les objets présents à l’aide du modèle **YOLO**, de générer des **embeddings textuels** à partir de ces objets détectés à l’aide de **SentenceTransformer**, et enfin de reconstruire une **vidéo annotée** à partir des images traitées.

**🔁 Ce projet s’inscrit dans le cadre de la branche `BDD`, dédiée à l’ingestion, la transformation et l’enrichissement de données pour les traitements aval.**

--- 

## 🚀 Fonctionnalités clés

### 🎞️ 1. Extraction de frames
- Découpe une vidéo en images selon un `FPS_TARGET` configurable.
- Redimensionnement automatique des images à une taille prédéfinie (`TARGET_SIZE`).

### 🧠 2. Détection d’objets
- Utilisation du modèle YOLO (`fichier yolo11n.pt`).
- Sauvegarde des images annotées dans un sous-dossier `analysed_frames`.

### 🧬 3. Génération d'embeddings
- Pour chaque objet détecté, un embedding vectoriel est généré à partir du label et de la confiance.
- Utilisation du modèle `all-MiniLM-L6-v2` de SentenceTransformer.
- Embeddings sérialisés dans un fichier `embeddings.pkl`.

### 🎥 4. Reconstitution vidéo
- Génère une vidéo à partir des frames annotées (`images.png`) en utilisant OpenCV.
- La vidéo de sortie est horodatée pour éviter tout écrasement.

---

## 🧪 Pipeline de traitement
```text
[Vidéo Originale]
        ↓
[Extraction + Redimensionnement de Frames]
        ↓
[Détection d'Objets avec YOLO]
        ↓
[Annotation + Embeddings avec SentenceTransformer]
        ↓
[Frames Annotées]
        ↓
[Vidéo Annotée + Embeddings.pkl]
```

--- 

## 🛠️ Pré-requis & Installation

**📦 Dépendances Python**
```bash
pip install opencv-python ultralytics sentence-transformers
```

**📂 Ressources nécessaires**
- Un fichier vidéo (ex. `Video_2.mp4`)
- Le modèle YOLO (`yolo11n.pt`) doit être placé dans le répertoire du script ou téléchargeable via Ultralytics.

--- 

## ⚙️ Paramètres configurables

| Paramètre         | Description                                      | Valeur par défaut        |
|------------------|--------------------------------------------------|--------------------------|
| `FPS_TARGET`     | Fréquence d’extraction des images (images/sec)  | `24`                     |
| `TARGET_SIZE`    | Dimensions des images redimensionnées `(w, h)`  | `(640, 360)`             |
| `start_frame`    | Frame de départ pour l'extraction               | `0`                      |
| `video_path`     | Chemin d'accès à la vidéo à traiter             | `"./Video/Video_2.mp4"`  |
| `frames_base_dir`| Répertoire de stockage des frames               | `"./Frames/"`            |
| `custom_folder`  | Répertoire de sortie personnalisé (optionnel)   | `""` (auto-généré)       |

---

## 🧾 Exemple d’utilisation

**🎯 Extraction et annotation**

Dans le script, les chemins sont définis comme suit :
```python
video_path = "C:/.../Video_2.mp4"
frames_base_dir = "C:/.../Frames"
```

**▶️ Exécution automatique :**
```bash
python main.py
```
- Les frames seront extraites et stockées dans un dossier horodaté.
- Les objets seront détectés puis annotés.
- Une vidéo annotée sera sauvegardée à l’emplacement défini.
- Les vecteurs d’embeddings seront stockés dans `embeddings.pkl` (pour les test).

---

## 🌿 Branche BDD

⚠️ Ce script est utilisé spécifiquement dans la branche `BDD` du projet.
Cette branche est consacrée à la collecte et la structuration des données, avant leur envoi en base ou leur utilisation dans des modèles d'IA aval.

--- 

## 📤 Structure attendue des fichiers
```
Projet/
├── Vidéo/
│   └── Video_2.mp4
├── Frames/
│   ├── frames_resized_2304_143210/
│   │   ├── frame_000000.png
│   │   └── analysed_frames/
│         └── frame_000000.png (annotée)
├── embeddings.pkl
└── video_output_2304_143215.mp4
```

