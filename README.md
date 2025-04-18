# IA_Generative_Surveillance

---

# 🧩 Objectif du Projet

**Ce projet a pour objectif de mettre en place un benchmark vidéo basé sur le dataset UCF Crimes, permettant :**

- L’extraction et le redimensionnement des frames d’une vidéo.
- La génération d’un embedding vectoriel pour chaque frame à l’aide d’un modèle CLIP.
- La recherche sémantique via similarité entre un texte et les images extraites.
- L’évaluation de robustesse via ajout de bruit ou de modifications artificielles.

---

# ⚙️ Dépendances

**Les bibliothèques suivantes sont nécessaires au bon fonctionnement :**

- opencv-python
- torch
- transformers
- sentence-transformers
- chromadb
- numpy
- Pillow
- shutil
- datetime
- os
- cv2
- clip

---

# 📂 Structure Principale du Projet

## 🔧 1. Paramètres de configuration
```
FPS_TARGET = 24   # Frame rate cible
TARGET_SIZE = (640, 360)  # Dimensions de sortie des frames
```
## 📁 2. Extraction et redimensionnement des frames

**Fonctions :**

- `extract_and_resize_frames(...)`: extrait les frames d’une vidéo, les redimensionne, et les stocke dans un dossier.
- `resize_frame(...)`: resize une frame unique à la taille cible.

**🔍 Points techniques :**

- Gestion automatique des FPS et adaptation de start_frame.
- Création d’un répertoire unique pour stocker les frames (via `create_unique_folder`).

## 📹 3. Génération vidéo à partir de frames
```
create_video_from_frames(frames_dir, output_path, fps)
```
Permet de recréer une vidéo à partir des images traitées.

---

# 🤖 Encodage et Similarité

## 💡 Encodage

- Utilisation du modèle OpenAI CLIP (`openai/clip-vit-base-patch32`) pour produire des embeddings vectoriels à partir d’images et de textes.
- Normalisation des vecteurs (`L2 norm`) après encodage.

## 🔍 Recherche sémantique

`find_similar_images(frames_dir, query_embedding, similarity_threshold, top_x)`
Calcule la similarité entre l’embedding texte (`query_embedding`) et ceux de chaque image dans la base de données vectorielle (`ChromaDB`).

**Deux critères :**

- Images au-dessus du seuil (`similarity_threshold`)
- Top X images les plus similaires

**✅ Le système adapte dynamiquement le seuil si demandé, en retirant 10% à la similarité maximale détectée.**

---

# 🧪 Tests de Robustesse

**Fonction intégrée :**

`add_salt_pepper_noise(image, amount=0.05)`
- Ajoute un bruit de type "sel et poivre" à une image.
- Permet de tester la robustesse de la reconnaissance visuelle par similarité.

---

# 💾 Stockage vectoriel avec ChromaDB

- Utilisation de Chroma comme base de données vectorielle.
- Les vecteurs sont insérés avec leurs métadonnées, ce qui permet une récupération simple de l’image associée.
- Si aucune métadonnée n’est stockée, on peut utiliser les noms de fichiers comme identifiants pour retrouver l’image d’origine.

--- 

# 🔄 Pipeline Complet

1. Charger une vidéo
2. Extraire & redimensionner les frames
3. Encoder chaque frame (image → vecteur)
4. Insérer les embeddings dans ChromaDB
5. Encoder la requête (texte → vecteur)
6. Calcul des similarités (cosine)
7. Filtrage des images (top-k ou seuil dynamique)
8. Reconstruction d’une vidéo ou affichage

--- 

# 🚀 Guide d’utilisation du programme

Cette section décrit comment utiliser le système étape par étape, depuis l’entrée d’une vidéo jusqu’à l’obtention des résultats de similarité.

## 1. 📥 Préparer votre vidéo

Placez votre vidéo dans un dossier accessible, par exemple :
`video_path = "path/to/your/video.mp4"`

## 2. 🖼️ Extraire et redimensionner les frames

Appelez la fonction suivante :
```
frames_dir = extract_and_resize_frames(
    video_path=video_path,
    output_dir="path/to/output/frames",
    target_fps=FPS_TARGET,
    target_size=TARGET_SIZE,
    start_frame=0  # optionnel
)
```
Cela extrait les frames, les redimensionne et les stocke dans un dossier.

## 3. 🔎 Encoder les images extraites

Parcourez le dossier de frames pour encoder chaque image via CLIP :
```
for frame in os.listdir(frames_dir):
    image_path = os.path.join(frames_dir, frame)
    embedding = encode_image(image_path)  # ou via votre classe
    # Insérez dans ChromaDB avec image_id = nom du fichier
```

## 4. 💬 Entrer une requête texte

Encodez votre requête :
```
query = "Person walking in a park"
query_embedding = get_text_embedding(query)
```

## 5. 🧠 Trouver les images similaires

Appelez la fonction de comparaison :
```
find_similar_images(
    frames_dir=frames_dir,
    query_embedding=query_embedding,
    similarity_threshold=None,  # auto-threshold si None
    top_x=50
)
```
Cela copie :

- Les images au-dessus du seuil calculé dynamiquement.
- Les Top X plus similaires dans deux sous-dossiers distincts.

## 6. 🎥 Recréer une vidéo (optionnel)

Vous pouvez ensuite recompiler les frames similaires en une vidéo :
```
create_video_from_frames(
    frames_dir="path/to/frames/analysed_frames",
    output_path="path/to/output_video.mp4",
    fps=FPS_TARGET
)
```
