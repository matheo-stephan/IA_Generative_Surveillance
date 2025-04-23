# V2 avec CLIP

---

# 🧩 Objectif du Projet

**Ce projet a pour objectif de mettre en place un benchmark vidéo basé sur le dataset UCF Crimes, permettant :**

- L’extraction et le redimensionnement des frames d’une vidéo.
- La génération d’un embedding vectoriel pour chaque frame à l’aide d’un modèle CLIP.
- L’évaluation de robustesse via ajout de bruit ou de modifications artificielles. (optionnelle)
- La génération d'un embedding textuel.
- Ajout des vecteurs dans une BDD vectiorelle.
- La recherche via similarité entre un texte et les images extraites.
- Triage par seuil de similarité ou par similarité décroissante.
- Extraction des frames selectionnées.
- Creation d'une nouvelle vidéo à partir des frames extraites.

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
```python
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
```python
create_video_from_frames(frames_dir, output_path, fps)
```
Permet de recréer une vidéo à partir des images traitées.

---

# 🤖 Encodage et Similarité

## 💡 Encodage

- Utilisation du modèle OpenAI CLIP (`openai/clip-vit-base-patch32`) pour produire des embeddings vectoriels à partir d’images et de textes.
- Normalisation des vecteurs (`L2 norm`) après encodage.

## 🔍 Recherche sémantique

```python
find_similar_images(frames_dir, query_embedding, similarity_threshold, top_x)
```
Calcule la similarité entre l’embedding texte (`query_embedding`) et ceux de chaque image dans la base de données vectorielle (`ChromaDB`).

**Deux critères :**

- Images au-dessus du seuil (`similarity_threshold`)
- Top X images les plus similaires

**✅ Le système adapte dynamiquement le seuil si demandé, en retirant 10% à la similarité maximale détectée.**

## 📐 Similarité

Le système utilise **la similarité cosinus** pour comparer les vecteurs d'embedding d'une image et d'un texte.

### 🔍 Qu'est-ce que la similarité cosinus ?

La similarité cosinus mesure **l’angle** entre deux vecteurs dans un espace vectoriel. Elle ne prend pas en compte la norme (la taille) des vecteurs, mais seulement leur direction.

Elle est calculée avec la formule suivante :
```python
similarity(A, B) = (A ⋅ B) / (||A|| * ||B||)
```
- `A ⋅ B` est le produit scalaire des deux vecteurs.
- `||A||` et `||B||` sont les normes (longueurs) des vecteurs.

### 🎯 Interprétation des valeurs

- **1.0** → les vecteurs pointent exactement dans la même direction (sémantiquement identiques).
- **0.0** → les vecteurs sont orthogonaux (aucune similarité).
- **< 0** → vecteurs opposés (sans lien), mais rarement observé ici car les vecteurs sont généralement **positifs** et **normalisés**.

### 🧠 Pourquoi la normalisation ?

Les embeddings sont souvent **L2-normalisés**, c’est-à-dire que leur norme est ramenée à 1 :

```python
embedding = embedding / embedding.norm(dim=-1, keepdim=True)
```
Cela permet que la similarité cosinus soit simplement le produit scalaire entre deux vecteurs de norme unitaire.

### 📊 Utilisation dans le projet
Dans ce projet, on compare un embedding texte à des embeddings image :
```python
similarity = cosine_similarity(text_embedding, image_embedding)[0][0]
```

**Puis :**

- On trie les résultats par ordre décroissant de similarité.
- On garde ceux au-dessus d’un seuil dynamique (90% de la similarité maximale observée).
- On sauvegarde les X meilleurs résultats dans un dossier dédié.
  
---

# 🧪 Tests de Robustesse

**Fonction intégrée :**

```python
add_salt_pepper_noise(image, amount=0.05)
```
- Ajoute un bruit de type "sel et poivre" (ajout aléatoire de pixel blanc ou noir) à une image.
- Permet de tester la robustesse de la reconnaissance visuelle par similarité.

---

# 💾 Stockage vectoriel avec ChromaDB

Ce projet utilise **ChromaDB** comme base de données vectorielle pour indexer et rechercher efficacement les embeddings des images extraites depuis des vidéos.

## 🧠 Qu'est-ce qu'une base de données vectorielle ?

Une base de données vectorielle permet de stocker des **vecteurs d'embedding** et d'effectuer des recherches de similarité entre eux de manière rapide et scalable. Contrairement aux bases de données classiques qui indexent des textes ou des valeurs numériques simples, ici ce sont des vecteurs à haute dimension (souvent 768 dimensions) qui sont stockés.

Chaque vecteur représente la **sémantique d’une image** (ou d’un texte) générée par un modèle de type CLIP.

## 🏗️ Structure de ChromaDB

Chaque vecteur est stocké dans une **collection**. Une collection contient :
- `id` : un identifiant unique (ex. nom de frame)
- `embedding` : le vecteur numérique représentant l’image
- `document` *(optionnel)* : texte associé (non utilisé ici)
- `metadata` : informations supplémentaires (par exemple le nom de la vidéo, l'index de frame, etc.)

Exemple d’ajout d’un vecteur :
```python
collection.add(
    ids=["frame_000012.png"],
    embeddings=[image_embedding],
    metadatas=[{"video_name": "video_demo", "frame_index": 12}]
)
```
### 🔍 Recherche par similarité

Dans ce projet, une fois les embeddings texte et image extraits et stockés dans ChromaDB, nous utilisons la **similarité cosinus** pour retrouver les images les plus proches d'une requête textuelle. La similarité cosinus mesure l'angle entre deux vecteurs dans l'espace, et varie entre -1 (opposés) et 1 (identiques).

#### Deux approches sont utilisées pour filtrer les résultats :

#### 🧪 1. Recherche `top_x` (meilleures similarités)
Cette méthode consiste à récupérer les **X images les plus proches** de la requête, peu importe leur score exact.

**Utilisation typique :**
```python
top_x_similar = sorted(similarities, key=lambda x: x["similarity"], reverse=True)[:top_x] 
```
Les X images avec la similarité la plus élevée sont ensuite copiées dans un dossier top_x_similar pour être visualisées.

#### 📏 2. Recherche above_threshold (filtrage par seuil)
Cette méthode consiste à ne garder que les images dont la similarité dépasse une certaine valeur seuil.

Le seuil peut être fixé manuellement (similarity_threshold = 0.18), ou défini dynamiquement comme suit :
```python
max_similarity = max(similarities, key=lambda x: x["similarity"])["similarity"]
similarity_threshold = max_similarity * 0.9 # 90% du maximum
```
Seules les images ayant une similarité supérieure ou égale à ce seuil sont conservées et stockées dans un dossier above_threshold.

## 📁 Organisation par vidéo

Chaque vidéo analysée est liée à une collection différente, identifiée via son nom (souvent `sanitized_video_name`). Cela permet de séparer les résultats par vidéo et d’éviter les collisions.

## 🎯 Intégration dans le pipeline

**Voici comment ChromaDB est utilisé dans le programme :**

- À chaque extraction de frame → l’image est encodée en embedding.
- L’embedding est ajouté à la collection correspondant à la vidéo.
- Lors de la recherche, l’embedding du texte est comparé à tous ceux de la collection.
- Les résultats sont triés, et les meilleures images sont copiées dans des dossiers (`top_x_similar`, `above_threshold`).

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
```python
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
```python
for frame in os.listdir(frames_dir):
    image_path = os.path.join(frames_dir, frame)
    embedding = encode_image(image_path)  # ou via votre classe
    # Insérez dans ChromaDB avec image_id = nom du fichier
```

## 4. 💬 Entrer une requête texte

Encodez votre requête :
```python
query = "Person walking in a park"
query_embedding = get_text_embedding(query)
```

## 5. 🧠 Trouver les images similaires

Appelez la fonction de comparaison :
```python
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
```python
create_video_from_frames(
    frames_dir="path/to/frames/analysed_frames",
    output_path="path/to/output_video.mp4",
    fps=FPS_TARGET
)
```
