# 🧠 Extraction et Embedding de Noms à partir de Texte Multilingue

Ce script permet d’extraire les **noms (objets)** d’un texte en français ou en anglais, de les traduire si nécessaire, puis de générer des **vecteurs d’embeddings** à l’aide de modèles NLP préentraînés.

> 📂 **Ce script est utilisé dans la branche `BDD` du projet**, dédiée à l’enrichissement sémantique des données avant stockage ou exploitation dans des systèmes intelligents.

---

## 📌 Fonctionnalités principales

- 🌍 Détection automatique de la langue du texte
- 🌐 Traduction vers l’anglais si le texte est en français (via `GoogleTranslator`)
- 📦 Extraction grammaticale des noms à l’aide de `spaCy`
- 🧬 Génération d’embeddings pour chaque nom avec `SentenceTransformer`

---

## 🧰 Dépendances

```bash
pip install spacy langdetect deep-translator sentence-transformers
python -m spacy download en_core_web_sm
```

--- 

## ⚙️ Fonctionnement

**1. Extraction des objets (noms)**
Détection de la langue, traduction si nécessaire, puis parsing du texte avec `spaCy` pour extraire tous les noms (`NOUN`).

**2. Embedding des objets**
Chaque objet extrait est transformé en vecteur via le modèle `all-MiniLM-L6-v2`.

---

## 💡 Exemple d’utilisation
```python
text = "une personne marchant avec une valise"

# Étape 1 : Extraction
objects = extract_objects(text)
print(f"Objets détectés : {objects}")

# Étape 2 : Embedding
embeddings = embed_objects(objects)
for obj, emb in embeddings.items():
    print(f"{obj} : {emb[:5]}...")
```

## 🔍 Résultat attendu :
```less
Objets détectés : ['personne', 'valise']

🔹 Vecteurs Embeddings :
personne : [-0.123, 0.451, -0.098, 0.221, ...]
valise : [0.042, -0.331, 0.104, 0.202, ...]
```

---

## 📦 Fonctions disponibles

`extract_objects(text: str) -> List[str]`
- Détecte la langue (fr ou autre)
- Traduit vers l’anglais si besoin
- Extrait les noms (objets) du texte

`embed_objects(objects: List[str]) -> Dict[str, List[float]]`
- Calcule les vecteurs pour chaque mot
- Retourne un dictionnaire : {objet: vecteur_embedding}

---

## 🧪 Cas d’usage

- 🔎 Recherche sémantique
- 🧠 Analyse de descriptions visuelles ou textuelles
- 📊 Clustering d’objets par similarité
- 🗣️ Prétraitement NLP pour IA générative ou multimodale

---

## 🧩 Intégration dans la branche BDD

Ce module est intégré dans la **branche BDD** pour enrichir les données textuelles provenant :
- de descriptions d’images/vidéos,
- de métadonnées utilisateurs,
- ou de résultats de détection automatique.

L’objectif est de normaliser et vectoriser ces objets afin de faciliter :
- l’indexation,
- la recherche par similarité,
- et les traitements downstream (ex : apprentissage, stockage vectoriel, etc.).
