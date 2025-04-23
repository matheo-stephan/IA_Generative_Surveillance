# 🔍 BDD et Système de Recherche Vectorielle avec ChromaDB et Requêtes LLM

Ce projet met en place une base de données vectorielle locale avec [ChromaDB](https://docs.trychroma.com/) pour stocker et rechercher des embeddings. Il permet également d'intégrer des requêtes en langage naturel en utilisant des LLM pour extraire et encoder des objets pertinents.

---

## 📦 Fonctionnalités principales

- Stockage local de vecteurs d'embedding avec **ChromaDB**.
- Ajout, suppression et consultation d'embeddings avec métadonnées.
- Recherche par **proximité vectorielle** (distance cosinus).
- Filtrage par **métadonnées** (ex: catégorie).
- Génération de vecteurs de test.
- Intégration d’un **LLM** pour extraire des objets depuis du texte libre et générer leurs embeddings.
- **(NOUVEAU)** ➕ Intégration d’un module complet de **prétraitement** pour les vidéos/images/textes.

ℹ️ Le **prétraitement** (segmentation des vidéos, extraction de frames, nettoyage, etc.) est détaillé dans un autre fichier `README.md` dans la branche Pre-Traitement.

---

## 🏗️ Structure du projet

### 1. 📁 Base de données vectorielle
```python
client = chromadb.PersistentClient(path="./chroma_db")
detection = client.get_or_create_collection("detection")
```
### 2. 🧠 Fonctions clés
`add_detection(ids, embeddings, metadatas)`
→ Ajoute des vecteurs avec métadonnées à la collection.

`del_detection(ids)`
→ Supprime des vecteurs selon leurs IDs.

`generate_test_data(n_vectors=10, dim=384)`
→ Génère des vecteurs de test avec catégories aléatoires (cat, dog, bird).

`ask_query(collection, query_vector, query=None, max_dist=1)`
→ Recherche les vecteurs les plus proches selon la distance cosinus, avec filtrage optionnel par métadonnées.

---

## 🚀 Exécution de base

### ➕ Ajouter des vecteurs
```python
vectors, metadatas = generate_test_data()
add_detection(ids=[str(i) for i in range(10)], embeddings=vectors, metadatas=metadatas)
```

### 🔍 Requête simple
```python
results, dist = ask_query(detection, vectors[0], query={"category": "cat"}, max_dist=0.8)
```

---

## 🤖 Intégration LLM

**Objectif**

Transformer une requête texte (ex: "un chien courant après une balle") en vecteur d’embedding via un LLM, puis rechercher les éléments similaires dans la base vectorielle.

**Étapes :**
```python
from Requete_LLM import extract_objects, embed_objects

text = "un chien courant après une balle"
objects = extract_objects(text)
embeddings = embed_objects(objects)

query_vector = np.mean(list(embeddings.values()), axis=0).tolist()
results, dist = ask_query(detection, query_vector, max_dist=0.95)
```

---

## 📌 Exemple de sortie
```yaml
🔹 ID: 3
🧠 Embedding: [...]
📌 Métadonnées: {'category': 'dog'}
📏 Distance: 0.47
```

---

## 🧪 Dépendances

- ChromaDB
- numpy
- scipy
- Optionnel : matplotlib, seaborn (pour visualiser les performances)

---

## 🧩 Modules associés

`Requete_LLM.py` – extraction & embedding d’objets depuis un texte
