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

---

## 🏗️ Structure du projet

### 1. 📁 Base de données vectorielle
```python
client = chromadb.PersistentClient(path="./chroma_db")
detection = client.get_or_create_collection("detection")
