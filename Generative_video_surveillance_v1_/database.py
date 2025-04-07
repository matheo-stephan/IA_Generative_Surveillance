import chromadb

# Initialisation du client ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")

# Création des collections
detection = client.get_or_create_collection("detection")
image = client.get_or_create_collection("image")

def add_detection(ids, embeddings, metadatas):
    """Ajoute des détections à la collection."""
    detection.add(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas
    )

def del_detection(ids):
    """Supprime des détections par ID."""
    detection.delete(ids=ids)

def generate_test_data(n_vectors=10, dim=384):
    """Génère des données de test aléatoires."""
    import numpy as np
    import random
    vectors = np.random.rand(n_vectors, dim).tolist()
    metadatas = [{"category": random.choice(["cat", "dog", "bird"])} for _ in range(n_vectors)]
    return vectors, metadatas