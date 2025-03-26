import spacy
import numpy as np
from langdetect import detect
from deep_translator import GoogleTranslator
from sentence_transformers import SentenceTransformer

# Charger le modèle de langue et les embeddings
nlp = spacy.load("en_core_web_sm")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")  # Modèle léger d'embeddings


def extract_objects(text):
    """Détecte la langue, traduit si nécessaire et extrait les noms (objets)."""
    lang = detect(text)
    if lang == "fr":
        text = GoogleTranslator(source="fr", target="en").translate(text)

    doc = nlp(text)
    objects = [token.text for token in doc if token.pos_ == "NOUN"]
    return objects


def embed_objects(objects):
    """Transforme les objets en embeddings et retourne les vecteurs."""
    if not objects:
        return {}

    embeddings = embed_model.encode(objects, normalize_embeddings=True)
    return {obj: emb.tolist() for obj, emb in zip(objects, embeddings)}


# ====================== Exemple d'utilisation ======================
text = "une personne marchant avec une valise"

# Étape 1 : Extraction des objets
objects = extract_objects(text)
print(f"Objets détectés : {objects}")

# Étape 2 : Embedding des objets
embeddings = embed_objects(objects)
print("\n🔹 Vecteurs Embeddings :")
for obj, emb in embeddings.items():
    print(f"{obj} : {emb[:5]}...")  # Affiche les 5 premières valeurs du vecteur
