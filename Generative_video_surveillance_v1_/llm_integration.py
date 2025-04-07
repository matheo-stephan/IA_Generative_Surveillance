import spacy
import numpy as np
from langdetect import detect
from deep_translator import GoogleTranslator
from sentence_transformers import SentenceTransformer

nlp = spacy.load("en_core_web_sm")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_objects(text):
    """Détecte la langue, traduit si nécessaire et extrait les noms."""
    lang = detect(text)
    if lang == "fr":
        text = GoogleTranslator(source="fr", target="en").translate(text)

    doc = nlp(text)
    objects = [token.text for token in doc if token.pos_ == "NOUN"]
    return objects

def embed_objects(objects):
    """Transforme les objets en embeddings."""
    if not objects:
        return {}
    embeddings = embed_model.encode(objects, normalize_embeddings=True)
    return {obj: emb.tolist() for obj, emb in zip(objects, embeddings)}