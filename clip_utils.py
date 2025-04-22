import torch
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

def load_clip_model(model_name, device):
    """Charge le modèle CLIP et le processeur."""
    processor = CLIPProcessor.from_pretrained(model_name)  # Suppression de use_fast=True
    model = CLIPModel.from_pretrained(model_name)
    model = model.to(device)
    return model, processor

def calculate_similarity(image_paths, texts, model, processor, device):
    """Calcule la similarité entre les images et les textes avec CLIP."""
    with torch.no_grad():
        # Calculer les embeddings des textes
        inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)
        text_embeddings = model.get_text_features(**inputs)
        
        # Charger et traiter les images
        images = [Image.open(image_path).convert("RGB") for image_path in image_paths]
        inputs = processor(images=images, return_tensors="pt").to(device)
        image_embeddings = model.get_image_features(**inputs)
        
        # Calculer la matrice de similarité
        similarity_matrix = cosine_similarity(image_embeddings.cpu(), text_embeddings.cpu())
    
    return similarity_matrix