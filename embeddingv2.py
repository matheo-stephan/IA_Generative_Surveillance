import torch
from torchvision import models
from PIL import Image

import torchvision.transforms as transforms

def get_normalized_image_embedding(image_path):
    # Charger un modèle pré-entraîné (par exemple, ResNet18)
    model = models.resnet18(pretrained=True)
    model.eval()  # Mettre le modèle en mode évaluation

    # Supprimer la dernière couche (classification) pour obtenir l'embedding
    model = torch.nn.Sequential(*list(model.children())[:-1])

    # Transformer l'image (redimensionnement, normalisation, etc.)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Charger l'image
    image = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0)  # Ajouter une dimension batch

    # Obtenir l'embedding
    with torch.no_grad():
        embedding = model(input_tensor).squeeze()

    # Normaliser l'embedding
    normalized_embedding = embedding / embedding.norm(p=2)

    return normalized_embedding