import torch
import clip
from PIL import Image
import requests
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
import numpy as np
from scipy.spatial.distance import cosine, euclidean
import pandas as pd
import matplotlib.pyplot as plt


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def extract_embedding(frames):
    image_embedding = model.encode_image(frames).float()
    image_embedding /= image_embedding.norm(dim=-1, keepdim=True)
    image_embedding = image_embedding.to(device)
    return image_embedding.cpu()

def compute_similarity(embedding1, embedding2):
    """
    Compute cosine similarity between image and text embeddings.
    """
    similarity_cos = 1 - cosine(embedding1.flatten(), embedding2.flatten())
    return similarity_cos




