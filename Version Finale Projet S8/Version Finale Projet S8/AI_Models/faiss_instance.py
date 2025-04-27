from AI_Models.faissClient import FaissClient
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAISS_DATA_PATH = os.path.join(BASE_DIR, "data.faiss")

# Créer une instance unique de FaissClient
faiss_client = FaissClient(dimension=512)

# Charger l'index FAISS
faiss_client.load(FAISS_DATA_PATH)