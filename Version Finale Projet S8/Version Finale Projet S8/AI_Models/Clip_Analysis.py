from imports import torch, os, shutil, cv2
from datetime import datetime
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from transformers import CLIPProcessor, CLIPModel
from AI_Models.faiss_instance import faiss_client

# --- Classe EmbeddingComparator ---
class EmbeddingComparator:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        """
        Initialise le modèle CLIP (base par défaut) et son processor.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name, use_fast=False)

    def encode_image(self, image_path):
        """
        Encode une image en vecteur d'embedding CLIP normalisé.
        Args:
            image_path (str): Chemin vers l'image.
        Returns:
            np.ndarray: Embedding de l'image.
        """
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
        image_embedding = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_embedding.cpu().numpy()

    def encode_text(self, text):
        """
        Encode un texte en vecteur d'embedding CLIP normalisé.

        Args:
            text (str): Phrase ou mot à encoder.
        Returns:
            np.ndarray: Embedding du texte.
        """
        inputs = self.processor(text=text, return_tensors="pt", padding=True).to(self.device)

        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)

        text_embedding = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_embedding.cpu().numpy()

    def compare_embeddings(self, embedding1, embedding2):
        """
        Calcule la similarité cosinus entre deux vecteurs d'embedding.
        Args:
            embedding1 (np.ndarray): Premier vecteur.
            embedding2 (np.ndarray): Deuxième vecteur.
        Returns:
            float: Score de similarité (entre -1 et 1).
        """
        return cosine_similarity(embedding1, embedding2)[0][0]


# --- Classe ClipAnalysis ---
class ClipAnalysis:
    def __init__(self, bank_path=None):
        """
        Initialise l'analyse avec un chemin facultatif pour la banque.
        """
        self.bank_path = bank_path
        self.comparator = EmbeddingComparator()  # Assurez-vous que cette classe est correctement définie

    def create_video_from_frames(self, frames_dir, output_path, fps, frame_extension=".jpg"):
        """
        Crée une vidéo à partir des frames d'un dossier donné.

        Args:
            frames_dir (str): Chemin vers le dossier contenant les frames.
            output_path (str): Chemin de sortie pour la vidéo générée.
            fps (int): Nombre d'images par seconde pour la vidéo.
            frame_extension (str): Extension des frames (par défaut ".jpg").
        """
        frames = sorted(f for f in os.listdir(frames_dir) if f.endswith(frame_extension))
        if not frames:
            print(f"Error: No frames found in '{frames_dir}' with extension '{frame_extension}'")
            return

        first_frame = cv2.imread(os.path.join(frames_dir, frames[0]))
        height, width, _ = first_frame.shape

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for frame_filename in frames:
            frame = cv2.imread(os.path.join(frames_dir, frame_filename))
            out.write(frame)

        out.release()
        print(f"Video saved to '{output_path}'")
        
    def analyse_prompt(self, analysis_path, prompt, similarity_threshold=None):
        """
        Analyse un prompt en comparant les embeddings du prompt avec ceux des images dans chaque collection.
        Inclut un dossier pour les Top5 et un autre pour les frames au-dessus d'un seuil de similarité.
        """
        comparator = self.comparator
        text_embedding = comparator.encode_text(prompt)  # Étape 1 : Encoder le prompt
        print(f"🔍 Embedding du prompt '{prompt}' : {text_embedding}")

        # Étape 2 : Comparer les embeddings dans chaque collection
        top_k = 5  # Nombre de résultats à retourner par collection
        collections_results = {}
        filtered_results = {}
        global_max_similarity = 0.0  # Initialiser la similarité maximale globale à 0

        # Parcourir chaque collection pour calculer les similarités
        for collection_name in set(meta["collection"] for meta in faiss_client.metadata.values()):
            # Filtrer les embeddings de la collection
            collection_embeddings = [
                (idx, meta) for idx, meta in faiss_client.metadata.items()
                if meta["collection"] == collection_name
            ]

            if not collection_embeddings:
                print(f"⚠️ Aucun embedding disponible pour la collection '{collection_name}'.")
                continue

            # Comparer chaque embedding avec le prompt
            all_results = []  # Stocker toutes les frames pour cette collection
            for idx, meta in collection_embeddings:
                try:
                    image_embedding = faiss_client.index.reconstruct(idx)
                    similarity = comparator.compare_embeddings(text_embedding, [image_embedding])
                    all_results.append({"id": meta["name"], "similarity": similarity})

                    # Mettre à jour la similarité maximale globale si nécessaire
                    if similarity > global_max_similarity:
                        global_max_similarity = similarity
                except Exception as e:
                    print(f"❌ Erreur lors de la reconstruction de l'embedding pour l'indice {idx}: {e}")
                    continue

            # Trier toutes les frames par similarité décroissante
            all_results = sorted(all_results, key=lambda x: x["similarity"], reverse=True)

            # Garder uniquement le Top 5 pour cette collection
            collections_results[collection_name] = all_results[:top_k]

            # Filtrer les frames au-dessus du seuil pour cette collection
            filtered_results[collection_name] = all_results

        # Calculer le seuil de similarité si non fourni
        if similarity_threshold is None:
            similarity_threshold = global_max_similarity * 0.9  # 90% de la similarité maximale globale
            print(f"🔧 Similarity threshold calculé : {similarity_threshold}")

        # Étape 3 : Créer un dossier pour enregistrer les résultats
        request_folder = os.path.join(analysis_path, f"Request_{prompt}")
        os.makedirs(request_folder, exist_ok=True)
        print(f"📂 Dossier de résultats créé : {request_folder}")

        # --- AboveThreshold Analysis ---
        AbTH_folder = os.path.join(request_folder, "AboveThreshold")
        os.makedirs(AbTH_folder, exist_ok=True)
        abth_info_path = os.path.join(AbTH_folder, "abTH_info.txt")
        with open(abth_info_path, "w") as abth_info_file:
            abth_info_file.write(f"Date: {datetime.now()}\n")
            abth_info_file.write(f"Prompt: {prompt}\n")
            abth_info_file.write(f"Similarity Threshold: {similarity_threshold}\n")
            abth_info_file.write(f"Number of collections: {len(filtered_results)}\n\n")

            # Écrire une matrice des meilleurs résultats pour chaque collection
            abth_info_file.write("Matrix of Best Results per Collection:\n")
            abth_info_file.write(f"{'Collection':<20}{'Best Frame':<30}{'Similarity':<10}{'Total Frames':<15}{'Avg Similarity':<15}\n")
            abth_info_file.write("-" * 90 + "\n")

            for collection_name, results in filtered_results.items():
                # Filtrer les frames au-dessus du seuil pour cette collection
                above_threshold_frames = [
                    result for result in results if result["similarity"] >= similarity_threshold
                ]

                if above_threshold_frames:  # Vérifier s'il y a des frames au-dessus du seuil
                    # Trouver le meilleur résultat dans la collection
                    best_result = max(above_threshold_frames, key=lambda x: x["similarity"])
                    best_frame = best_result["id"]
                    best_similarity = best_result["similarity"]

                    # Calculer le nombre total de frames et la similarité moyenne
                    total_frames = len(above_threshold_frames)
                    avg_similarity = sum(result["similarity"] for result in above_threshold_frames) / total_frames

                    # Écrire les informations dans la matrice
                    abth_info_file.write(f"{collection_name:<20}{best_frame:<30}{best_similarity:<10.4f}{total_frames:<15}{avg_similarity:<15.4f}\n")

                    # Créer un dossier parent "AbTH_Frames" pour toutes les collections
                    frames_folder = os.path.join(AbTH_folder, "AbTH_Frames")
                    os.makedirs(frames_folder, exist_ok=True)

                    # Créer un sous-dossier pour chaque collection dans "AbTH_Frames"
                    collection_folder = os.path.join(frames_folder, f"AbTH_{collection_name}")
                    os.makedirs(collection_folder, exist_ok=True)

                    # Copier les images correspondantes dans le sous-dossier de la collection
                    for result in above_threshold_frames:
                        image_id = result["id"]
                        image_folder = f"{collection_name}_frames"
                        image_path = os.path.join("Bank", "Frames", image_folder, image_id)
                        if os.path.exists(image_path):
                            shutil.copy(image_path, os.path.join(collection_folder, os.path.basename(image_id)))

                    # Créer une vidéo à partir des frames au-dessus du seuil
                    video_output_path = os.path.join(AbTH_folder, f"{collection_name}_above_threshold.mp4")
                    self.create_video_from_frames(collection_folder, video_output_path, fps=30, frame_extension=".jpg")

        return request_folder