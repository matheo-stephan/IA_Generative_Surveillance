import os
import torch
import shutil
import numpy as np
import json
from download_videos import download_kinetics_subset
from process_videos import load_kinetics_annotations
from clip_utils import load_clip_model, calculate_similarity
from display_results import display_similarity

def calculate_metrics(similarity_matrix, labels, text_labels):
    """Calcule les métriques de performance."""
    num_samples = similarity_matrix.shape[0]
    
    # Précision@1 et Précision@3
    precision_at_1 = 0
    precision_at_3 = 0
    
    # Distances moyennes
    correct_distances = []
    incorrect_distances = []
    
    for i in range(num_samples):
        true_label = labels[i]
        true_idx = text_labels.index(true_label)  # Indice de la query correcte
        
        # Prédictions triées par score de similarité
        similarities = similarity_matrix[i]
        sorted_indices = np.argsort(similarities)[::-1]  # Indices triés par ordre décroissant
        
        # Précision@1
        if sorted_indices[0] == true_idx:
            precision_at_1 += 1
        
        # Précision@3
        if true_idx in sorted_indices[:3]:
            precision_at_3 += 1
        
        # Distances
        correct_distances.append(similarities[true_idx])
        incorrect_indices = [idx for idx in range(len(text_labels)) if idx != true_idx]
        incorrect_distances.extend(similarities[incorrect_indices])
    
    precision_at_1 = precision_at_1 / num_samples
    precision_at_3 = precision_at_3 / num_samples
    avg_correct_distance = np.mean(correct_distances)
    avg_incorrect_distance = np.mean(incorrect_distances)
    
    return {
        "precision@1": precision_at_1,
        "precision@3": precision_at_3,
        "avg_correct_distance": avg_correct_distance,
        "avg_incorrect_distance": avg_incorrect_distance
    }

def main():
    # Nettoyer le dossier temp_frames s'il existe
    temp_dir = "temp_frames"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

    # Charger le sous-ensemble sélectionné
    with open("annotations/selected_subset.json", "r") as f:
        kinetics_subset = json.load(f)

    video_dir = "videos"
    annotation_file = "annotations.json"
    max_videos = len(kinetics_subset)  # 250 vidéos (50 concepts x 5 vidéos)
    model_name = "openai/clip-vit-base-patch32"

    # Étape 1 : Télécharger les vidéos et générer les annotations
    print("Étape 1 : Téléchargement des vidéos...")
    download_kinetics_subset(kinetics_subset, video_dir, annotation_file)

    # Étape 2 : Charger le modèle CLIP
    print("Étape 2 : Chargement du modèle CLIP...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor = load_clip_model(model_name, device)

    # Étape 3 : Extraire les frames et charger les annotations
    print("Étape 3 : Extraction des frames et chargement des annotations...")
    all_frame_paths, labels, video_ids = load_kinetics_annotations(annotation_file, video_dir, max_videos)
    print(f"Vidéos traitées : {len(all_frame_paths)}")
    print(f"Étiquettes : {labels}")
    print(f"IDs des vidéos : {video_ids}")

    # Vérifier si des frames ont été extraites
    if not all_frame_paths:
        print("Erreur : Aucune frame n'a été extraite. Vérifiez que les vidéos ont été téléchargées correctement.")
        return

    # Étape 4 : Créer des requêtes textuelles descriptives et sélectionner la meilleure frame
    print("Étape 4 : Calcul des similarités et sélection de la meilleure frame...")
    
    # Liste des classes uniques (concepts) dans le sous-ensemble
    concepts = sorted(list(set(labels)))
    
    # Définir des queries descriptives spécifiques pour chaque concept
    query_dict = {
        "playing basketball": "A group of people playing basketball on an outdoor court with a hoop",
        "playing soccer": "A group of people kicking a soccer ball on a grassy field with goalposts",
        "cooking": "A person in a kitchen chopping vegetables on a cutting board",
        "painting": "A person painting a canvas with colorful paints in a studio",
        "swimming": "A person swimming in a blue pool under the sun",
        "reading a book": "A person sitting on a couch reading a book in a cozy room",
        "playing guitar": "A person playing an acoustic guitar in a park",
        "riding a bike": "A person riding a bike on a sunny trail with trees",
        "doing yoga": "A person doing yoga on a mat in a calm studio",
        "writing on a whiteboard": "A person writing on a whiteboard in a classroom",
        "gardening": "A person gardening in a backyard with flowers and plants",
        "dancing": "A group of people dancing in colorful costumes on a stage with bright lights",
        "jumping": "A person in a blue outfit jumping on a trampoline in a gym",
        "running": "A person wearing a red shirt running in a park with green grass",
        "singing": "A person singing into a microphone on a stage with spotlights",
        # Ajoutez les autres classes ici...
    }

    # Générer les queries
    text_queries = []
    for concept in concepts:
        query = query_dict.get(concept, f"A person performing the action of {concept} in a typical setting")
        text_queries.append(query)
    print(f"Queries générées : {text_queries}")

    # Sélectionner la meilleure frame pour chaque vidéo
    best_frame_paths = []  # Liste des chemins des meilleures frames (une par vidéo)
    best_similarity_matrices = []  # Liste des matrices de similarité pour les meilleures frames

    for video_idx, frame_paths in enumerate(all_frame_paths):
        print(f"Analyse des frames pour la vidéo {video_ids[video_idx]}...")
        if not frame_paths:
            print(f"Aucune frame pour la vidéo {video_ids[video_idx]}. Passage à la suivante.")
            continue

        # Calculer la similarité pour toutes les frames de cette vidéo
        similarity_matrix = calculate_similarity(frame_paths, text_queries, model, processor, device)

        # Trouver la frame avec le meilleur score de similarité (maximum sur toutes les colonnes pour chaque frame)
        best_score_idx = np.argmax(similarity_matrix.max(axis=1))
        best_frame_path = frame_paths[best_score_idx]
        best_frame_similarity = similarity_matrix[best_score_idx]

        print(f"Meilleure frame pour {video_ids[video_idx]} : {best_frame_path}, scores : {best_frame_similarity}")
        best_frame_paths.append(best_frame_path)
        best_similarity_matrices.append(best_frame_similarity)

        # Supprimer les autres frames pour économiser de l'espace
        for frame_path in frame_paths:
            if frame_path != best_frame_path:
                os.remove(frame_path)

    # Vérifier si des frames ont été sélectionnées
    if not best_frame_paths:
        print("Erreur : Aucune frame n'a été sélectionnée après analyse.")
        return

    # Étape 5 : Calculer les métriques
    print("Étape 5 : Calcul des métriques...")
    final_similarity_matrix = np.array(best_similarity_matrices)
    metrics = calculate_metrics(final_similarity_matrix, labels, concepts)
    print("Métriques :")
    print(f"Précision@1 : {metrics['precision@1']:.2f}")
    print(f"Précision@3 : {metrics['precision@3']:.2f}")
    print(f"Distance moyenne (correcte) : {metrics['avg_correct_distance']:.2f}")
    print(f"Distance moyenne (incorrecte) : {metrics['avg_incorrect_distance']:.2f}")

    # Étape 6 : Afficher les résultats
    print("Étape 6 : Affichage des résultats...")
    display_results_with_metrics(best_frame_paths, text_queries, labels, video_ids, final_similarity_matrix, metrics)

    # Nettoyer les frames temporaires
    for frame_path in best_frame_paths:
        os.remove(frame_path)

def display_results_with_metrics(image_paths, texts, labels, video_ids, similarity_matrix, metrics):
    """Affiche les résultats et les métriques dans un fichier HTML avec Bootstrap."""
    import base64
    import numpy as np

    def image_to_data_uri(img_path):
        try:
            with open(img_path, "rb") as img_file:
                encoded = base64.b64encode(img_file.read()).decode('utf-8')
                print(f"Image {img_path} encodée avec succès, longueur de l'URI : {len(encoded)}")
                return encoded
        except Exception as e:
            print(f"Erreur lors de l'encodage de {img_path} : {e}")
            return None

    # Convertir la matrice en numpy pour faciliter le traitement
    similarity_matrix_np = similarity_matrix
    
    # Créer le tableau HTML avec Bootstrap
    # Échapper les accolades dans le CSS en les doublant
    html = """
    <html>
    <head>
        <title>CLIP Results with Kinetics-700 - Metrics</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body {{ padding: 20px; }}
            table {{ width: 100%; }}
            th, td {{ vertical-align: middle; text-align: center; }}
        </style>
    </head>
    <body>
    <div class="container">
        <h1 class="my-4">CLIP Results with Kinetics-700 - Metrics</h1>
        <h2>Métriques</h2>
        <table class="table table-bordered">
            <tr><th>Précision@1</th><td>{:.2f}</td></tr>
            <tr><th>Précision@3</th><td>{:.2f}</td></tr>
            <tr><th>Distance moyenne (correcte)</th><td>{:.2f}</td></tr>
            <tr><th>Distance moyenne (incorrecte)</th><td>{:.2f}</td></tr>
        </table>
        <h2>Résultats détaillés</h2>
        <table class="table table-bordered table-striped">
        <thead class="table-light">
        <tr>
            <th>Video ID</th>
            <th>Frame</th>
            <th>Ground Truth Label</th>
            <th>Best Matched Text</th>
    """.format(metrics['precision@1'], metrics['precision@3'], metrics['avg_correct_distance'], metrics['avg_incorrect_distance'])

    for text in texts:
        html += f'<th>{text}</th>'
    html += "</tr></thead><tbody>"
    
    for i, (img_path, label, video_id) in enumerate(zip(image_paths, labels, video_ids)):
        img_data_uri = image_to_data_uri(img_path)
        if img_data_uri is None:
            img_display = "Image indisponible"
        else:
            img_display = f"<img src='data:image/png;base64,{img_data_uri}' width='100'>"
        
        similarities = similarity_matrix_np[i, :]
        best_match_index = np.argmax(similarities)
        best_matched_text = texts[best_match_index]
        
        html += f"""
        <tr>
            <td>{video_id}</td>
            <td>{img_display}</td>
            <td>{label}</td>
            <td>{best_matched_text}</td>
        """
        for j, similarity_score in enumerate(similarities):
            if j == best_match_index:
                html += f"<td class='bg-info'><b>{similarity_score:.2f}</b></td>"
            else:
                html += f"<td>{similarity_score:.2f}</td>"
        html += "</tr>"
    
    html += """
        </tbody>
        </table>
    </div>
    </body>
    </html>
    """
    
    # Écrire le fichier HTML
    with open("results_metrics.html", "w") as f:
        f.write(html)
    
    print("Résultats et métriques sauvegardés dans results_metrics.html. Ouvrez ce fichier dans un navigateur pour voir les résultats.")

if __name__ == "__main__":
    main()