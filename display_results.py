import os
import base64
import numpy as np

def image_to_data_uri(img_path):
    """Converts an image to a data URI for embedding directly in HTML."""
    try:
        with open(img_path, "rb") as img_file:
            encoded = base64.b64encode(img_file.read()).decode('utf-8')
            print(f"Image {img_path} encodée avec succès, longueur de l'URI : {len(encoded)}")
            return encoded
    except Exception as e:
        print(f"Erreur lors de l'encodage de {img_path} : {e}")
        return None

def display_similarity(image_paths, texts, labels, video_ids, similarity_matrix):
    """Affiche les similarités dans un fichier HTML avec Bootstrap."""
    # Convertir la matrice en numpy pour faciliter le traitement
    similarity_matrix_np = similarity_matrix
    
    # Créer le tableau HTML avec Bootstrap
    html = """
    <html>
    <head>
        <title>CLIP Results with Kinetics-700</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body { padding: 20px; }
            table { width: 100%; }
            th, td { vertical-align: middle; text-align: center; }
        </style>
    </head>
    <body>
    <div class="container">
        <h1 class="my-4">CLIP Results with Kinetics-700</h1>
        <table class="table table-bordered table-striped">
        <thead class="table-light">
        <tr>
            <th>Video ID</th>
            <th>Frame</th>
            <th>Ground Truth Label</th>
            <th>Best Matched Text</th>
    """
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
    with open("results.html", "w") as f:
        f.write(html)
    
    print("Résultats sauvegardés dans results.html. Ouvrez ce fichier dans un navigateur pour voir les résultats.")