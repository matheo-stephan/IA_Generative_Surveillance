## Résultats CLIP avec Kinetics-700
# Aperçu
Ce projet utilise le modèle CLIP (Contrastive Language–Image Pretraining) pour analyser un sous-ensemble de la base de données Kinetics-700, une vaste collection de vidéos d'actions humaines. L'objectif est d'extraire des frames représentatives des vidéos, de calculer leurs embeddings avec CLIP, de les stocker dans une base de données vectorielle ChromaDB, et d'effectuer des recherches de similarité texte-image. Le projet évalue la performance de CLIP pour associer des frames vidéo à des descriptions textuelles, en utilisant des métriques telles que la précision@1, la précision@3 et les similarités cosinus moyennes.

Le projet inclut également une fonctionnalité de recherche interactive, permettant aux utilisateurs de saisir des requêtes textuelles et de récupérer les frames les plus similaires, avec des résultats visualisés dans des fichiers HTML. De plus, un histogramme de distribution des similarités aide à analyser la capacité discriminative du modèle.

# Base de données Kinetics-700
Kinetics-700 est une base de données à grande échelle de vidéos d'actions humaines, contenant plus de 650 000 clips vidéo répartis sur 700 catégories d'actions. Pour ce projet, nous utilisons un sous-ensemble de 50 classes d'actions, avec 5 vidéos par classe, soit un total de 250 vidéos. En raison de problèmes de téléchargement, 218 vidéos ont été traitées avec succès.

Source : Le sous-ensemble est défini dans annotations/selected_subset.json, qui liste les 50 classes d'actions (par exemple, "playing basketball", "using bagging machine").
Annotations : Les métadonnées des vidéos (URL, étiquette, etc.) sont stockées dans annotations.json.
Vidéos : Les vidéos sont stockées dans le répertoire videos/ sous les noms video_1.mp4 à video_218.mp4.
Classes d'actions utilisées
Le sous-ensemble comprend 50 actions variées, telles que :

using bagging machine
playing basketball
juggling fire
snowkiting
getting a haircut
(Voir la liste complète dans la table des métriques ci-dessous ou dans selected_subset.json).

## Structure du projet
Le projet est organisé selon la structure de répertoire suivante :

```
C:/Users/SANGLI Kenneth/Documents/project_root/
│
├── videos/                     # Répertoire contenant les fichiers vidéo téléchargés (video_1.mp4 à video_218.mp4)
├── annotations/                # Répertoire pour les fichiers d'annotations
│   ├── selected_subset.json    # Fichier JSON définissant le sous-ensemble de 50 classes et leurs vidéos
│   └── annotations.json        # Fichier JSON avec les métadonnées de chaque vidéo (chemin, étiquette, etc.)
├── output/                     # Répertoire pour les fichiers de sortie (frames, résultats HTML, graphiques)
│   ├── frame_video_1.png       # Frames extraites (une par vidéo)
│   ├── results_kinetics.html   # Fichier HTML avec les résultats globaux et les métriques
│   ├── similarity_distribution.png  # Graphique de distribution des similarités
│   └── search_results_*.html   # Fichiers HTML pour les résultats des requêtes de recherche
├── chroma_db/                  # Répertoire pour la base de données vectorielle ChromaDB
└── V2_kinetics.ipynb   # Script principal du projet
```

## Fonctionnement
## 1. Préparation des données
Téléchargement des vidéos : Les vidéos sont téléchargées depuis Kinetics-700 en fonction de selected_subset.json et stockées dans videos/.
Extraction des frames : Pour chaque vidéo, une frame représentative (la frame centrale) est extraite avec OpenCV et sauvegardée dans output/ sous le nom frame_video_X.png.
## 2. Analyse avec CLIP
Modèle : Nous utilisons le modèle CLIP (ViT-B/32) d'OpenAI, qui génère des embeddings pour les images et les textes.
Embeddings des frames : Chaque frame extraite est encodée en un embedding de 512 dimensions avec l'encodeur d'images de CLIP. Les embeddings sont normalisés pour garantir que la similarité cosinus reste dans l'intervalle [0, 1].
Embeddings des textes : Pour chaque classe d'action, une requête textuelle descriptive est générée (par exemple, "Une personne effectuant l'action de using bagging machine dans un cadre typique"). Ces requêtes sont encodées en embeddings avec l'encodeur de texte de CLIP et normalisées.
Stockage : Les embeddings des frames sont stockés dans une base de données vectorielle ChromaDB (chroma_db/) dans une collection nommée kinetics_700_subset, accompagnés de métadonnées (chemin de la frame, étiquette, ID de la vidéo).
## 3. Calcul des similarités
Matrice de similarité : Une matrice de similarité est calculée entre tous les embeddings des frames et des textes en utilisant la similarité cosinus. Cette matrice est utilisée pour évaluer la capacité de CLIP à associer les frames aux descriptions d'actions correspondantes.
Métriques :
Précision@1 : Pourcentage de frames où la requête textuelle la mieux notée correspond à l'étiquette réelle.
Précision@3 : Pourcentage de frames où l'étiquette réelle est parmi les 3 requêtes textuelles les mieux notées.
Distance moyenne (correcte) : Similarité cosinus moyenne entre une frame et sa requête textuelle correcte.
Distance moyenne (incorrecte) : Similarité cosinus moyenne entre une frame et les requêtes textuelles incorrectes.
Distribution des similarités : Un histogramme est généré pour visualiser la distribution des similarités cosinus pour les paires correctes (frame et texte correct) et incorrectes (frame et texte incorrect).
## 4. Recherche par requête textuelle
Recherche interactive : Les utilisateurs peuvent entrer une requête textuelle (par exemple, "people playing outdoor"). La requête est encodée en un embedding avec CLIP, et ChromaDB récupère les frames les plus similaires en fonction de la similarité cosinus.
Recalcul manuel des similarités : Pour garantir des similarités précises, le script récupère les embeddings depuis ChromaDB et recalcule les similarités cosinus manuellement, en s'assurant que les valeurs restent dans l'intervalle [0, 1].
Affichage des résultats : Les résultats sont affichés dans un fichier HTML (search_results_*.html), montrant l'ID de la vidéo, la frame, l'étiquette et le score de similarité pour chaque correspondance.


## Métriques obtenues
Les métriques suivantes ont été obtenues après l'analyse des 218 vidéos :

Précision@1 : 0.43
43 % des frames ont été correctement associées à leur étiquette réelle en première position.
Précision@3 : 0.53
53 % des frames avaient leur étiquette réelle parmi les 3 premières positions.
Distance moyenne (correcte) : 0.25
Similarité cosinus moyenne entre les frames et leurs requêtes textuelles correctes.
Distance moyenne (incorrecte) : 0.20
Similarité cosinus moyenne entre les frames et les requêtes textuelles incorrectes.


## Distribution des similarités
L'histogramme de l'image similarity_distribution.png visualise la distribution des similarités cosinus pour les paires correctes et incorrectes :

Paires correctes (vert) : Représentent les similarités cosinus entre les frames et leurs requêtes textuelles correspondantes.
Paires incorrectes (rouge) : Représentent les similarités cosinus entre les frames et les requêtes textuelles incorrectes.
Analyse :
Les deux distributions sont fortement concentrées vers des valeurs de similarité faibles (0.05 à 0.35), avec un pic autour de 0.20–0.25.
Il y a un chevauchement important entre les deux distributions, ce qui indique que CLIP a du mal à distinguer les paires correctes des incorrectes. Cela est cohérent avec l'écart réduit entre la distance moyenne correcte (0.25) et incorrecte (0.20).
Ce chevauchement suggère que les embeddings ne sont pas très discriminants pour cette tâche, probablement en raison de requêtes textuelles génériques ou du choix d'une seule frame par vidéo.
Résultats détaillés (échantillon)
Le tableau ci-dessous montre un échantillon des résultats détaillés pour les premières vidéos :

ID Vidéo	Frame	Étiquette réelle	Meilleure correspondance textuelle	... (Scores de similarité pour chaque requête textuelle)
video_1	[Img]	using bagging machine	Une personne effectuant l'action de using bagging machine...	0.31 (meilleur), 0.28, 0.26, ...
video_2	[Img]	using bagging machine	Une personne effectuant l'action de using bagging machine...	0.31 (meilleur), 0.28, 0.26, ...
video_3	[Img]	using bagging machine	Une personne effectuant l'action de using bagging machine...	0.27 (meilleur), 0.26, 0.24, ...
video_4	[Img]	using bagging machine	Une personne effectuant l'action de using bagging machine...	0.26 (meilleur), 0.24, 0.22, ...
Liste complète des classes d'actions :

applauding, bathing dog, beatboxing, being in zero gravity, bench pressing, bobsledding, carving ice, celebrating, checking tires, checking watch, chiseling wood, clay pottery making, closing door, country line dancing, cutting nails, diving cliff, doing jigsaw puzzle, filling cake, getting a haircut, giving or receiving award, golf chipping, gospel singing in church, headbutting, hoverboarding, juggling fire, kitesurfing, land sailing, long jump, opening bottle (not wine), opening refrigerator, parasailing, playing clarinet, playing cricket, playing scrabble, playing squash or racquetball, punching person (boxing), putting on sari, sharpening pencil, skipping stone, smoking, snowkiting, springboard diving, sword swallowing, tackling, testifying, threading needle, using bagging machine, wading through water, wood burning (art).

## Exemple de recherche par requête
Pour la requête "people playing outdoor", les 5 meilleurs résultats étaient :

ID Vidéo	Frame	Étiquette	Similarité
video_168	[Img]	juggling fire	0.2741
video_167	[Img]	threading needle	0.2712
video_54	[Img]	tackling	0.2663
video_153	[Img]	getting a haircut	0.2627
video_185	[Img]	snowkiting	0.2573
Note sur les valeurs de similarité :

Les exécutions précédentes du script ont donné des scores de similarité négatifs (par exemple, -0.4519), ce qui était incorrect pour la similarité cosinus. Ce problème a été corrigé en :
Assurant une normalisation correcte des embeddings avant leur stockage dans ChromaDB et lors des requêtes.
Recalculant les similarités manuellement avec cosine_similarity au lieu de se fier à la conversion de la métrique de distance de ChromaDB.
Limitant les valeurs de similarité à l'intervalle [0, 1] pour gérer les erreurs numériques.
Les similarités actuelles (par exemple, 0.2741) sont maintenant positives et dans l'intervalle attendu, confirmant la correction.


## Aperçu du code
Script principal : V2_kinetics.ipynb
Ce script orchestre l'ensemble du flux de travail :

Extraction des frames : Extrait la frame centrale de chaque vidéo avec OpenCV.
Génération des embeddings : Utilise CLIP (ViT-B/32) pour encoder les frames et les requêtes textuelles en embeddings, avec une normalisation explicite.
Stockage dans ChromaDB : Stocke les embeddings des frames dans une collection ChromaDB.
Calcul des similarités : Calcule une matrice de similarité entre les frames et les requêtes textuelles.
Calcul des métriques : Calcule la précision@1, la précision@3 et les distances moyennes.
Visualisation : Génère des fichiers HTML (results_kinetics.html, search_results_*.html) et un histogramme de distribution des similarités (similarity_distribution.png).
Recherche interactive : Permet aux utilisateurs de saisir des requêtes textuelles et de récupérer les frames les plus similaires.
Fonctions clés
extract_single_frame(video_path) : Extrait la frame centrale d'une vidéo.
encode_image(image_path) : Encode une image en un embedding CLIP avec normalisation.
encode_text(text) : Encode une requête textuelle en un embedding CLIP avec normalisation.
calculate_metrics(similarity_matrix, labels, text_labels) : Calcule les métriques d'évaluation.
generate_results_html(...) : Génère le fichier HTML des résultats principaux.
plot_similarity_distribution(correct_distances, incorrect_distances) : Génère l'histogramme de distribution des similarités.
search_similar_images(query, top_k) : Recherche les frames les plus similaires à une requête textuelle.
display_search_results(...) : Génère un fichier HTML pour les résultats des requêtes de recherche.
Dépendances
Le projet repose sur les bibliothèques Python suivantes :

numpy : Pour les opérations numériques.
torch : Pour exécuter le modèle CLIP.
opencv-python-headless : Pour l'extraction des frames vidéo.
pillow : Pour le traitement des images.
chromadb : Pour le stockage et la recherche dans la base de données vectorielle.
clip (de openai-clip) : Pour le modèle CLIP et le prétraitement.
sklearn : Pour le calcul de la similarité cosinus.
matplotlib : Pour tracer la distribution des similarités.
Installez-les avec :

```bash
pip install numpy torch opencv-python-headless pillow chromadb clip sklearn matplotlib
```

## Comment exécuter
## 1. Préparer l'environnement
Assurez-vous que Python 3.11 est installé et installez les dépendances :


pip install numpy torch opencv-python-headless pillow chromadb clip sklearn matplotlib
## 2. Configurer le répertoire
Placez le script kinetics_clip_analysis.py dans le répertoire racine du projet :


C:/Users/SANGLI Kenneth/Documents/project_root/
Assurez-vous que les éléments suivants sont présents :

videos/ avec les fichiers vidéo (video_1.mp4 à video_218.mp4).
annotations/selected_subset.json et annotations.json.
## 3. Exécuter le script
Lancez le script :

```bash
cd C:\Users\SANGLI Kenneth\Documents\project_root
python kinetics_clip_analysis.py
```
## 4. Flux d'exécution du script
Étapes 1-7 : Le script traite les vidéos, extrait les frames, calcule les embeddings, les stocke dans ChromaDB, calcule les métriques et génère les visualisations.
Étape 8 : Passe en mode de recherche interactive :
Entrez une requête textuelle (par exemple, "people playing outdoor").
Spécifiez le nombre de résultats (par défaut 5).
Les résultats sont sauvegardés dans output/search_results_<requête>.html.
Tapez "quitter" pour sortir.
## 5. Voir les résultats
Ouvrez output/results_kinetics.html dans un navigateur pour voir les résultats globaux et les métriques.
Ouvrez output/similarity_distribution.png pour voir l'histogramme de distribution des similarités.
Ouvrez output/search_results_<requête>.html pour voir les résultats de vos requêtes textuelles.

## Conclusion
Ce projet montre l'application de CLIP pour associer des frames vidéo de Kinetics-700 à des descriptions textuelles, en mettant l'accent sur l'évaluation des similarités et la recherche textuelle. Les résultats indiquent une performance modérée (Précision@1 : 0.43, Précision@3 : 0.53), avec des scores de similarité faibles et un chevauchement important dans la distribution des similarités, ce qui laisse place à des améliorations. La fonctionnalité de recherche interactive permet aux utilisateurs d'explorer la base de données de manière dynamique, en faisant un outil utile pour les tâches de reconnaissance d'actions.

La correction des similarités négatives garantit que les résultats sont désormais dans l'intervalle attendu, mais la performance globale suggère que des requêtes plus spécifiques, une meilleure sélection des frames ou un modèle plus performant pourraient améliorer les résultats. Pour toute assistance supplémentaire ou pour contribuer, veuillez contacter l'auteur du projet.
