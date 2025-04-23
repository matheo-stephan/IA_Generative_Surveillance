# **Florence Détection d'Actions dans les Vidéos**   
Ce projet utilise le modèle Florence-2 pour effectuer de la détection d'actions dans des vidéos. Il compare des descriptions textuelles avec des vidéos analysées et retourne la vidéo la plus pertinente en fonction de la correspondance sémantique.

## **Structure du projet** 

- FlorenceAD.py : Script principal de détection d'actions. Ce fichier contient la classe FlorenceActionDetector qui gère l'analyse des vidéos et la comparaison des descriptions textuelles.

- results.json : Fichier contenant les résultats de l'analyse, y compris les vidéos les plus pertinentes selon les critères de détection d'actions.

- video/ : Dossier contenant les vidéos utilisées pour l'analyse.

## **Prérequis** 
Avant de pouvoir exécuter ce projet, vous devez installer les dépendances suivantes :
Copier
```bash
pip install torch transformers sentence-transformers opencv-python tqdm
```
## **Utilisation** 
- Chargement du modèle : Le script charge un modèle Florence pré-entraîné pour effectuer des comparaisons entre les vidéos et les descriptions textuelles.

- Détection d'actions : Le script analyse les vidéos dans le dossier video/ et les compare avec des descriptions fournies sous forme de texte.

- Résultats : Les résultats sont enregistrés dans le fichier results.json, qui contient des informations sur la vidéo la plus pertinente par rapport à la description fournie.

## **Exemple d'exécution :** 
Copier
```bash
python FlorenceAD.py
```
Cela lancera l'analyse et générera les résultats dans le fichier results.json.

## **Résultats de l'analyse des vidéos**

1. Requête : "People walking in the road"
   
     - Vidéo analysée : jump.mp4
       - Score de similarité : 0.0448

     - Vidéo analysée : Tokyo.mp4
       - Score de similarité : 0.2702

     - Vidéo analysée : velo.mp4
       - Score de similarité : 0.0495

    - Vidéo analysée : walk1.mp4
      - Score de similarité : 0.3846

**🎯 Vidéo la plus pertinente : walk1.mp4**

**🎯 Description générée : person**

**🎯 Frame correspondante : 59**

**🎯 Les résultats montrent que la vidéo walk1.mp4 est la plus proche de la requête "people walking in the road" avec un score de similarité de 0.3846.**


2. Requête : "Car on the road"
   
    - Vidéo analysée : jump.mp4
      - Score de similarité : 0.0168

    - Vidéo analysée : Tokyo.mp4
      - Score de similarité : 0.3823
  
    - Vidéo analysée : velo.mp4
      - Score de similarité : 0.2433
  
    - Vidéo analysée : walk1.mp4
      - Score de similarité : 0.2657

**🎯 Vidéo la plus pertinente : Tokyo.mp4**

**🎯 Similarité : 0.3823**

**🎯 Frame correspondante : 200**

**🎯 La vidéo Tokyo.mp4 est la plus pertinente pour la requête "Car on the road" avec un score de 0.3823.**



3. Requête : "A person walking on an urban roadway"

     - Vidéo analysée : jump.mp4
       - Score de similarité : 0.0858

     - Vidéo analysée : Tokyo.mp4
       - Score de similarité : 0.3928

     - Vidéo analysée : velo.mp4
       - Score de similarité : 0.3559

     - Vidéo analysée : walk1.mp4
       - Score de similarité : 0.5672

**🎯 Vidéo la plus pertinente : walk1.mp4**

**🎯 Similarité : 0.5672**

**🎯 Description générée : man walking on street with stroller in background**

**🎯 Frame correspondante : 118**

**🎯 La vidéo walk1.mp4 se distingue par un score de similarité de 0.5672 pour la requête "A person walking on an urban roadway", ce qui en fait le résultat le plus pertinent.**


# **Conclision**   
Le modèle Florence-2 a permis d'identifier avec succès les vidéos les plus pertinentes en fonction des requêtes. Les résultats varient en fonction des scènes présentes dans les vidéos, avec une meilleure correspondance pour des requêtes liées à des personnes en mouvement, notamment la requête "A person walking on an urban roadway", où la vidéo walk1.mp4 a obtenu le meilleur score de similarité.


