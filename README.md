# 🧪 Branche de Test : Recherche sur l'Encodage et la Similarité Image/Texte

## 🎯 Objectif

Cette branche a été créée dans le but de mener des recherches exploratoires et des tests pratiques autour des thématiques suivantes :
- Encodage d’images et de textes en vecteurs,
- Stockage de ces vecteurs dans une base vectorielle (`ChromaDB`),
- Évaluation de la similarité entre images et textes via la distance cosinus,
- Filtrage hybride utilisant les métadonnées pour enrichir les recherches.

---

## 🔍 Détails techniques

Utilisation de **ChromaDB** pour créer une base locale persistante.

Ajout de documents avec :
- Des **embeddings** (vecteurs issus de modèles d'encodage),
- Des **métadonnées** (ex : catégorie de l’image).

Génération de données de test simulant de vrais embeddings d’images.
Comparaison de similarité avec un **filtrage par catégorie** et un **seuil de distance cosinus** personnalisable.

---

## 🚧 Statut

Cette branche est expérimentale :

Elle sert à poser les bases de la logique de recherche par vecteurs et tester des cas concrets avant d'intégrer une solution plus robuste ou complète dans la branche principale.
