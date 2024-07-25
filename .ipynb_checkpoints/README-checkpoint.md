# projet-pompiers
# Architecture du Projet

## Introduction
Ce projet vise à prédire le temps de réponse des pompiers dans la ville de Londres.

## Architecture du Projet



### Composantes

- **Source des Données**
  - Données extraites de fichiers CSV situés dans le répertoire `data/`.
  - Les données sont chargées et nettoyées à l'aide des scripts dans `scripts/data_preprocessing.py`.

- **Prétraitement**
  - Nettoyage des données, gestion des valeurs manquantes et normalisation.
  - Les transformations sont réalisées dans `scripts/preprocessing.py`.

- **Analyse et Modélisation**
  - Utilisation de la régression linéaire et de modèles d'ensemble.
  - Les modèles sont définis et entraînés dans `notebooks/model_training.ipynb`.

- **Évaluation**
  - Évaluation des modèles à l'aide de la validation croisée et des métriques de performance (MSE, RMSE).
  - Les résultats sont analysés dans `notebooks/model_evaluation.ipynb`.

- **Visualisation et Reporting**
  - Visualisation des résultats à l'aide de `matplotlib` et `seaborn`.
  - Les graphiques sont générés dans `notebooks/visualization.ipynb`.

## Installation et Exécution

1. Clonez le dépôt :
   ```bash
   git clone https://github.com/username/project.git
