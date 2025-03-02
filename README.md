# NLP
## Table des matières
- [Collecte de Données](#collecte_de_données)
- [Prétraitement des Données](#prétraitement_des_données)
- [Modèles de Génération de Texte](#modèles_de_génération_de_texte)
- [Génération de Texte](#génération_de_texte)
- [Clustering Thématique](#clustering_thématique)
- [Interface Utilisateur](#interface_utilisateur)

## Collecte de Données
Le script web_scraping.py extrait des textes scientifiques et les sauvegarde au format CSV.
```bash
python web_scraping.py
```
Output: nips.csv
## Prétraitement des Données
Le notebook text_preprocessing.ipynb nettoie et prépare les données textuelles pour l'entraînement.

Opérations principales:

 📂 Importation et Exploration des Données
  
   - Chargement du dataset nips.csv depuis Google Drive.
   - Suppression des abstracts manquants.
   - Exploration des données : nombre d'entrées, types de colonnes et aperçu du texte.
  
 🔍 Prétraitement du Texte

   - Suppression des caractères non-ASCII et du bruit (URLs, citations, ponctuation).
   - Normalisation du texte : conversion en minuscules, remplacement des nombres par "NUMBER".
   - Standardisation des termes en anglais américain.
   - Tokenisation et analyse du vocabulaire (nombre total et unique de mots).

Output: nips_clean.txt


## Modèles de Génération de Texte
* 3.1 Modèle from Scratch :

Le notebook text_generation_word_train_notebook.ipynb implémente et entraîne un modèle de génération de texte à partir de zéro.

* 3.2 Fine-tuning de GPT-2 (Modèle retenu)

  Le notebook `gpt2_train.ipynb` adapte le modèle pré-entraîné GPT-2 à notre corpus scientifique.

   - Adaptation du modèle au domaine scientifique  
   - Perte plus faible comparée à celle du modèle entraîné from scratch  
   - Meilleure capacité de généralisation  

## Génération de Texte
Le notebook gpt2_generate.ipynb utilise le modèle GPT-2 fine-tuné pour générer de nouveaux textes scientifiques.

## Clustering Thématique
Le notebook de clustering applique l'algorithme K-means pour regrouper les textes scientifiques par thématique ou domaine.

Étapes:

  - Vectorisation des textes (TF-IDF)
  - Classification des documents
  - Visualisation des clusters

## Interface Utilisateur
L'application Streamlit (app.py) intègre toutes les fonctionnalités dans une interface conviviale.

Pour lancer l'application:

```bash
streamlit run app.py
```
