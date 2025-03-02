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
#### 3.1 Modèle LSTM :   Ce notebook implémente un modèle de génération de texte en utilisant un réseau de neurones récurrent (RNN) basé sur une architecture LSTM bidirectionnelle.

 Préparation des données
 - Chargement du corpus : Le texte utilisé provient d'un fichier contenant des articles de NIPS.
 - Nettoyage des données : Suppression des mots rares (fréquence < 3) pour éviter le bruit dans l'entraînement.
 - Tokenisation : Conversion des mots en séquences d'entiers à l'aide de Tokenizer de Keras.
 - Création des séquences : Génération de séquences de longueur fixe (200 mots), avec la dernière position utilisée comme cible pour la prédiction.

 Entraînement du modèle
  - Une couche d'embedding (256 dimensions) pour capturer les relations entre les mots.
  - Une couche LSTM bidirectionnelle (128 unités) pour exploiter les dépendances dans les deux directions du texte.
  - Une couche de dropout (0.2) pour éviter l'overfitting.
  - Une couche dense avec activation softmax pour la prédiction du mot suivant.
  - Compilation avec la fonction de perte categorical_crossentropy et l'optimiseur Adam.
  - Callback EarlyStopping pour stopper l'entraînement si la perte ne diminue plus après 5 epochs.
  - Génération des batchs : Une fonction de générateur est utilisée pour alimenter le modèle avec les données en mémoire de manière efficace.
  - Entraînement effectué sur 300 epochs avec un batch size de 32.
 
 Sauvegarde du modèle
  -Modèle enregistré sous model.h5 pour une future réutilisation.
  -Tokenizer sauvegardé (tokenizer.pkl) pour conserver l'indexation des mots.
  -Séquences de mots enregistrées (sequences_words.txt) pour une éventuelle reprise du prétraitement.

#### 3.2 Fine-tuning de GPT-2 (Modèle retenu)

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
