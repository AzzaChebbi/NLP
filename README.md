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
  - Modèle enregistré sous model.h5 pour une future réutilisation.
  - Tokenizer sauvegardé (tokenizer.pkl) pour conserver l'indexation des mots.
  - Séquences de mots enregistrées (sequences_words.txt) pour une éventuelle reprise du prétraitement.

#### 3.2 Fine-tuning de GPT-2 (Modèle retenu) : Le notebook `gpt2_train.ipynb` adapte le modèle pré-entraîné GPT-2 à notre corpus scientifique.

   - Adaptation du modèle au domaine scientifique  
   - Perte plus faible comparée à celle du modèle entraîné from scratch  
   - Meilleure capacité de généralisation  

Ce notebook implémente un fine-tuning du modèle GPT-2 (124M) sur un corpus spécifique afin de générer du texte adapté aux données fournies.

  - Installation et Préparation de l’Environnement
  - Installation de la bibliothèque gpt-2-simple pour faciliter l’utilisation de GPT-2.
  - Importation des bibliothèques nécessaires (gpt_2_simple, tensorflow).
  - Téléchargement du modèle pré-entraîné GPT-2 (version 124M).
  - Montage de Google Drive pour stocker et récupérer les fichiers d’entraînement.
  - Copie du fichier de texte nips_clean.txt depuis Google Drive vers l’environnement Colab.

Entraînement du modèle sur un corpus spécifique
  - Démarrage d’une session TensorFlow pour gérer l’entraînement du modèle.
  - Fine-tuning initial :
    - Chargement du corpus.
    - Entraînement du modèle avec steps=1000.
    - Sauvegarde automatique du modèle toutes les 500 itérations.
    - Génération d’échantillons de texte toutes les 200 itérations pour évaluer la progression.
  - Continuation du fine-tuning :
    - Réinitialisation de la session TensorFlow et chargement du dernier checkpoint.
    - Extension de l’entraînement à 2000 steps supplémentaires.
    - Nouvelle reprise avec 2000 steps supplémentaires pour améliorer la qualité de la génération.

Sauvegarde et Chargement du Modèle Entraîné
 - Sauvegarde des checkpoints sur Google Drive pour éviter toute perte de progrès.
 - Récupération des checkpoints en extrayant les fichiers du modèle depuis Google Drive vers l’environnement Colab.
 - Vérification du contenu du dossier run1 contenant le modèle entraîné.Génération de texte avec le modèle fine-tuné

Chargement du modèle entraîné (run1).
 - Génération de texte en utilisant le modèle ajusté sur le corpus spécifique.
## Génération de Texte
Le notebook gpt2_generate.ipynb utilise le modèle GPT-2 fine-tuné pour générer de nouveaux textes scientifiques.

Après avoir évalué les 2 modèles, nous avons opté pour le modèle GPT-2 fine-tuné en raison de sa performance optimale, notamment grâce à la minimisation de la fonction de perte (loss)
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

Voilà des captures de notre application : 

![Result_NLP1](https://github.com/user-attachments/assets/0dfcd9a7-56b7-4f73-ad44-b737988b00ab)
