import streamlit as st 
import tensorflow.compat.v1 as tf
import gpt_2_simple as gpt2
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from streamlit_extras.add_vertical_space import add_vertical_space


tf.disable_v2_behavior()

# Custom Streamlit Page Config
st.set_page_config(
    page_title="AI-Powered Text Generation and Classification",
    page_icon="📝",
    layout="wide"
)

# Fonction pour charger le modèle fine tuné GPT-2 
def load_model():
    tf.reset_default_graph()
    sess = gpt2.start_tf_sess()
    gpt2.load_gpt2(sess, run_name="run1", checkpoint_dir="./")
    return sess

# Fonction pour générer le texte 
def generate_text(sess, user_input):
    generated_texts = gpt2.generate(
        sess,
        length=200,
        temperature=0.5,
        prefix=user_input,
        nsamples=1,
        batch_size=1,
        checkpoint_dir="./",
        return_as_list=True
    )
    return generated_texts[0]

# Fonction pour charger et faire le prétraitement de texte 
def load_texts_from_file(file):
    """Charge les textes depuis un fichier .txt et les renvoie sous forme de liste."""
    texts = file.readlines()
    texts = [text.strip() for text in texts if text.strip()]
    return texts

def preprocess_text(text):
    """Nettoie le texte en supprimant les caractères spéciaux et en le mettant en minuscule."""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

def perform_clustering(texts, n_clusters=5):
    """Applique le clustering K-Means sur les textes."""
    vectorizer = TfidfVectorizer(stop_words='english')
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    pipeline = make_pipeline(vectorizer, kmeans)
    pipeline.fit(texts)
    labels = kmeans.labels_
    return pipeline, labels, kmeans, vectorizer

def classify_new_text(pipeline, new_text):
    """Classe un nouveau texte dans l'un des clusters existants."""
    cluster = pipeline.predict([new_text])[0]
    return cluster

def display_wordclouds(labels, texts):
    cluster_texts = {}
    for i, label in enumerate(labels):
        if label not in cluster_texts:
            cluster_texts[label] = []
        cluster_texts[label].append(texts[i])

    if not cluster_texts:
        st.write("Aucun texte dans les clusters.")
        return

    n_clusters = len(cluster_texts)
    n_cols = 2  
    n_rows = (n_clusters + n_cols - 1) // n_cols 

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 8))
    axes = axes.flatten()

    for i, (cluster, cluster_text) in enumerate(cluster_texts.items()):
        text = ' '.join(t.decode('utf-8') if isinstance(t, bytes) else t for t in cluster_text)
        wc = WordCloud(width=800, height=400, max_words=100, background_color='white').generate(text)

        axes[i].imshow(wc, interpolation='bilinear')
        axes[i].axis('off')
        axes[i].set_title(f'Cluster {cluster}')

    plt.tight_layout()
    st.pyplot(fig)



# Streamlit UI nouveau thème
st.markdown("""
    <style>
        body {
            background-color: #00172B;
            color: #FFF;
            font-family: sans-serif;
        }
        .big-font {
            font-size: 30px !important;
            font-weight: bold;
            color: #E694FF;
        }
        .subheader {
            font-size: 22px;
            color: #E694FF;
        }
        .stButton > button {
            background-color: #0083B8;
            color: white;
            font-size: 18px;
            border-radius: 8px;
        }
        .stTextInput, .stTextArea {
            background-color: #00172B;
            color: white;
            border: 1px solid #0083B8;
        }
        .stTextInput input, .stTextArea textarea {
            color: #FFF;
        }
    </style>
""", unsafe_allow_html=True)

# Titre
st.markdown('<p class="big-font">📝 AI-Powered Text Generation & Clustering</p>', unsafe_allow_html=True)
st.write("Enhance your text processing experience with GPT-2 and K-Means clustering.")
add_vertical_space(2)

# Tabs pour une meilleure organisation
tabs = st.tabs(["GPT-2 Text Generation", "Text Clustering"])

# Séction de GPT-2 Text Generation 
with tabs[0]:
    st.subheader("📜 Generate Text with GPT-2")
    user_input = st.text_area("💬 Enter a prompt:", "Deep learning has transformed the field of...")
    if st.button("✨ Generate Text"):
        with st.spinner("🔄 Generating text..."):
            try:
                sess = load_model()
                generated_text = generate_text(sess, user_input)
                st.success("✔️ Text generated successfully!")
                st.write(generated_text)
            except Exception as e:
                st.error(f"❌ Error: {e}")

# Séction de texte clustering
with tabs[1]:
    st.subheader("📊 Text Clustering with K-Means")
    uploaded_file = st.file_uploader("Téléchargez un fichier texte", type=["txt"])

    if uploaded_file is not None:
        texts = load_texts_from_file(uploaded_file)
        st.success("Fichier chargé avec succès!")

        n_clusters = st.slider("Nombre de clusters", 2, 10, 5)
        
        # clustering
        if st.button("Exécuter le clustering"):
            # sauvegarde de la session 
            if 'pipeline' not in st.session_state:
                pipeline, labels, kmeans, vectorizer = perform_clustering(texts, n_clusters)
                st.session_state.pipeline = pipeline
                st.session_state.labels = labels
                st.session_state.kmeans = kmeans
                st.session_state.texts = texts
                st.session_state.vectorizer = vectorizer
                st.write("Clustering terminé!")
                display_wordclouds(labels, texts)  
            else:
                st.write("Clustering déjà effectué!")

        # Classification 
        st.subheader("Classification d'un nouveau texte")
        new_text = st.text_area("Entrez un texte à classifier")

        if st.button("Classifier"):
            if 'pipeline' in st.session_state:
                if new_text.strip():
                    cluster = classify_new_text(st.session_state.pipeline, new_text)
                    st.success(f"Le texte appartient au cluster {cluster}.")
                else:
                    st.warning("Veuillez entrer un texte valide.")
            else:
                st.warning("Veuillez exécuter d'abord le clustering.")
        
