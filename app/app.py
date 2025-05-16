import streamlit as st
import torch
import numpy as np
import pandas as pd
import faiss
import re
import os
import asyncio
import sys
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

# Ajout du répertoire parent au PYTHONPATH pour l'import des modules classifier
parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)

# Configuration pour éviter le conflit OpenMP
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Assurer qu'un event loop est disponible
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

from classifier.article_dataset import ArticleDataset
from classifier.models.mlp_classifier import MLPClassifier

# Chemins fixes pour le modèle et le dataset
MODEL_PATH = str(Path(__file__).resolve().parent / "mlp_title_fulldb.pth")
DATASET_PATH = str(Path(__file__).resolve().parent / "data" / "articles.csv")

# --- Chargement du modèle pré-entraîné et de ses paramètres ---
@st.cache_resource(show_spinner=False)
def load_model_and_params(checkpoint_path: str, device: str):
    try:
        # Conversion en Path pour une meilleure gestion des chemins
        model_path = Path(checkpoint_path)
        if not model_path.exists():
            st.error(f"Le fichier modèle {model_path} n'existe pas.")
            return None, None, None
        
        with st.spinner("Chargement du modèle..."):
            # Chargement du checkpoint complet
            checkpoint = torch.load(str(model_path), map_location=device)
            
            # Extraction des différentes parties du checkpoint
            model_state = checkpoint.get('model_state_dict')
            hyperparams = checkpoint.get('hyperparameters', {})
            dataset_filters = checkpoint.get('dataset_filters', {"min_freq": 5})
            
            if not model_state:
                st.error("Le fichier de checkpoint ne contient pas d'état du modèle.")
                return None, None, None
            
            # Initialisation du modèle avec les hyperparamètres
            model = MLPClassifier(
                vocab_size=hyperparams.get('vocab_size', 0),
                embedding_dim=hyperparams.get('embedding_dim', 128),
                hidden_dim=hyperparams.get('hidden_dim', 128),
                num_classes=hyperparams.get('num_classes', 0),
                num_hidden_layers=hyperparams.get('num_hidden_layers', 1),
                dropout=hyperparams.get('dropout', 0.3)
            ).to(device)
            
            # Chargement des poids du modèle
            model.load_state_dict(model_state)
            model.eval()
            
            return model, hyperparams, dataset_filters
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle: {str(e)}")
        return None, None, None

# --- Chargement des données et du dataset ---
@st.cache_data(show_spinner=False)
def load_article_dataset(csv_path: str, dataset_filters: dict = None, use_summary: bool = False, classification_level: str = "category") -> ArticleDataset:
    try:
        with st.spinner("Chargement du dataset..."):
            # Si pas de filtres spécifiés, utiliser un filtre par défaut
            if dataset_filters is None:
                dataset_filters = {"min_freq": 5}
            
            # Charge les articles et initialise le dataset
            dataset = ArticleDataset(
                csv_file=csv_path,
                use_summary=use_summary,
                classification_level=classification_level,
                selected_categories=None
            )
            dataset.apply_filters(dataset_filters)
            return dataset
    except Exception as e:
        st.error(f"Erreur lors du chargement du dataset: {str(e)}")
        return None

# --- Construction de l'index FAISS pour la recherche ---
@st.cache_resource
def build_search_index(_model: MLPClassifier, _dataset: ArticleDataset):
    if _model is None or _dataset is None:
        return None
    
    # Extrait la matrice d'embeddings (vocab_size, embed_dim)
    emb_weights = _model.embedding.weight.data.cpu().numpy()
    titles = _dataset.data['title'].fillna('').tolist()
    vectors = []
    for title in titles:
        # Tokenisation comme dans la méthode __getitem__ de ArticleDataset
        tokens = re.findall(r'[a-z0-9]+', title.lower())
        idxs = [_dataset.word_to_index.get(t, 0) for t in tokens]
        if idxs:
            vec = emb_weights[idxs].mean(axis=0)
        else:
            vec = emb_weights[0]
        vectors.append(vec)
    X = np.vstack(vectors)
    faiss.normalize_L2(X)
    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X)
    return index

# --- Prédiction de catégorie ---
def classify_text(text: str, model: MLPClassifier, dataset: ArticleDataset, device: str) -> str:
    if model is None or dataset is None:
        return "Erreur: Modèle ou dataset non chargé"
    
    # Tokenization comme dans ArticleDataset.__getitem__
    tokens = re.findall(r'[a-z0-9]+', text.lower())
    idxs = [dataset.word_to_index.get(t, 0) for t in tokens]
    tensor = torch.tensor(idxs, dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        pred = logits.argmax(dim=1).item()
    return dataset.index_to_class[pred]

# --- Recherche d'articles similaires ---
def search_articles(query: str, model: MLPClassifier, dataset: ArticleDataset, index, top_k: int = 5):
    if model is None or dataset is None or index is None:
        return []
    
    emb_weights = model.embedding.weight.data.cpu().numpy()
    tokens = re.findall(r'[a-z0-9]+', query.lower())
    idxs = [dataset.word_to_index.get(t, 0) for t in tokens]
    if idxs:
        q_vec = emb_weights[idxs].mean(axis=0, keepdims=True) 
    else:
        q_vec = emb_weights[0].reshape(1, -1)
    faiss.normalize_L2(q_vec) 
    distances, indices = index.search(q_vec, top_k)
    return indices[0]

# --- Détails d'un article ---
def clean_text(text):
    if pd.isna(text):
        return "N/A"
    # Remplacer les sauts de ligne par des espaces
    text = text.replace('\n', ' ')
    # Remplacer les multiples espaces par un seul
    text = ' '.join(text.split())
    # Supprimer les caractères de contrôle tout en gardant les caractères spéciaux légitimes
    text = ''.join(char for char in text if ord(char) >= 32 or char in ['\n', '\t'])
    return text

def format_date(date_str):
    if pd.isna(date_str):
        return "N/A"
    try:
        # Convertir la date ISO en objet datetime
        from datetime import datetime
        dt = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ")
        # Formater la date en format français
        return dt.strftime("%d/%m/%Y")
    except:
        return date_str

def show_article_details(dataset: ArticleDataset, idx: int):
    if dataset is None or idx < 0 or idx >= len(dataset.data):
        st.error("Données invalides")
        return
        
    row = dataset.data.iloc[idx]
    
    # Création d'un conteneur pour les détails de l'article
    with st.container():
        # Nettoyage et affichage du titre
        title = clean_text(row['title'])
        st.markdown(f"### {title}")
        
        # Création de colonnes pour les métadonnées
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Catégorie :** {row.get('category', 'N/A')}")
            formatted_date = format_date(row.get('published'))
            st.markdown(f"**Date :** {formatted_date}")
        
        # Affichage des auteurs sur toute la largeur car peuvent être nombreux
        if pd.notna(row.get('authors')):
            authors = clean_text(row.get('authors'))
            st.markdown(f"**Auteurs :** {authors}")
        
        # Affichage du résumé avec un style distinct
        if pd.notna(row.get('summary')):
            with st.expander("Voir le résumé", expanded=True):
                summary = clean_text(row.get('summary'))
                st.markdown(summary)
        
        # Lien vers l'article si disponible
        if 'id' in row and pd.notna(row['id']):
            st.markdown(f"[Voir l'article sur Arxiv]({row['id']})")
        
        # Ajout d'une séparation visuelle entre chaque article
        st.markdown("---")

# --- Interface Streamlit ---
def main():
    st.title("Arxiv-ML : Classification et Recherche d'Articles")
    
    # Chargement du modèle et du dataset en arrière-plan
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    with st.spinner("Chargement du modèle et des données..."):
        model, hyperparams, dataset_filters = load_model_and_params(MODEL_PATH, device)
        dataset = load_article_dataset(DATASET_PATH, dataset_filters)
        
        if model is None or dataset is None:
            st.error("Erreur lors du chargement de l'application.")
            return
            
        index = build_search_index(model, dataset)

    tab1, tab2 = st.tabs(["Recherche", "Classification"])

    
    with tab1:
        st.header("Recherche d'articles similaires")
        st.markdown("""
        Recherchez des articles similaires en entrant un titre ou des mots-clés.
        Le système trouvera les articles dont le contenu est le plus proche sémantiquement.
        """)
        
        # Utilisation d'un formulaire pour gérer la soumission par Entrée
        with st.form(key="search_form"):
            query = st.text_input("Entrez vos mots-clés :")
            num_results = st.slider("Nombre d'articles à afficher", min_value=1, max_value=10, value=5)
            submit_button = st.form_submit_button("Rechercher")
            
            if submit_button or query:  # Réagit au bouton ou à la touche Entrée
                if query.strip():
                    with st.spinner("Recherche en cours..."):
                        hits = search_articles(query, model, dataset, index, top_k=num_results)
                    if len(hits) > 0:
                        st.write(f"Top {len(hits)} résultats :")
                        for idx in hits:
                            show_article_details(dataset, idx)
                    else:
                        st.info("Aucun résultat trouvé.")
                else:
                    st.warning("Veuillez entrer une requête de recherche.")
    with tab2:
        st.header("Classification d'un article")
        st.markdown("""
        Ce modèle a été entraîné pour classifier les **titres** d'articles en catégories principales d'arXiv.
        Entrez le titre d'un article pour découvrir sa catégorie probable.
        """)
        
        text_input = st.text_area("Entrez le titre de l'article :", height=100)
        if st.button("Classifier", key="btn_classify"):
            if text_input.strip():
                with st.spinner("Classification en cours..."):
                    category = classify_text(text_input, model, dataset, device)
                st.success(f"Catégorie prédite : **{category}**")
            else:
                st.warning("Veuillez entrer un titre à classifier.")

if __name__ == '__main__':
    main()
