import streamlit as st
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from sklearn.feature_extraction.text import CountVectorizer

query = "Meetup deep learning strasbourg".lower()

st.set_page_config(
    page_title="mini-search",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "Searching into Dailymotion database.  Experimental app. April 2023",
    },
)

@st.cache_data
def load_corpus():
    return pd.read_csv("./dataset.csv", index_col=0)

@st.cache_data
def craft_bm25():
    corpus = load_corpus().content.to_list()
    # BM25
    tokenized_corpus = [doc.lower().split(" ") for doc in corpus]
    return BM25Okapi(tokenized_corpus)

@st.cache_data
def craft_tf():
    corpus = load_corpus().content.to_list()
    tf = CountVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.9)
    tf.fit(corpus)
    return tf

@st.cache_data
def get_corpus_tf():
    corpus = load_corpus().content.to_list()
    tf = craft_tf()
    return tf.transform(corpus)
    

if "corpus" not in st.session_state:
    st.session_state.corpus = load_corpus()
    
bm25 = craft_bm25()
tf = craft_tf()
corpus_tf = get_corpus_tf()

with st.sidebar:
    st.text_input("query", key="query", value="meetup strasbourg deep learning")
    
    
query_tf = tf.transform([st.session_state.query])

df = st.session_state.corpus.copy()
df["score"] = 1
df["snippet"] = df.content.apply(lambda s: s[:250] + "...")
df["query_len"] = len(st.session_state.query)
df["content_bm25"] = np.log1p(bm25.get_scores(st.session_state.query.lower().split(" ")))
df["n_occurence"] = np.array((corpus_tf * query_tf.T).todense()).ravel()
df = df[["score", "snippet", *[c for c in df.columns if c not in  {"snippet", "score"}]]]

features_names = [
    "content_length",
    "query_len",
    "content_bm25",
    "n_occurence",
]

st.markdown("""# Étude de cas avec des documents de la littérature
4 features caractérisent l'adéquation query/documents:
- `content_length`: longueur du contenu 
- `query_len`: longueur de la query
- `content_bm25`: BM25 du contenu x query
- `n_occurence`: occurence des mots de la query dans le doc
Modèle linéaire avec pondérations :""")
cols = st.columns(len(features_names))

for fname, col in zip(features_names, cols):
    with col:
        st.slider(fname, min_value=0.0, max_value=10.0, value=0.0, key=f"w_{fname}")
# update scores


st.header("Search engine results for query: " + st.session_state.query)
ddf = df.sample(2)
df["score"] = np.array([st.session_state[f"w_{fname}"] * df[fname].to_numpy() for fname in features_names]).sum(axis=0)
st.table(df.drop(columns=["content"]).nlargest(10, "score"))

