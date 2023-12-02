import streamlit as st
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from scipy.stats import rankdata


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
st.markdown(
    """
    <style>
    .katex-html {
        text-align: left;
    }
    </style>""",
    unsafe_allow_html=True,
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


@st.cache_data
def prepare_df():
    df = load_corpus()
    df["rel"] = [0, 2, 2, 4, 0, 1] + [0] * (len(df) - 6)
    df["snippet"] = df.content.apply(lambda s: s[:250] + "...")
    df["hash"] = df.content.apply(hash)
    df = df.reset_index().rename(columns={"index": "id"})
    return df


if "step" not in st.session_state:
    st.session_state.step = "docs"

if "cutoff" not in st.session_state:
    st.session_state.cutoff = 1

if "query" not in st.session_state:
    st.session_state.query = "meetup strasbourg deep learning"

bm25 = craft_bm25()
tf = craft_tf()
corpus_tf = get_corpus_tf()


def highlight_greaterthan(s, threshold, column):
    is_max = pd.Series(data=False, index=s.index)
    is_max[column] = s.loc[column] >= threshold
    return ["background-color: #0099cc" if is_max.any() else "" for v in is_max]


query_tf = tf.transform([st.session_state.query])

# prepare df
df = prepare_df()

COL_OPTIMAL_ORDER = [
    "relevance",
    "score",
    "DG",
    "snippet",
    "query_len",
    "content_bm25",
    "n_occurence",
    "content_length",
]

features_names = [
    "content_length",
    "query_len",
    "content_bm25",
    "n_occurence",
]

# hyperparams of display
ordering_col = "hash"
highlight_column = "content_length"
highlight_threshold = 1000
ascending = False


DEMO_STEPS = ["docs", "features", "score", "relevance", "ndcg", "ndcg-values", "ltr"]

# display
cols = st.columns([3, 5, 1])
with cols[0]:
    st.markdown(f"### Démo moteur de recherche")
with cols[1]:
    st.radio(label="version", options=DEMO_STEPS, horizontal=True, key="step")
with cols[2]:
    st.radio(label="display", options=["wide", "long"], horizontal=True, key="display")

if st.session_state.display == "wide":
    col_features = st.columns([130, 75, 155, 1])
else:
    col_features = st.columns([15, 10, 1])

with col_features[-1]:
    for _ in range(32):
        st.markdown("")
with col_features[0]:
    # version features
    if DEMO_STEPS.index(st.session_state.step) >= DEMO_STEPS.index("features"):
        df["query_len"] = len(st.session_state.query)
        df["content_bm25"] = np.log1p(
            bm25.get_scores(st.session_state.query.lower().split(" "))
        )
        df["n_occurence"] = np.array((corpus_tf * query_tf.T).todense()).ravel()
        st.markdown(
            """
            ##### 4 features caractérisent l'adéquation query/documents:
        - `content_length`: longueur du contenu 
        - `query_len`: longueur de la query
        - `content_bm25`: BM25 du contenu x query
        - `n_occurence`: occurence des mots de la query dans le doc"""
        )
        highlight_column = "content_bm25"
        highlight_threshold = 2.1
        ordering_col = "content_bm25"

with col_features[1]:
    # AJOUT D'UN SCORE LINEAIRE
    if DEMO_STEPS.index(st.session_state.step) >= DEMO_STEPS.index("score"):
        st.markdown("##### Modèle linéaire avec pondérations :")
        for fname in features_names:
            st.slider(fname, min_value=0.0, max_value=10.0, value=0.0, key=f"w_{fname}")
        df["score"] = np.array(
            [
                st.session_state[f"w_{fname}"] * df[fname].to_numpy()
                for fname in features_names
            ]
        ).sum(axis=0)
        df["rang"] = rankdata(-df.score, method="ordinal")
        ordering_col = "score"
        ascending = False
        highlight_column = "score"
        highlight_threshold = df.iloc[1].score


if DEMO_STEPS.index(st.session_state.step) >= DEMO_STEPS.index("relevance"):
    df["relevance"] = df.rel
    ordering_col = "score"
    highlight_column = "relevance"
    highlight_threshold = 1

if DEMO_STEPS.index(st.session_state.step) >= DEMO_STEPS.index("ndcg"):
    with col_features[0]:
        st.markdown("##### Mesure de la qualité d'un ranking")
        # st.markdown("Cutoff k=5")
        col_params = [3, 5]
        cols = st.columns(col_params)
        with cols[0]:
            st.write("")
            st.write("Gain d'un document")
        with cols[1]:
            st.latex(r"DG_i = \frac{r_i}{\log(i+1)}")
        cols = st.columns(col_params)
        with cols[0]:
            st.write("")
            st.write("")
            st.write("Gain cumulé")
        with cols[1]:
            st.latex(r"DCG_k = \sum_{i=1}^k DG_i = \sum_{i=1}^k \frac{r_i}{\log(i+1)}")
        cols = st.columns(col_params)
        with cols[0]:
            st.write("")
            st.write("Normalisation par DCG optimal")
        with cols[1]:
            st.latex(r"nDCG_k = DCG_k / optimalDCG_k")

        df["DG"] = df["relevance"] / np.log1p(df["rang"])

if DEMO_STEPS.index(st.session_state.step) >= DEMO_STEPS.index("ndcg-values"):
    with col_features[0]:
        st.radio(
            "Choix de la valeur du seuil k",
            options=[1, 3, 5, 10],
            index=1,
            key="cutoff",
            horizontal=True,
            label_visibility="visible",
        )
        k = st.session_state.cutoff
        dcgs = df.sort_values("score", ascending=False).DG.iloc[:k]
        dcg = dcgs.sum()
        idcgs = (
            df[df.relevance > 0]
            .sort_values("relevance", ascending=False)
            .reset_index(drop=True)
            .reset_index()
        )
        idcg = (idcgs.relevance / np.log1p(1 + idcgs.index)).iloc[:k].sum()

if st.session_state.display == "wide":
    cc = col_features[2]
else:
    st.markdown("---")
    cc = st.columns(1)[0]
with cc:
    cols = st.columns([2, 2])
    with cols[0]:
        st.markdown('#### Résultats pour "' + st.session_state.query + '"')
    with cols[1]:
        if DEMO_STEPS.index(st.session_state.step) >= DEMO_STEPS.index("ndcg-values"):
            st.latex(rf"nDCG_{k}={dcg / idcg:1.3f}")
    display_df = df.nlargest(10, ordering_col).sort_values(
        ordering_col, ascending=ascending
    )
    st.dataframe(
        display_df.style.apply(
            highlight_greaterthan,
            threshold=highlight_threshold,
            column=highlight_column,
            axis=1,
        ).format(precision=2),
        hide_index=True,
        column_order=[c for c in COL_OPTIMAL_ORDER if c in display_df.columns],
    )
