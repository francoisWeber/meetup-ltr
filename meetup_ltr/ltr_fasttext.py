import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import fasttext.util
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F
from openai import OpenAI
from os import environ

query_vec = np.load("./query_ft.npy")

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

with st.sidebar:
    st.radio("embedding type", options=["FastText", "E5"], key="embedding")
    st.toggle("RAG", value=False, key="rag")
    if st.session_state.rag:
        st.slider("RAG's top-k", min_value=2, max_value=5, step=1, key="rag_top_k")


def similarity(docs_array: np.ndarray, query_tf):
    return cosine_similarity(docs_array, query_vec)


def average_pool(
    last_hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


@st.cache_data(max_entries=1, show_spinner=True)
def load_corpus(embedding_type: str):
    if embedding_type == "FastText":
        return pd.read_csv(
            "/Users/f.weber/Code/meetup-ltr/dataset_fasttext.csv",
            index_col=0,
            converters={"fasttext": lambda s: np.fromstring(s.strip("[]\n"), sep=" ")},
        ).rename(columns={"fasttext": "embedding"})
    elif embedding_type == "E5":
        return pd.read_csv(
            "/Users/f.weber/Code/meetup-ltr/dataset_e5.csv",
            index_col=0,
            converters={"e5": lambda s: np.fromstring(s.strip("[]\n"), sep=" ")},
        ).rename(columns={"e5": "embedding"})


@st.cache_resource
def get_bot():
    return OpenAI(api_key=environ["OPENAI_API_KEY"])


@st.spinner("Doing magic :p")
def summurize(query: str, docs: list, pre_prompt: str):
    content = [
        f"Voici la question de l'utilisateur: {query}",
        "Voici les documents sur lequels t'appuyer pour répondre:",
    ]
    content += [f"document {i+1}: {doc}." for i, doc in enumerate(docs)]
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": pre_prompt,
            },
            {
                "role": "user",
                "content": "\n".join(content),
            },
        ],
    )
    return completion.choices[0].message.content


class FasttextVectorizer:
    def __init__(self, ft):
        self.ft = ft

    def vectorize(self, text: str):
        return self.ft.get_sentence_vector(text).reshape(1, -1)


class E5Vectorizer:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    @st.spinner(text="computing embedding ...")
    def vectorize(self, text: str):
        d = self.tokenizer(
            "query: " + text,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            e = self.model(**d)
        return F.normalize(
            average_pool(e.last_hidden_state, d["attention_mask"]), p=2, dim=1
        ).numpy()


@st.cache_resource(max_entries=1, show_spinner=True)
def load_vectorizer(embedding_type: str):
    if embedding_type == "FastText":
        return FasttextVectorizer(
            fasttext.load_model("/Users/f.weber/Code/cc.fr.300.bin")
        )
    else:
        return E5Vectorizer("intfloat/multilingual-e5-small")


df = load_corpus(st.session_state.embedding)
vectorizer = load_vectorizer(st.session_state.embedding)
docs_array = np.array(df.embedding.tolist())

st.text_input("query ?", value="Meetup deep learning strasbourg".lower(), key="query")
query_vec = vectorizer.vectorize(st.session_state.query)
st.markdown(
    f"query embedding via {st.session_state.embedding}: {query_vec.ravel()[:5]} ... "
)

df["similarity"] = similarity(docs_array, query_vec)

# now get closest docs
df.sort_values("similarity", ascending=False)

st.dataframe(
    df.sort_values("similarity", ascending=False).iloc[:10].style.format(precision=3),
    hide_index=True,
    column_order=["similarity", "content", "embedding"],
)


if st.session_state.rag:
    client = get_bot()
    # pre_prompt = """You are a helpful assistant, ready to summurize relevant documents to answer a query.
    # If the query is vague, don't hesitate to derive a plausible question from it.
    # If a document is not relevant, try to find anything related to the query in it or ignore it"""
    pre_prompt = """You are a question-answering assistant that take as input a query and documents that might answer the query. 
    You have to summurize the corpus to answer the question. 
    If a document does not answer the question, cherry-pick some information related to the query to build your answer"""

    st.markdown(
        f"Considérons les {st.session_state.rag_top_k} meilleurs documents et essayons de répondre à la question avec ..."
    )
    docs_top_k = df.nlargest(st.session_state.rag_top_k, "similarity").content.tolist()
    resp = summurize(st.session_state.query, docs_top_k, pre_prompt=pre_prompt)

    st.markdown(resp)
