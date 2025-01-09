"""
Preprocess the data for topic modelling, including tokenization and using a word representation.
Analyse the data with topic modeling.
"""
from typing import Tuple, List

import pandas as pd
import spacy
import nltk
import string

from gensim.corpora import Dictionary
from nltk.corpus import stopwords
from gensim.models import CoherenceModel, LdaModel
from gensim import corpora


# Retrieve pre-defined stop words.
nltk.download('stopwords')
stop_words = stopwords.words('english')

# Load spacy corpus.
nlp = spacy.load('en_core_web_sm')


def _read_data() -> pd.DataFrame:
    return pd.read_csv('./data/cleaned.csv')


def _preprocess_text(text: list):
    # TODO do proper preprocessing steps for topic modelling.
    include_features = ['ADJ', 'INTJ', 'NOUN', 'PROPN', 'VERB']

    # Lemmatization, stop-word removal and POS feature selection.
    text = ' '.join([ent.lemma_.strip() for ent in text if ent.pos_ in include_features and ent.text not in stop_words])

    # Remove punctuation.
    text = text.translate(str.maketrans(' ', ' ', string.punctuation))

    # Remove numbers.
    text = text.translate(str.maketrans(' ', ' ', string.digits))

    # Enforce lower-casing.
    text = text.lower()

    return text


def _preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df['text_preprocess'] = df['text'].apply(nlp)
    df['text_preprocess'] = df['text_preprocess'].apply(_preprocess_text)
    return df


def _write_data(df: pd.DataFrame) -> None:
    df.to_csv('./data/tm_preprocess.csv', index=False)


def preprocess() -> None:
    df = _read_data()
    df = _preprocess_data(df)
    _write_data(df)


def _read_preprocessed_data() -> pd.DataFrame:
    return pd.read_csv('./data/tm_preprocess.csv')


def _create_lda_model(id2word, corpus, num_topics: int) -> LdaModel:
    return LdaModel(
        corpus=corpus,
        id2word=id2word,
        num_topics=num_topics,
        alpha='symmetric',
        eta='symmetric',
        random_state=123
    )


def _setup_lda_data(df: pd.DataFrame) -> tuple[Dictionary, list[list[tuple[int, int]]]]:
    # Create word2id dictionary.
    comments = df['text_preprocess'].apply(lambda x: x.split(' ')).copy()
    id2word = corpora.Dictionary(comments.values)

    # Filter out extremes to limit the number of features.
    id2word.filter_extremes(
        no_below=2,
        no_above=0.95
    )

    # Create BoW collection.
    corpus = [id2word.doc2bow(doc) for doc in comments.values]

    return id2word, corpus


def _write_topics(df: pd.DataFrame) -> None:
    df.to_csv('./data/tm_topics.csv', index=False)


def analyse() -> None:
    df = _read_preprocessed_data()
    id2word, corpus = _setup_lda_data(df)
    lda_model = _create_lda_model(id2word, corpus, 7)

    # TODO doesn't full work yet i think, but brain is now deadge
    # TODO apart from getting global topics, we should also assign topics to each entry (or perhaps only the dominant topic, as Dennis mentioned).
    # TODO choose word representation and model for topic modelling.

    _write_topics(df)
