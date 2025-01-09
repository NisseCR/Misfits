"""
Preprocess the data for topic modelling, including tokenization and using a word representation.
Analyse the data with topic modeling.
"""


import pandas as pd
import spacy
import nltk
import string
from nltk.corpus import stopwords
from gensim.models import CoherenceModel, LdaModel


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


def analyse() -> None:
    df = _read_preprocessed_data()
    print(df.head())
