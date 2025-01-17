import pandas as pd
import spacy
import nltk
import string
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Retrieve pre-defined stop words.
nltk.download('stopwords')
default_stop_words = set(stopwords.words('english'))

# Load spacy corpus and define stemmer
nlp = spacy.load('en_core_web_sm')
stemmer = PorterStemmer()

# Define custom stopwords
custom_stopwords = {'ghana', 'nigeria', 'country', 'countries', 'city'}


def _read_data() -> pd.DataFrame:
    return pd.read_csv('./data/filtered.csv')


def _preprocess_text(text: list) -> str:
    """
    Methond to preprocess text data. The method performs the following steps:
    1. Lemmatize text and isolate nouns.
    2. Remove stop words.
    3. Remove punctuation, numbers, special characters, and extra whitespaces.
    4. Stemming.
    """
    include_features = ['NOUN']
    all_stop_words = default_stop_words.union(custom_stopwords)

    # Lemmatize text and remove stop words.
    text = ' '.join([ent.lemma_.strip() for ent in text
                     if ent.pos_ in include_features and ent.text
                     not in all_stop_words])

    # Remove punctuation, numbers, special characters, and extra whitespaces.
    text = text.translate(str.maketrans('', '',
                                        string.punctuation + string.digits))
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.strip().lower()

    # Stemming
    # text = ' '.join([stemmer.stem(word) for word in text.split()])

    return text


def _preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df['text_preprocess'] = df['text'].apply(nlp)
    df['text_preprocess'] = df['text_preprocess'].apply(
            lambda text: _preprocess_text(text))
    return df


def _write_data(df: pd.DataFrame) -> None:
    df.to_csv('./data/tm_preprocess.csv', index=False)


def preprocess() -> None:
    df = _read_data()
    df = _preprocess_data(df)
    _write_data(df)
