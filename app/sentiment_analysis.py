"""
Preprocess the data for sentiment analysis, including tokenization and using a word representation.
"""


import pandas as pd
import spacy


def _read_data() -> pd.DataFrame:
    df = pd.read_csv('./data/cleaned.csv')
    return df


def _write_data(df: pd.DataFrame) -> None:
    df.to_csv('./data/sa_preprocess.csv', index=True)


def preprocess() -> None:
    df = _read_data()
    _write_data(df)


def _read_preprocessed_data() -> pd.DataFrame:
    return pd.read_csv('./data/sa_preprocess.csv')


def _write_topics(df: pd.DataFrame) -> None:
    df.to_csv('./data/sa_sentiments.csv', index=False)


def analyse() -> None:
    df = _read_preprocessed_data()
    _write_topics(df)
