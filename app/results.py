from typing import Tuple

import pandas as pd
from pandas import DataFrame


def _read_data() -> tuple[DataFrame, DataFrame]:
    df_topics = pd.read_csv('./data/tm_topics.csv')
    df_sentiments = pd.read_csv('./data/sa_sentiments.csv')
    return df_topics, df_sentiments


def _merge_data(df_topics: DataFrame, df_sentiments: DataFrame) -> pd.DataFrame:
    # Select only columns unique to right-hand side.
    df_sentiments = df_sentiments[['id', 'compound_sentiment']]

    # Perform merge of data.
    df = pd.merge(df_topics, df_sentiments, how='inner', on='id')
    return df


def get_results() -> pd.DataFrame:
    df_topics, df_sentiments = _read_data()
    df = _merge_data(df_topics, df_sentiments)
    df = df[['id', 'headline', 'text', 'dominant_topic', 'compound_sentiment']]
    return df
