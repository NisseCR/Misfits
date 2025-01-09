"""
Read, clean and filter the data prior to preprocessing and analysis.
"""


import pandas as pd


def _read_data() -> pd.DataFrame:
    df = pd.read_csv('./data/News_China_Africa.csv')
    return df


def _clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df[['headline', 'text', 'label', 'word_count']]
    return df


def _filter_data(df: pd.DataFrame) -> pd.DataFrame:
    # TODO implement actual news article filter.
    # Dummy sample for now (30% of data).
    df = df.sample(frac=0.3, random_state=1)
    return df


def _write_cleaned_data(df: pd.DataFrame) -> None:
    df.to_csv('./data/Cleaned.csv', index=False)


def setup() -> None:
    """
    Wrapper function that retrieves data suitable for further preprocessing and analysis and saves it in the data
    directory.
    """
    df = _read_data()
    df = _clean_data(df)
    df = _filter_data(df)
    _write_cleaned_data(df)
