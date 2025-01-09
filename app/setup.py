"""
Read, clean and filter the data prior to preprocessing and analysis.
"""


import pandas as pd


def _read_data() -> pd.DataFrame:
    return pd.read_csv('./data/News_China_Africa.csv')


def _clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Keep only informative attributes.
    df = df[['headline', 'text', 'label', 'word_count']]

    # Save idnex of each entry in dedicated column 'id'.
    df = df.reset_index(names='id')
    return df


def _filter_data(df: pd.DataFrame, frac: float) -> pd.DataFrame:
    # TODO implement actual news article filter.
    # Dummy sample for now.
    df = df.sample(frac=frac, random_state=1)
    return df


def _write_data(df: pd.DataFrame) -> None:
    df.to_csv('./data/cleaned.csv', index=False)


def setup(sample: float = 1) -> None:
    """
    Wrapper function that retrieves data suitable for further preprocessing and analysis and saves it in the data
    directory.
    """
    df = _read_data()
    df = _clean_data(df)
    df = _filter_data(df, sample)
    _write_data(df)
