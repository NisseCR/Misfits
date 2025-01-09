import pandas as pd


def _read_data() -> pd.DataFrame:
    return pd.read_csv('./data/cleaned.csv')


def _preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df['text_preprocess'] = df['text']
    # TODO copying data for now, dunno how much preprocessing needed for SA.
    # TODO Thus _preprocess_text is not used atm.
    return df


def _write_data(df: pd.DataFrame) -> None:
    df.to_csv('./data/sa_preprocess.csv', index=False)


def preprocess() -> None:
    df = _read_data()
    df = _preprocess_data(df)
    _write_data(df)
