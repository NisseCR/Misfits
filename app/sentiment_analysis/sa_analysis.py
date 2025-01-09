import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def _read_data() -> pd.DataFrame:
    return pd.read_csv('./data/sa_preprocess.csv')


def _get_sentiment(text: str, model: SentimentIntensityAnalyzer) -> float:
    scores = model.polarity_scores(text)
    return scores['compound']


def _get_sentiments(df: pd.DataFrame) -> pd.DataFrame:
    model = SentimentIntensityAnalyzer()
    df['compound_sentiment'] = df['text_preprocess'].apply(lambda x: _get_sentiment(x, model))
    return df


def _write_data(df: pd.DataFrame) -> None:
    df.to_csv('./data/sa_sentiments.csv', index=False)


def analyse() -> None:
    # https://www.kaggle.com/code/hassanamin/unsupervised-sentiment-analysis-using-vader
    # https://www.geeksforgeeks.org/python-sentiment-analysis-using-vader/
    # TODO Our SA is unsupervised, which forces us to use lexicon based methods prob.

    df = _read_data()
    df = _get_sentiments(df)
    _write_data(df)
