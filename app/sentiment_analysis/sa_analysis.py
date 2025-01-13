import pandas as pd
from transformers import pipeline, Pipeline


def _read_data() -> pd.DataFrame:
    return pd.read_csv('./data/sa_preprocess.csv')


def _get_sentiment(text: str, classifier: Pipeline) -> dict:
    # TODO Currently runs into an error due to texts being too long.
    # TODO https://github.com/huggingface/transformers/issues/1791
    # TODO fix, otherwise works on shorter texts
    return classifier(text)[0]


def _get_sentiments(df: pd.DataFrame) -> pd.DataFrame:
    # TODO Be more mindful of parameters we choose, rather that sticking to default values.
    classifier = pipeline(model='distilbert/distilbert-base-uncased-finetuned-sst-2-english')

    # Save classifier sentiment output.
    df['class'] = df['text_preprocess'].apply(lambda x: _get_sentiment(x, classifier))
    df['sentiment'] = df['class'].apply(lambda x: x['label'])
    df['score'] = df['class'].apply(lambda x: x['score'])
    df = df.drop(columns=['class'])
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
