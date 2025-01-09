import pandas as pd

from gensim.corpora import Dictionary
from gensim.models import CoherenceModel, LdaModel
from gensim import corpora


def _read_data() -> pd.DataFrame:
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


def _write_data(df: pd.DataFrame) -> None:
    df.to_csv('./data/tm_topics.csv', index=False)


def analyse() -> None:
    df = _read_data()
    id2word, corpus = _setup_lda_data(df)
    lda_model = _create_lda_model(id2word, corpus, 7)

    # TODO doesn't full work yet i think, but brain is now deadge
    # TODO apart from getting global topics, we should also assign topics to each entry (or perhaps only the dominant topic, as Dennis mentioned).
    # TODO choose word representation and model for topic modelling.

    _write_data(df)
