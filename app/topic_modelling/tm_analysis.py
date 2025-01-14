import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel, LdaModel
from gensim import corpora


def _read_data() -> pd.DataFrame:
    return pd.read_csv('./data/tm_preprocess.csv')


def _compute_tfidf(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Compute the TF-IDF matrix to identify the most relevant words for LDA,
    and to apply the k-means clustering.
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['text_preprocess'])
    feature_names = vectorizer.get_feature_names_out()
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
    return tfidf_df, feature_names


# Based on the tfidf matrix, filter out words for LDA based on thresholds.
def _filter_by_tfidf(tfidf_df: pd.DataFrame, feature_names: list,
                     low_threshold: float, high_threshold: float) -> list[str]:
    feature_scores = tfidf_df.mean(axis=0)
    keep_features = [feature_names[i] for i, score in enumerate(feature_scores)
                     if low_threshold <= score <= high_threshold]
    return keep_features


# Create an id2word representation and corpus for LDA model
def _setup_lda_data(df: pd.DataFrame, relvant_features: list) -> (
        tuple)[Dictionary, list[list[tuple[int, int]]]]:
    comments = df['text_preprocess'].apply(
            lambda x: [word for word in x.split() if word in relvant_features])
    id2word = corpora.Dictionary(comments.values)

    # Create BoW collection.
    corpus = [id2word.doc2bow(doc) for doc in comments.values]

    return id2word, corpus


def _create_lda_model(id2word, corpus, num_topics: int) -> LdaModel:
    return LdaModel(
            corpus=corpus,
            id2word=id2word,
            num_topics=num_topics,
            alpha='auto',
            random_state=123,
            per_word_topics=True
    )


def _get_dominant_topic(lda_model, corpus) -> list[int]:
    dominant_topics = []
    for doc in corpus:
        topics = lda_model.get_document_topics(doc)
        dominant_topic = max(topics, key=lambda x: x[1])[0]
        dominant_topics.append(dominant_topic)
    return dominant_topics


def _kmeans_clustering(tfidf_df: pd.DataFrame,
                       relevant_features: list,
                       num_clusters: int) -> list[int]:
    # Use the same features as for LDA.
    relevant_tfidf = tfidf_df[relevant_features]
    kmeans = KMeans(n_clusters=num_clusters, random_state=123)
    kmeans.fit(tfidf_df)
    return kmeans.labels_


def _write_data(df: pd.DataFrame) -> None:
    df.to_csv('./data/tm_topics.csv', index=False)


def analyse():
    df = _read_data()

    tfidf_df, feature_names = _compute_tfidf(df)
    relevant_features = _filter_by_tfidf(tfidf_df, feature_names, 0.002,
                                         0.007)

    id2word, corpus = _setup_lda_data(df, relevant_features)
    lda_model = _create_lda_model(id2word, corpus, 7)
    df['dominant_topic'] = _get_dominant_topic(lda_model, corpus)

    kmeans_labels = _kmeans_clustering(tfidf_df, relevant_features, 7)
    df['kmeans_cluster'] = kmeans_labels

    print(df)
    _write_data(df)
