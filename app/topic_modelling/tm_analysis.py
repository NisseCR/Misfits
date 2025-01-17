import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel, LdaModel
from gensim import corpora
from matplotlib import pyplot as plt
from wordcloud import WordCloud
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import confusion_matrix
import seaborn as sns


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
                       num_clusters: int) -> KMeans:
    # Use the same features as for LDA, dumb dumb
    relevant_tfidf = tfidf_df[relevant_features]
    kmeans = KMeans(n_clusters=num_clusters, random_state=123)
    kmeans.fit(relevant_tfidf)
    return kmeans


def plot_wordcloud(model, feature_names=None, num_words=10):
    """
    Generate word clouds for the provided model.

    :param model: Trained LDA or KMeans model.
    :param feature_names: List of feature names used by the model.
    :param num_words: Number of top words to display per topic/cluster.
    """
    if isinstance(model, LdaModel):
        topics = model.show_topics(formatted=False, num_topics=-1)
        num_topics = len(topics)
    elif isinstance(model, KMeans):
        num_topics = len(model.cluster_centers_)
    else:
        raise ValueError("Unsupported model.")

    num_cols = 3
    num_rows = (num_topics + num_cols - 1) // num_cols

    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, num_rows * 5))
    axes = axes.flatten()

    # Loop through topics or clusters and generate word clouds
    for i in range(num_topics):
        if isinstance(model, LdaModel):
            topic = topics[i]
            topic_words = {word: prob for word, prob in topic[1][:num_words]}
        elif isinstance(model, KMeans):
            order_centroids = model.cluster_centers_.argsort()[:, ::-1]
            topic_words = {feature_names[ind]: model.cluster_centers_[i, ind]
                           for ind in order_centroids[i, :num_words]}

        cloud = WordCloud(max_font_size=300,
                          background_color="white").generate_from_frequencies(topic_words)
        ax = axes[i]
        ax.imshow(cloud, interpolation='bilinear')
        ax.set_title(f"Topic {i}" if isinstance(model, LdaModel) else f"Cluster {i}", fontdict={'size': 16})
        ax.axis('off')

    # Hide unused axes if any
    for unused in range(num_topics, len(axes)):
        axes[unused].axis('off')

    plt.show()


# THis code is copied from a medium article
def _lda_best_num_topics(corpus, id2word, df):
    min_topics = 5
    max_topics = 30
    step_size = 5
    topics_range = range(min_topics, max_topics + 1, step_size)

    coherence_scores = []
    optimal_num_topics = None
    max_coherence_score = -np.inf

    # Loop through numbers of topics
    for num_topics in topics_range:
        try:
            lda_model = LdaModel(corpus=corpus, id2word=id2word, num_topics=num_topics)

            # Coherence model
            coherence_model_lda = CoherenceModel(model=lda_model,
                                                 texts=df['text_preprocess'].apply(lambda x: x.split()),
                                                 dictionary=id2word, coherence='c_v')
            coherence_lda = coherence_model_lda.get_coherence()

            # Track the best coherence score and number of topics
            coherence_scores.append(coherence_lda)

            if coherence_lda > max_coherence_score:
                max_coherence_score = coherence_lda
                optimal_num_topics = num_topics

        except Exception as e:
            print(f"Error with {num_topics} topics: {e}")
            coherence_scores.append(0)

    print("Optimal Number of Topics:", optimal_num_topics)
    print("Coherence Score for Optimal Number of Topics:", max_coherence_score)

    # Plotting the relationship between the number of topics and coherence score
    plt.figure(figsize=(10, 6))
    plt.plot(topics_range, coherence_scores, marker='o', color='b', label='Coherence Score')
    plt.xlabel('Number of Topics')
    plt.ylabel('Coherence Score')
    plt.title('Number of Topics vs Coherence Score')
    plt.xticks(topics_range)
    plt.legend()
    plt.grid(True)
    plt.show()

    return optimal_num_topics


def compare_rand_index(df):
    ari_score = adjusted_rand_score(df['dominant_topic'], df['kmeans_cluster'])
    print(f"Adjusted Rand Index between LDA Topics and K-means Clusters: {ari_score:.4f}")


def get_top_lda_words(lda_model, num_words=10):
    topics = lda_model.show_topics(formatted=False, num_topics=-1)
    top_words = {}

    for topic_id, topic in topics:
        words = [word for word, _ in topic[:num_words]]
        top_words[topic_id] = words

    return top_words


def get_top_kmeans_words(kmeans_model, feature_names, num_words=10):
    top_words = {}

    # Sort centroids for each cluster
    order_centroids = kmeans_model.cluster_centers_.argsort()[:, ::-1]

    for cluster_id in range(len(order_centroids)):
        words = [feature_names[ind] for ind in order_centroids[cluster_id, :num_words]]
        top_words[cluster_id] = words

    return top_words


def _write_data(df: pd.DataFrame) -> None:
    df.to_csv('./data/tm_topics.csv', index=False)


def analyse():
    df = _read_data()

    tfidf_df, feature_names = _compute_tfidf(df)
    relevant_features = _filter_by_tfidf(tfidf_df, feature_names, 0.002,
                                         0.007)
    id2word, corpus = _setup_lda_data(df, relevant_features)

    lda_model = _create_lda_model(id2word, corpus, 10)
    plot_wordcloud(lda_model, relevant_features)
    #_lda_best_num_topics(corpus, id2word, df)
    df['dominant_topic'] = _get_dominant_topic(lda_model, corpus)
    top_lda_words = get_top_lda_words(lda_model)
    for topic_id, words in top_lda_words.items():
        print(f"Topic {topic_id}: {', '.join(words)}")

    kmeans_model = _kmeans_clustering(tfidf_df, relevant_features, 10)
    plot_wordcloud(kmeans_model, relevant_features)
    df['kmeans_cluster'] = kmeans_model.labels_
    top_kmeans_words = get_top_kmeans_words(kmeans_model, feature_names)
    for cluster_id, words in top_kmeans_words.items():
        print(f"Cluster {cluster_id}: {', '.join(words)}")

    compare_rand_index(df)

    print(df)
    _write_data(df)
