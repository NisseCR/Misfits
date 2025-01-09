import pandas as pd

from app.setup import setup
from app.topic_modelling import tm_preprocessing, tm_analysis
from app.sentiment_analysis import sa_preprocessing, sa_analysis
from app import sentiment_analysis

# Define pandas DataFrame print settings.
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 1000)


def main():
    # Clean and filter raw data.
    setup(sample=0.001)

    # Perform topic modelling.
    # tm_preprocessing.preprocess()
    # tm_analysis.analyse()

    # Perform sentiment analysis.
    sa_preprocessing.preprocess()
    sa_analysis.analyse()


if __name__ == '__main__':
    main()
