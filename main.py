import pandas as pd

from app.setup import setup
from app import topic_modelling
from app import sentiment_analysis

# Define pandas DataFrame print settings.
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 1000)


def main():
    # Clean and filter raw data.
    # setup(sample=0.001)

    # Perform topic modelling.
    # topic_modelling.preprocess()
    topic_modelling.analyse()

    # Perform sentiment analysis.
    sentiment_analysis.preprocess()


if __name__ == '__main__':
    main()
