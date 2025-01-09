import pandas as pd

from app.setup import setup
from app.topic_modelling import tm_preprocessing, tm_analysis
from app.sentiment_analysis import sa_preprocessing, sa_analysis
from app.results import get_results

# Define pandas DataFrame print settings.
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 1000)


def main():
    # MEEP MEEP you can actually comment out each step if you don't want to rerun the whole thing.
    # Intermediate results get saved in files, so you can e.g. go straight to sentiment analysis if need be.

    # Clean and filter raw data.
    setup(sample=0.001)

    # Perform topic modelling.
    tm_preprocessing.preprocess()
    tm_analysis.analyse()

    # Perform sentiment analysis.
    sa_preprocessing.preprocess()
    sa_analysis.analyse()

    # Results
    df = get_results()
    print(df.head())


if __name__ == '__main__':
    main()
