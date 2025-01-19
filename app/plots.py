import pandas as pd
import matplotlib.pyplot as plt


# Define pandas DataFrame print settings.
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 1000)


def _read_data_source() -> pd.DataFrame:
    return pd.read_csv('../data/News_China_Africa.csv')


def _read_data_results() -> pd.DataFrame:
    return pd.read_csv('../data/results.csv')


def _plot_word_counts(df: pd.DataFrame) -> None:
    df['word_count'].plot(kind='hist')
    plt.show()


def plot():
    df = _read_data_results()

    # Plots
    plt.hist(df['sentiment'])
    plt.show()


if __name__ == '__main__':
    plot()
