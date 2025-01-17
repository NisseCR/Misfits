import pandas as pd
import matplotlib.pyplot as plt


def _read_data_source() -> pd.DataFrame:
    return pd.read_csv('../data/News_China_Africa.csv')


def _read_data_results() -> pd.DataFrame:
    return pd.read_csv('../data/results.csv')


def _plot_word_counts(df: pd.DataFrame) -> None:
    df['word_count'].plot(kind='hist')
    plt.show()


def plot():
    df_source = _read_data_source()
    df_results = _read_data_results()

    # Plots
    _plot_word_counts(df_source)


if __name__ == '__main__':
    plot()
