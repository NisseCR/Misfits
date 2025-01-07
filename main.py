import pandas as pd


def read_data() -> pd.DataFrame:
    df = pd.read_csv('./data/News_China_Africa.csv')
    return df


if __name__ == '__main__':
    df = read_data()
    print(df.head())