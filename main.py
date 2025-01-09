import pandas as pd

from app.setup import setup


# Define pandas DataFrame print settings.
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 1000)


def main():
    setup()


if __name__ == '__main__':
    main()
