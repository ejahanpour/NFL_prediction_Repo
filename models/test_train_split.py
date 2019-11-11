from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


def train_test_split(dataset, column, train_size, sample_size):
    unique_games = dataset[column].unique()
    split_point = int(np.floor(sample_size * train_size * len(unique_games)))
    partition_size = int(np.ceil(sample_size * len(unique_games)))
    np.random.shuffle(unique_games)
    train_games, test_games = unique_games[:split_point], unique_games[split_point:partition_size]
    train_data = dataset[dataset[column].isin(train_games)]
    train_data.to_csv('datasets/split_train.csv', index = False)
    test_data = dataset[dataset[column].isin(test_games)]
    test_data.to_csv('datasets/split_test.csv', index = False)


if __name__ == "__main__":
    dataset = pd.read_csv('datasets/train.csv', low_memory = False)
    train_test_split(dataset, 'GameId', 0.9, 1)