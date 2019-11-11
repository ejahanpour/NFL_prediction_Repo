import sys, os
sys.path.insert(0, '/Users/ehsan/Desktop/NFL_prediction')

import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

from models import data_cleaning
from models import spatial_feature_engineering
from imputation import impute_categories


""" this is the mimic of the final doc that has the two functions for submitting the final results"""



def cleaning_blue_print(data, is_train = True, save = False, player_statistics = None):
    # cleaning dataset 
    clean_data = data_cleaning.clean_reformat(data)
    clean_data = data_cleaning.group_feature(clean_data)
    # engineering new features
    if is_train:
        player_strength = data_cleaning.rush_player_statistics(clean_data)
    elif is_train == False:
        player_strength = player_statistics
    # print(player_strength.head())
    clean_data = pd.merge(clean_data, player_strength, how = 'left', left_on = 'NflIdRusher', right_on = 'RusherId')
    cols_to_fillna = ['RusherHeight', 'RusherWeight', 'RusherYards', 'RusherAge', 'RusherX', 'RusherY', 'RusherSpeed']
    clean_data[cols_to_fillna] = clean_data[cols_to_fillna].fillna(0)
    engineered_data = data_cleaning.engineer_feature(clean_data)
    # imputation 
    imputed_data = data_cleaning.impute_feature(engineered_data)
    # trimming (remove unncessary columns)
    trimmed_data = data_cleaning.trim_data(imputed_data)
    if is_train & save:
        trimmed_data.to_csv('datasets/train_cleaned_data_v1_1.csv', index = False)
    elif save:
        trimmed_data.to_csv('datasets/test_cleaned_data_v1_1.csv', index = False)
    return trimmed_data, player_strength

def spatial_blue_print(data, is_train = True, save = False):
    # clean_dataset
    cleaned_data = data_cleaning.clean_reformat(data)
    # print(cleaned_data[cleaned_data.GameSnap == '20170917082017-09-17T20:06:10.000Z'])
    # group positions based on their place on field
    grouped_data = spatial_feature_engineering.group_positions(cleaned_data)
    # spatial_data = cleaned_data
    spatial_data = spatial_feature_engineering.extract_spatial_features(grouped_data, 'GameSnap')
    spatial_data = spatial_feature_engineering.tweek_positions_prior_knowledge(spatial_data)
    if is_train & save:
        spatial_data.to_csv('datasets/train_spatial_data_v1_1.csv', index = False)
    elif save:
        spatial_data.to_csv('datasets/test_spatial_data_v1_1.csv', index = False)
    return(spatial_data)


def train_my_model(train_dataset):
    clean_data, rusher_char = cleaning_blue_print(train_dataset, save = False)
    spatial_data = spatial_blue_print(train_dataset, save = False)
    # for testing purposes
    # clean_data = pd.read_csv('datasets/train_cleaned_data_v1_1.csv')
    # spatial_data = pd.read_csv('datasets/train_spatial_data_v1_1.csv')
    # print(clean_data.dtypes['GameSnap'], spatial_data.dtypes['_GameSnap'])
    total_data = pd.merge(clean_data, spatial_data, left_on = 'GameSnap', right_on = '_GameSnap', how = 'left')
    dataset = total_data[total_data.QB1_offense_mean_distance.notnull()].drop(['GameSnap', '_GameSnap'], axis = 1)   # in the dataset this column is not populated
    train, test = train_test_split(dataset, test_size = 0.3, random_state = 123)
    # print(train.isnull().sum().to_string())
    x_train = train.drop(['Yards'], axis = 1)
    x_test = test.drop(['Yards'], axis = 1)
    y_train = train['Yards']
    y_test = test['Yards']

    # finding the categorical variables
    cat_features = train.select_dtypes(include = ['object']).columns
    # Ordinal encoding of the categorical variables
    enc = OrdinalEncoder()
    enc.fit(dataset[cat_features])
    x_train[cat_features] = enc.transform(x_train[cat_features])
    # print(cat_features)
    # Random Forrest model
    RF_model = RandomForestRegressor()
    RF_model.fit(x_train, y_train)
    return(RF_model, enc, cat_features, rusher_char)    

def make_my_predictions(model, encoder, cat_features, dataset, sample_prediction_df, rusher_char):
    clean_dataset, _ = cleaning_blue_print(data = dataset, is_train = False, player_statistics = rusher_char)
    # print(clean_dataset.isnull().sum())
    clean_dataset = impute_categories(clean_dataset, cat_features, encoder.categories_)
    spatial_data = spatial_blue_print(dataset)
    total_data = pd.merge(clean_dataset, spatial_data, left_on = "GameSnap", right_on = "_GameSnap", how = 'left')
    model_dataset = total_data.drop(['GameSnap', '_GameSnap'], axis = 1)
    # print(model_dataset.isnull().sum().to_string())
    # cat_features = model_dataset.select_dtypes(include = ['object']).columns
    # model_dataset[cat_features] = encoder.transform(model_dataset[cat_features])
    predictions = model.predict(model_dataset)
    for i, pred in enumerate(predictions):
        sample_prediction_df.iloc[i, :int(98 + pred)] = 0
        sample_prediction_df.iloc[i, int(98 + pred)] = 0.5
        sample_prediction_df.iloc[i, int(99 + pred):] = 1
    return sample_prediction_df, predictions

def evaluate_model(predicion_matrix, observations):
    observation_df =  pd.DataFrame(0, index = np.arange(len(observations)), columns = np.arange(-99, 100))
    for i, obs in enumerate(observations):
        observation_df.iloc[i, int(99 + obs):] = 1
    C = ((predicion_matrix - observation_df)**2).sum().sum()/199*len(observations)
    return(C)

def MSE(predictions, observations):
    MSE = np.sqrt(np.sum((predictions - observations)**2))
    return MSE


if __name__ == "__main__":
    # train_ds = pd.read_csv('datasets/split_train_30.csv')
    # clean_data, rusher_char = cleaning_blue_print(train_dataset, save = False)
    # spatial_data = spatial_blue_print(train_dataset, save = False)
    # total_data = pd.merge(clean_data, spatial_data, left_on = 'GameSnap', right_on = '_GameSnap', how = 'left')


    train_ds = pd.read_csv('datasets/split_train_30.csv')[:418]
    test_ds = pd.read_csv('datasets/split_test_30.csv')[:418]
    observations = test_ds[['PlayId', 'Yards']].drop_duplicates().Yards
    test_ds = test_ds.drop('Yards', axis = 1)
    model, encoder, cat_features, rusher = train_my_model(train_ds)
    sample_prediction_df =  pd.DataFrame(0, index = np.arange(len(observations)), columns = np.arange(-99, 100))
    prediction_matrix, predictions = make_my_predictions(model, encoder, cat_features, test_ds, sample_prediction_df, rusher_char = rusher)
    model_performance = evaluate_model(prediction_matrix, observations)
    model_mse = MSE(predictions, observations)
    print('the performance of the model is %.2f and the model MSE is %.2f' %(model_performance, model_mse))
