import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import random
random.seed(123)
import warnings
warnings.filterwarnings("ignore")
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def clean_reformat(dataset):
    dataset.loc[dataset.PossessionTeam == 'ARZ', 'PossessionTeam'] = 'ARI'
    dataset.loc[dataset.PossessionTeam == 'BLT', 'PossessionTeam'] = 'BAL'
    dataset.loc[dataset.PossessionTeam == 'CLV', 'PossessionTeam'] = 'CLE'
    dataset.loc[dataset.PossessionTeam == 'HST', 'PossessionTeam'] = 'HOU'
    dataset.loc[dataset.PossessionTeam == 'ARZ', 'FieldPosition'] = 'ARI'
    dataset.loc[dataset.PossessionTeam == 'BLT', 'FieldPosition'] = 'BAL'
    dataset.loc[dataset.PossessionTeam == 'CLV', 'FieldPosition'] = 'CLE'
    dataset.loc[dataset.PossessionTeam == 'HST', 'FieldPosition'] = 'HOU'
    # dataset['GameSnap'] = dataset['GameId'].map(str) + dataset['TimeSnap'].map(str)
    # dataset['GameSnap'] = dataset['PlayId'].map(str)
    # dataset['TimeHandoff'] = pd.to_datetime(dataset['TimeHandoff'], format = "%Y-%m-%dT%H:%M:%S")
    # dataset['TimeSnap'] = pd.to_datetime(dataset['TimeSnap'], format = "%Y-%m-%dT%H:%M:%S")
    # # WindSpeed and WindDirection are switched for some rows in the dataset, the following script is used to switch them back to the sampe spot
    # mask = dataset['WindDirection'].map(lambda x: str(x).isnumeric()) & (dataset['WindSpeed'].map(lambda x: not str(x).isnumeric()))
    # wrong_ds = dataset[mask]
    # dataset.loc[mask, 'WindDirection'] = wrong_ds['WindSpeed']
    # dataset.loc[mask, 'WindSpeed'] = wrong_ds['WindDirection'].astype(int)
    # dataset['WindSpeed'] = dataset['WindSpeed'].apply(lambda x: numerize(x))
    # dataset['Stadium'] = dataset['Stadium'].str.replace('Stadium', '')
    # dataset['Stadium'] = dataset['Stadium'].str.strip()   # remove the leading spaces from the rows
    # dataset['WindSpeed'] = dataset['WindSpeed'].astype(str).str.replace('(mph|MPH|MPh|-.*|g.*)', '')
    # dataset['WindSpeed'] = dataset['WindSpeed'].astype(str).str.strip()
    # dataset.loc[dataset.WindSpeed == 'Calm', 'WindSpeed'] = '0'
    return dataset

def train_test_split(dataset, subset_rate, test_portion):
    game_id = random.sample(list(dataset.GameId.unique()), int(subset_rate * len(dataset.GameId.unique())))
    test_id, train_id = game_id[:int(test_portion * len(game_id))], game_id[int(test_portion * len(game_id)):]
    train_plays = dataset.loc[dataset.GameId.isin(train_id), 'PlayId'].unique()
    test_plays = dataset.loc[dataset.GameId.isin(test_id), 'PlayId'].unique()
    # x_train, y_train = transform_train_test(train_plays)
    # x_test, y_test = transform_train_test(test_plays)
    return train_plays, test_plays

def engineer_rusher_feature(dataset, play_id_list):
    dataset['IsRusher'] = False
    dataset.loc[dataset.NflId == dataset.NflIdRusher, 'IsRusher'] = True
    # dataset['OffenseDefense'] = 'D'
    dataset['OffenseDefense'] = np.where((dataset.PossessionTeam == dataset.HomeTeamAbbr) & \
                                        (dataset.Team == 'home'), 'O', 
                                    np.where((dataset.PossessionTeam == dataset.VisitorTeamAbbr) & \
                                            (dataset.Team == 'away'), 'O', 'D'))
    X = []
    Y = []
    for play_id in play_id_list:
        game_data = dataset[dataset.PlayId == play_id]
        cols_to_dl = ['PlayId', 'X', 'Y', 'S', 'A', 'Orientation', 'IsRusher', 'OffenseDefense', 'PlayDirection', 'Yards']
        spatial_data = game_data[cols_to_dl]
        rusher_data = spatial_data[spatial_data.IsRusher]
        if spatial_data.PlayDirection.values[0] == 'right':
            spatial_data[['X', 'Y']] = spatial_data[['X', 'Y']].values - rusher_data[['X', 'Y']].values
        else:
            spatial_data[['X', 'Y']] = rusher_data[['X', 'Y']].values - spatial_data[['X', 'Y']].values
        spatial_data['RusherDistance'] = np.sqrt(np.square(spatial_data.X) + np.square(spatial_data.Y))
        spatial_data = spatial_data.sort_values(by = ['OffenseDefense', 'RusherDistance'])
        dl_input = list(pd.concat([spatial_data.X, spatial_data.Y, spatial_data.S, spatial_data.A, 
                                  spatial_data.Orientation, spatial_data.RusherDistance]))
        X.append(dl_input)
        dl_output = []
        dl_output[:199] = [0] * 199
        dl_output[99 + int(spatial_data.Yards.values[0])] = 1
        Y.append(dl_output)
    return torch.FloatTensor(X), torch.FloatTensor(Y)

def train_model(x_train, y_train, batch_size = 32, lr = 0.01, epoch_size = 500):
    n_in, n_h, n_out= x_train.shape[1], 200, y_train.shape[1]
    model = nn.Sequential(nn.Linear(n_in, n_h),
                        nn.Sigmoid(), 
                        nn.Linear(n_h, n_out), 
                        nn.Softmax()
                        )
    criterion = nn.MSELoss()
    optim = torch.optim.SGD(model.parameters(), lr = lr)

    for epoch in range(epoch_size):
        y_pred = model(x_train)
        
        # get the loss function
        loss = criterion(y_pred, y_train)
        if epoch % 100 == 0:
            print("epoch %s loss value: %.3f"%(epoch, loss.item()))
            
        # zero the gradient
        optim.zero_grad()
        
        # back propogate
        loss.backward()
        
        # update the parameter
        optim.step()
    return(model)

def validate_model(model, x_test, y_test):
    test_pred = model(x_test)
    test_MSE = torch.mean((test_pred - y_test)**2)
    print('test mean squared error is %.3f' %(test_MSE.item()))


if __name__ == "__main__":
    # metadata
    subset_rate, test_portion = 0.5, 0.2
    dataset = pd.read_csv('datasets/train.csv', low_memory=False)
    dataset = clean_reformat(dataset)
    train_playid, test_playid = train_test_split(dataset, subset_rate = subset_rate, test_portion = test_portion)
    # print(train_playid, test_playid)
    model_x, model_y = engineer_rusher_feature(dataset = dataset, play_id_list = train_playid)
    valid_x, valid_y = engineer_rusher_feature(dataset = dataset, play_id_list = test_playid)
    dl_model = train_model(model_x, model_y, batch_size = 32, lr = 0.01, epoch_size = 10)
    validate_model(dl_model, valid_x, valid_y)
