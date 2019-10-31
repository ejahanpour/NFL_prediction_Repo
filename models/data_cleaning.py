import pandas as pd
import numpy as np
import re

## helper functions
def numerize(x):
    try:
        return str(round(float(x)))
    except ValueError:
        return x

def find_position_number(row, pos):
    try:
        return int(re.findall('\d', [i for i in row.split(',') if pos in i][0])[0])
    except:
        return 0

###########
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
    dataset['GameSnap'] = dataset['PlayId'].map(str)
    dataset['TimeHandoff'] = pd.to_datetime(dataset['TimeHandoff'], format = "%Y-%m-%dT%H:%M:%S")
    dataset['TimeSnap'] = pd.to_datetime(dataset['TimeSnap'], format = "%Y-%m-%dT%H:%M:%S")
    # WindSpeed and WindDirection are switched for some rows in the dataset, the following script is used to switch them back to the sampe spot
    mask = dataset['WindDirection'].map(lambda x: str(x).isnumeric()) & (dataset['WindSpeed'].map(lambda x: not str(x).isnumeric()))
    wrong_ds = dataset[mask]
    dataset.loc[mask, 'WindDirection'] = wrong_ds['WindSpeed']
    dataset.loc[mask, 'WindSpeed'] = wrong_ds['WindDirection'].astype(int)
    dataset['WindSpeed'] = dataset['WindSpeed'].apply(lambda x: numerize(x))
    dataset['Stadium'] = dataset['Stadium'].str.replace('Stadium', '')
    dataset['Stadium'] = dataset['Stadium'].str.strip()   # remove the leading spaces from the rows
    dataset['WindSpeed'] = dataset['WindSpeed'].astype(str).str.replace('(mph|MPH|MPh|-.*|g.*)', '')
    dataset['WindSpeed'] = dataset['WindSpeed'].astype(str).str.strip()
    dataset.loc[dataset.WindSpeed == 'Calm', 'WindSpeed'] = '0'
    return dataset
    
def group_feature(dataset):
    """ This is to group different type of typings error
    """
    directions = {'East': 'E', 'from W': 'E', 'From W': 'E', 'EAST': 'E',
              'West': 'W',
              'North': 'N', 'From S':'N', 
              'South': 'S', 's': 'S',
              'Northwest':'NW', 
              'NorthEast' :'NE', 'North East': 'NE',
              'SouthWest':'SW', 'Southwest': 'SW', 
              'Southeast': 'SE',
              'Northeast': 'NE', 'From SW': 'NE', 
              'West-Southwest': 'WSW', 'W-SW': 'WSW',
              'South Southeast': 'SSE', 'From NNW': 'SSE',
              'W-NW': 'WNW',
              'South Southwest': 'SSW', 'From NNE': 'SSW',
              'From WSW': 'ENE', 'East North East': 'ENE',
              'West Northwest': 'WNW', 'From ESE': 'WNW',
              'From SSE': 'NNW',
              'East Southeast': 'ESE',
              'From SSW': 'NNE', 'North/Northwest': 'NNE', 'N-NE': 'NNE',
              }
    dataset = dataset.replace({'WindDirection': directions})

    ST = {'Outdoors': 'Outdoor', 'Open': 'Outdoor', 'Outddors': 'Outdoor', 'Ourdoor': 'Outdoor', 'Oudoor': 'Outdoor', 'Outdor': 'Outdoor', 'Outside': 'Outdoor',
     'Indoors': 'Indoor', 'Indoor, Roof Closed': 'Indoor', 'Indoor, Open Roof': 'Indoor',
     'Retr. Roof-Closed': 'Retractable Roof', 'Retr. Roof - Closed': 'Retractable Roof', 'Retr. Roof Closed': 'Retractable Roof',
     'Retr. Roof-Open': 'Retractable Roof', 'Retr. Roof - Open': 'Retractable Roof',
     'Outdoor Retr Roof-Open': 'Retractable Roof',
     'Domed, closed': 'Dome', 'Closed Dome': 'Dome', 'Dome, closed': 'Dome', 'Domed, Open': 'Dome', 'Domed, open': 'Dome', 'Domed': 'Dome',
      'Cloudy': np.nan, 'Bowl': np.nan, 'Heinz Field': np.nan
     }
    dataset = dataset.replace({'StadiumType': ST})
    return dataset

def engineer_feature(dataset):
    dataset['SnapToHandoff'] = (dataset['TimeHandoff'] - dataset['TimeSnap']).astype('timedelta64[s]')
    fraction = [1, 1/60, 0]
    dataset['TimeRemained'] = dataset.GameClock.apply(lambda x: sum([a*b for a, b in zip(fraction, map(int, x.split(':')))]))
    dataset['PreviousGameStat'] = 'won'
    dataset.loc[dataset.HomeScoreBeforePlay < dataset.VisitorScoreBeforePlay , 'PreviousGameStat'] = 'lost'
    dataset.loc[dataset.HomeScoreBeforePlay == dataset.VisitorScoreBeforePlay , 'PreviousGameStat'] = 'tie'
    dataset.loc[dataset.Week== 1, 'PreviousGameStat'] = 'first'
    dataset['PreviousGameGoalGap'] = dataset['HomeScoreBeforePlay'] - dataset['VisitorScoreBeforePlay']
    # calculate the actual distance to goal
    dataset['DistanceToGoal'] = np.where(dataset['PossessionTeam'] == dataset['FieldPosition'], 50 + dataset['YardLine'], dataset['YardLine'])
    def_pos = ['DL', 'LB', 'DB', 'OL']
    off_pos = ['RB', 'TE', 'WR', 'QB']
    for dp in def_pos:
        dataset['D-'+dp] = [find_position_number(row, dp) for row in dataset.DefensePersonnel]
    for op in off_pos:
        dataset['O-'+op] = [find_position_number(row, op) for row in dataset.OffensePersonnel]
    dataset['O-QB'] = np.where(dataset['O-QB'] == 0, 1, dataset['O-QB'])
    dataset['O-OL'] = 11- dataset[['O-RB', 'O-TE', 'O-WR', 'O-QB']].sum(axis = 1)
    return dataset

def impute_feature(dataset):
    dataset['FieldSide'] = 'other'
    dataset.loc[dataset.PossessionTeam == dataset.FieldPosition, 'FieldSide'] = 'own'
    # no windspeed is kind of considered as 'Calm' in WindDirection
    dataset.loc[dataset.WindSpeed == 0, 'WindDirection'] = 'Calm'
    dataset.loc[dataset.WindDirection == 'Calm', 'WindSpeed'] = 0
    # impute stadium
    studium_type = {'Heinz Field': 'Outdoor', 'MetLife': 'Outdoor', 'StubHub Center': 'Outdoor', 'TIAA Bank Field': 'Outdoor'}
    studium_type = {'Heinz Field': 'Outdoor', 'MetLife': 'Outdoor', 'StubHub Center': 'Outdoor', 'TIAA Bank Field': 'Outdoor'}
    dataset['StadiumType'] = dataset['StadiumType'].fillna(dataset.Stadium.map(studium_type))
    return dataset

def trim_data(dataset):
    redundant_cols = ['GameId', 'Team', # being used to define the schematic features
                 'DisplayName', # NflId has the same info
                 'JerseyNumber',
                 'PlayId', # GameId has the same info 
                 'TimeHandoff', 'TimeSnap', #the time between them is important
                 'X', 'Y', 'S', 'A', #player related info and should be extracted for the game
                 'Dis', 'Orientation', 'Dir',
                 'NflId', 'DisplayName', 'JerseyNumber', 
                  'GameClock',   # featured new timebased out of that   
                 'FieldPosition',   # engineered new features out of these two
                 'HomeScoreBeforePlay', 'VisitorScoreBeforePlay',  # engineered new features 
                 'PlayerHeight', 'PlayerWeight', 'PlayerBirthDate',
                 'PlayerCollegeName', 'Position', 
                 'Stadium', 'Location', # highly correlated with the HomeTeamAbbreviation
                  'DefensePersonnel', 'OffensePersonnel',  #splitted the results into multiple columns
                  'StadiumType' # there might be not known in test data
                 ]
    clean_data = dataset.drop(redundant_cols, axis = 1).drop_duplicates()
    
    temp_removals = ['GameWeather', 'Temperature', 'Humidity', 'WindSpeed', 'WindDirection']
    clean_data = clean_data.drop(temp_removals, axis=1)
    clean_data['OffenseFormation'].fillna(clean_data['OffenseFormation'].mode()[0], inplace=True)
    clean_data['DefendersInTheBox'].fillna(clean_data['DefendersInTheBox'].mode()[0], inplace=True)
    # clean_data.to_csv(file_name, index = False)
    return(clean_data)

if __name__ == "__main__":
    dataset = pd.read_csv("datasets/train.csv")
    dataset = clean_reformat(dataset)
    dataset = group_feature(dataset)
    dataset = engineer_feature(dataset)
    dataset = impute_feature(dataset)
    # reduce_feature_save(dataset, 'datasets/clean_v1.csv')