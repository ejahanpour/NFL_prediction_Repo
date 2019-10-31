import pandas as pd
import numpy as np
from tqdm import tqdm
from data_cleaning import clean_reformat
import warnings
warnings.filterwarnings('ignore')


# helper function
def change_pos(spatial_data, change_pos, with_pos, fake_position, defense = True):
    if defense:
        cols = ['_defense_mean_distance', '_defense_min_distance', '_defense_pressure', '_defense_std_distance']
    else: 
        cols = ['_offense_mean_distance', '_offense_min_distance', '_offense_pressure', '_offense_std_distance']
    for col in cols:
        change_col = change_pos + col
        with_col = with_pos + col
        if change_col not in spatial_data.columns:
            spatial_data[change_col] = np.nan
        try:
            mask = (spatial_data[with_col].notnull() & spatial_data[change_col].isnull())
            spatial_data.loc[mask, change_col] = spatial_data.loc[mask, with_col]
            spatial_data.loc[mask, with_col] = np.nan
            fake_position.extend([with_pos])
        except:
            pass
    return spatial_data , fake_position

def rush_player_statistics(dataset):
    player_features = dataset[['NflId', 'PlayerHeight', 'PlayerWeight', 'PlayerBirthDate', 'Position']].drop_duplicates()
    player_rush_info = dataset.loc[dataset.NflId == dataset.NflIdRusher, ['PlayId', 'NflId', 'NflIdRusher', 'Yards']].drop_duplicates()
    player_yards = player_rush_info[['NflId', 'Yards']].groupby('NflId').mean().reset_index()
    player_features = pd.merge(player_features, player_yards, on = 'NflId', how = 'left')
    player_features['PlayerHeight'] = player_features.PlayerHeight.map(lambda x: 12.0*int(x.split('-')[0]) + int(x.split('-')[1]))
    player_features['PlayerAge'] = (pd.Timestamp('now') - pd.to_datetime(player_features['PlayerBirthDate']))//np.timedelta64(1,'Y')
    player_features.drop(['PlayerBirthDate'], axis = 1, inplace=True)
    cols_to_std = ['PlayerHeight', 'PlayerWeight', 'PlayerAge', 'Yards']
    sc = preprocessing.StandardScaler()
    player_features[cols_to_std] = sc.fit_transform(player_features[cols_to_std])
    player_features.Yards.fillna(0, inplace=True)
    return(player_features, sc)
    

####################

def group_positions(dataset):
    positions = {'OLB': 'LB', 'MLB': 'LB', 'ILB': 'LB', # lineback
            'G': 'L', 'T': 'L', 'C': 'L', 'DE':'L', 'DT':'L', 'NT':'L', 'OT': 'L', 'OG': 'L', 'DL': 'L',    # linemen
            'TE': 'R', 'WR': 'R',       # receiver
             'SS': 'B', 'FS':'B', 'CB':'B', 'RB': 'B', 'FB': 'B', 'HB': 'B', 'DB': 'B', 'SAF':'B', 'S': 'B'     # Backs
            }
    dataset = dataset.replace({'Position': positions})
    return dataset

def extract_spatial_features(dataset, unique_id):
    game_snaps = dataset[unique_id].unique()
    defense_game_shema = pd.DataFrame()
    offense_game_shema = pd.DataFrame()
    for game in game_snaps:
        game_stat = dataset[(dataset[unique_id] == game)]
        game_stat['OffenseDefence'] = 'D'
        game_stat.loc[(game_stat.HomeTeamAbbr == game_stat.PossessionTeam) & (game_stat['Team'] == 'home' ), 'OffenseDefence'] = 'O'
        game_stat.loc[(game_stat.VisitorTeamAbbr == game_stat.PossessionTeam) & (game_stat['Team'] == 'away' ), 'OffenseDefence'] = 'O'
        offense_stat = game_stat.loc[game_stat.OffenseDefence == 'O', [unique_id, 'X', 'Y', 'Position', 'NflId']]
        defense_stat = game_stat.loc[game_stat.OffenseDefence == 'D', [unique_id, 'X', 'Y', 'Position', 'NflId']]
        # make the Position unique value (add the value of the occurance to the position)
        offense_stat['Position'] = offense_stat['Position'] + offense_stat.groupby("Position")['NflId'].rank(method="dense").astype(np.int).map(str)
        defense_stat['Position'] = defense_stat['Position'] + defense_stat.groupby("Position")['NflId'].rank(method="dense").astype(np.int).map(str)
        offense_x = np.tile(np.array(offense_stat.X), (11, 1))
        defense_x = np.tile(np.array(defense_stat.X), (11, 1)).transpose()
        d_o_x_distance = (offense_x - defense_x)**2     # The distance along x-axis between defense row+1 and offense col+1
        offense_y = np.tile(np.array(offense_stat.Y), (11, 1))
        defense_y = np.tile(np.array(defense_stat.Y), (11, 1)).transpose()
        d_o_y_distance = (offense_y - defense_y)**2     # The distance along y-axis between defense row+1 and offense col+1
        D_O_Distance = np.sqrt(d_o_x_distance + d_o_y_distance)
        defense_mean_dist_to_others = np.mean(D_O_Distance, axis=0)
        offense_mean_dist_to_others = np.mean(D_O_Distance, axis=1)
        defense_std_dist_to_others = np.std(D_O_Distance, axis=0)
        offense_std_dist_to_others = np.std(D_O_Distance, axis=1)
        defense_min_dist_to_others = np.min(D_O_Distance, axis=0)
        offense_min_dist_to_others = np.min(D_O_Distance, axis=1)
        defense_compact = np.percentile(D_O_Distance, 20, axis=0)   # The 3rd closest player distance to the defense
        offense_compact = np.percentile(D_O_Distance, 20, axis=1)
        #np.sqrt(d_o_x_distance)[0] is the distance of offense player q from different defense players
        defense_schema = pd.DataFrame({'GameSnap' : defense_stat[unique_id], 'Position': defense_stat.Position, 
                               'defense_mean_distance': defense_mean_dist_to_others, 'defense_std_distance': defense_std_dist_to_others, 
                               'defense_min_distance': defense_min_dist_to_others, 'defense_pressure': defense_compact})
        game_snap_datafame = defense_schema.pivot(index = unique_id,
                            columns= 'Position',
                            values = ['defense_mean_distance', 'defense_std_distance', 'defense_min_distance', 'defense_pressure']).reset_index()
        game_snap_datafame.columns = [d + '_' + c for c, d in game_snap_datafame.columns]
        defense_game_shema = defense_game_shema.append(game_snap_datafame)
        ## offense team
        offense_schema = pd.DataFrame({unique_id : offense_stat[unique_id], 'Position': offense_stat.Position, 
                               'offense_mean_distance': offense_mean_dist_to_others, 'offense_std_distance': offense_std_dist_to_others, 
                               'offense_min_distance': offense_min_dist_to_others, 'offense_pressure': offense_compact})
        game_snap_datafame = offense_schema.pivot(index = unique_id,
                            columns= 'Position',
                            values = ['offense_mean_distance', 'offense_std_distance', 'offense_min_distance', 'offense_pressure']).reset_index()
        game_snap_datafame.columns = [d + '_' + c for c, d in game_snap_datafame.columns]
        offense_game_shema = offense_game_shema.append(game_snap_datafame)
    game_spatial_features = defense_game_shema.merge(offense_game_shema, on = '_GameSnap', how = 'inner' )
    return game_spatial_features

def tweek_positions_prior_knowledge(spatial_data):
    ###### Defense Update
    # update back position
    fake_position = []
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='B2', with_pos = 'LB4', fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='B3', with_pos= 'LB4', fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='B4', with_pos = 'LB4', fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='B2', with_pos= 'LB5', fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='B3', with_pos = 'LB5', fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='B4', with_pos= 'LB5', fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='B2', with_pos= 'LB6', fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='B3', with_pos = 'LB6', fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='B4', with_pos= 'LB6', fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='B2', with_pos = 'R1', fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='B3', with_pos= 'R1', fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='B4', with_pos = 'R1', fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='B2', with_pos = 'L5', fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='B3', with_pos= 'L5', fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='B4', with_pos = 'L5', fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='B2', with_pos = 'L6', fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='B3', with_pos= 'L6', fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='B4', with_pos = 'L6', fake_position = fake_position)
    # update the lineman list
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='L1', with_pos = 'B5', fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='L2', with_pos= 'B5', fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='L3', with_pos = 'B5', fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='L4', with_pos= 'B5', fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='L1', with_pos = 'B6', fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='L2', with_pos= 'B6', fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='L3', with_pos = 'B6', fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='L4', with_pos= 'B6', fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='L1', with_pos = 'B7', fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='L2', with_pos= 'B7', fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='L3', with_pos = 'B7', fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='L4', with_pos= 'B7', fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='L1', with_pos = 'B8', fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='L2', with_pos= 'B8', fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='L3', with_pos = 'B8', fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='L4', with_pos= 'B8', fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='L1', with_pos = 'LB4', fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='L2', with_pos= 'LB4', fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='L3', with_pos = 'LB4', fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='L4', with_pos= 'LB4', fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='L1', with_pos = 'LB5', fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='L2', with_pos= 'LB5', fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='L3', with_pos = 'LB5', fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='L4', with_pos= 'LB5', fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='L1', with_pos = 'LB6', fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='L2', with_pos= 'LB6', fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='L3', with_pos = 'LB6', fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='L4', with_pos= 'LB6', fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='L1', with_pos = 'R1', fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='L2', with_pos= 'R1', fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='L3', with_pos = 'R1', fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='L4', with_pos = 'R1', fake_position = fake_position)
    # update lineback
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='LB1', with_pos = 'B5', fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='LB2', with_pos= 'B5', fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='LB3', with_pos = 'B5', fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='LB1', with_pos = 'B6', fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='LB2', with_pos= 'B6', fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='LB3', with_pos = 'B6', fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='LB1', with_pos = 'B7', fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='LB2', with_pos= 'B7', fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='LB3', with_pos = 'B7', fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='LB1', with_pos = 'B8', fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='LB2', with_pos= 'B8', fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='LB3', with_pos = 'B8', fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='LB1', with_pos = 'L5', fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='LB2', with_pos= 'L5', fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='LB3', with_pos = 'L5', fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='LB1', with_pos = 'L6', fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='LB2', with_pos= 'L6', fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='LB3', with_pos = 'L6', fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='LB1', with_pos = 'R1', fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='LB2', with_pos= 'R1', fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='LB3', with_pos = 'R1', fake_position = fake_position)
    ### Offense Update
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='R1', with_pos = 'B1', defense = False, fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='R2', with_pos = 'B1', defense = False, fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='R3', with_pos= 'B1', defense = False, fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='R4', with_pos = 'B1', defense = False, fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='R5', with_pos= 'B1', defense = False, fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='R1', with_pos = 'B2', defense = False, fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='R2', with_pos = 'B2', defense = False, fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='R3', with_pos= 'B2', defense = False, fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='R4', with_pos = 'B2', defense = False, fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='R5', with_pos= 'B2', defense = False, fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='R1', with_pos = 'B3', defense = False, fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='R2', with_pos = 'B3', defense = False, fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='R3', with_pos= 'B3', defense = False, fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='R4', with_pos = 'B3', defense = False, fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='R5', with_pos= 'B3', defense = False, fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='R1', with_pos = 'QB2', defense = False, fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='R2', with_pos = 'QB2', defense = False, fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='R3', with_pos= 'QB2', defense = False, fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='R4', with_pos = 'QB2', defense = False, fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='R5', with_pos= 'QB2', defense = False, fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='R1', with_pos = 'L6', defense = False, fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='R2', with_pos = 'L6', defense = False, fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='R3', with_pos= 'L6', defense = False, fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='R4', with_pos = 'L6', defense = False, fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='R5', with_pos= 'L6', defense = False, fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='R1', with_pos = 'L7', defense = False, fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='R2', with_pos = 'L7', defense = False, fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='R3', with_pos= 'L7', defense = False, fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='R4', with_pos = 'L7', defense = False, fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='R5', with_pos= 'L7', defense = False, fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='L4', with_pos = 'B1', defense = False, fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='L4', with_pos= 'B2', defense = False, fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='L4', with_pos = 'B3', defense = False, fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='L4', with_pos = 'LB1', defense = False, fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='L5', with_pos = 'B1', defense = False, fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='L5', with_pos= 'B2', defense = False, fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='L5', with_pos = 'B3', defense = False, fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='L5', with_pos = 'LB1', defense = False, fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='R1', with_pos = 'LB1', defense = False, fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='R2', with_pos= 'LB1', defense = False, fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='R3', with_pos = 'LB1', defense = False, fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='R4', with_pos = 'LB1', defense = False, fake_position = fake_position)
    spatial_data , fake_position = change_pos(spatial_data = spatial_data, change_pos='R5', with_pos = 'LB1', defense = False, fake_position = fake_position)
    fake_positions = ['B5_defense_mean_distance', 'B5_defense_min_distance',
       'B5_defense_pressure', 'B5_defense_std_distance',
       'B6_defense_mean_distance', 'B6_defense_min_distance',
       'B6_defense_pressure', 'B6_defense_std_distance',
       'B7_defense_mean_distance', 'B7_defense_min_distance',
       'B7_defense_pressure', 'B7_defense_std_distance',
       'B8_defense_mean_distance', 'B8_defense_min_distance',
       'B8_defense_pressure', 'B8_defense_std_distance',
       'L5_defense_mean_distance', 'L5_defense_min_distance',
       'L5_defense_pressure', 'L5_defense_std_distance',
       'L6_defense_mean_distance', 'L6_defense_min_distance',
       'L6_defense_pressure', 'L6_defense_std_distance',
       'LB4_defense_mean_distance', 'LB4_defense_min_distance',
       'LB4_defense_pressure', 'LB4_defense_std_distance',
       'LB5_defense_mean_distance', 'LB5_defense_min_distance',
       'LB5_defense_pressure', 'LB5_defense_std_distance',
       'LB6_defense_mean_distance', 'LB6_defense_min_distance',
       'LB6_defense_pressure', 'LB6_defense_std_distance',
       'R1_defense_mean_distance', 'R1_defense_min_distance',
       'R1_defense_pressure', 'R1_defense_std_distance',
       'B1_offense_mean_distance', 'B1_offense_min_distance',
       'B1_offense_pressure', 'B1_offense_std_distance',
       'B2_offense_mean_distance', 'B2_offense_min_distance',
       'B2_offense_pressure', 'B2_offense_std_distance',
       'B3_offense_mean_distance', 'B3_offense_min_distance',
       'B3_offense_pressure', 'B3_offense_std_distance',
       'L6_offense_mean_distance', 'L6_offense_min_distance',
       'L6_offense_pressure', 'L6_offense_std_distance',
       'L7_offense_mean_distance', 'L7_offense_min_distance',
       'L7_offense_pressure', 'L7_offense_std_distance',
       'LB1_offense_mean_distance', 'LB1_offense_min_distance',
       'LB1_offense_pressure', 'LB1_offense_std_distance',
       'QB2_offense_mean_distance', 'QB2_offense_min_distance',
       'QB2_offense_pressure', 'QB2_offense_std_distance']
    fake_pos_available = [i for i in spatial_data.columns if i in fake_positions]
    spatial_data = spatial_data.drop(fake_pos_available, axis = 1)
    return spatial_data


if __name__ == "__main__":
    dataset = pd.read_csv("datasets/train.csv")
    dataset = clean_reformat(dataset)
    dataset = group_positions(dataset)
    dataset = extract_spatial_features(dataset, unique_id = 'GameSnap')
    dataset = tweek_positions_prior_knowledge(dataset)
    # dataset.to_csv('datasets/spatial_ds_v1.csv', index = False ) 