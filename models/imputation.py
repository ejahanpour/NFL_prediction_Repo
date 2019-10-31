import pandas as pd
import numpy as np

""" sample_encoder_object
    encoder.categories: [array(['BUF', 'NYJ'], dtype=object), array(['I_FORM', 'JUMBO', 'SHOTGUN', 'SINGLEBACK'], dtype=object), 
    array(['left', 'right'], dtype=object), array(['BUF'], dtype=object), array(['NYJ'], dtype=object), 
    array(['A-Turf Titan'], dtype=object), array(['first'], dtype=object), array(['other', 'own'], dtype=object)]
    list(cat_features): ['PossessionTeam', 'OffenseFormation', 'PlayDirection', 'HomeTeamAbbr', 'VisitorTeamAbbr', 'Turf', 'PreviousGameStat', 'FieldSide']
"""
def impute_categories(dataset, cat_features, encoder_list):
    for i, cat in enumerate(cat_features):
        temp = {k:v for v, k in enumerate(encoder_list[i])}
        dataset[cat] = dataset[cat].map(temp).fillna(100).astype(int)
    # print(dataset.head(1).transpose())
    return(dataset)



if __name__ == "__main__":
    test_ds = pd.read_csv('datasets/split_test_30.csv')
    encoder_list = [np.array(['BUF', 'NYJ'], dtype=object), np.array(['I_FORM', 'JUMBO', 'SHOTGUN', 'SINGLEBACK'], dtype=object), \
    np.array(['left', 'right'], dtype=object), np.array(['BUF'], dtype=object), np.array(['NYJ'], dtype=object), \
    np.array(['A-Turf Titan'], dtype=object), np.array(['first'], dtype=object), np.array(['other', 'own'], dtype=object)]
    cat_feature_list = ['PossessionTeam', 'OffenseFormation', 'PlayDirection', 'HomeTeamAbbr', 'VisitorTeamAbbr', 'Turf', 'PreviousGameStat', 'FieldSide']
    impute_categories(test_ds, cat_feature_list, encoder_list)