import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_squared_error
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

clean_data = pd.read_csv('datasets/clean_v1.csv')
spatial_dataset = pd.read_csv('datasets/spatial_ds_v1.csv')
dataset = clean_data.join(spatial_dataset)
dataset['GameId'] = dataset['GameSnap'].str[10:]
dataset = dataset[dataset.QB1_offense_mean_distance.notnull()][:1000]

train, test = train_test_split(dataset, test_size = 0.3, random_state = 123)
x_train = train.drop(['Yards', 'GameSnap'], axis = 1)
x_test = test.drop(['Yards', 'GameSnap'], axis = 1)
y_train = train['Yards']
y_test = test['Yards']
game_id = x_test['GameId']

x_test = x_test.drop(['GameId', '_GameSnap'], axis = 1)
x_train = x_train.drop(['GameId', '_GameSnap'], axis = 1)


# finding the categorical variables
cat_features = x_train.select_dtypes(include = ['object']).columns
# Ordinal encoding of the categorical variables
enc = OrdinalEncoder()
enc.fit(dataset[cat_features])
x_train[cat_features] = enc.transform(x_train[cat_features])
# Random Forrest model
RF_model = RandomForestRegressor()
RF_model.fit(x_train, y_train)

# predict and evaluate
x_test[cat_features] = enc.transform(x_test[cat_features])
y_pred = RF_model.predict(x_test)
pre_submission = pd.DataFrame({'GameId': game_id, 'Preds':y_pred})
pre_submission.to_csv('datasets/pre_submission.csv', index = False)
submission = pd.DataFrame(0, index = pre_submission.GameId.unique(), columns = np.arange(-99, 100))
for i, rows in pre_submission.iterrows():
    submission.iloc[submission.index == rows['GameId'], int(98 + np.ceil(rows['Preds']))] = 0.5
    submission.iloc[submission.index == rows['GameId'], int(99 + np.ceil(rows['Preds'])):] = 1
submission.to_csv('datasets/post_submission.csv') 
print(mean_squared_error(y_pred, y_test))

