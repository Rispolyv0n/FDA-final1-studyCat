import datetime
import logging
import numbers
import random
import math
import sys
from collections import defaultdict

import pandas as pd
import numpy as np

import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error

import func.get_data as getData

def train_model(model, train_x, train_y, test_x, test_y, model_name):
    logging.info('Training model - ' + model_name + '...')
    model.fit(train_x, train_y)
    train_pred = model.predict(train_x)
    test_pred = model.predict(test_x)
    logging.info('Training rmse: %.8f' % math.sqrt(mean_squared_error(train_y, train_pred)))
    logging.info('Testing rmse: %.8f' % math.sqrt(mean_squared_error(test_y, test_pred)))
    return

def grid_search_for_models(model, param, name, train_x, train_y):
    logging.info('Start grid searching - ' + name)
    gs = GridSearchCV(estimator=model,  
                        param_grid=param,
                        scoring='neg_mean_squared_error',
                        cv=5,
                        n_jobs=2,
                        verbose=2)
    gs.fit(train_x, train_y)
    print(gs.best_params_)
    print(gs.best_score_)
    logging.info('=====================================')
    return

# Initializing
logging.basicConfig(level=logging.INFO,
            format='\n%(asctime)s %(name)-5s === %(levelname)-5s === %(message)s\n')
logging.info('Initializing...')
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.expand_frame_repr', False)

data_path = './data/data.json'
train_test_ratio = 0.85


# Reading data
logging.info('Reading data...')

acc_data = getData.load(path=data_path, just_accs=True)
data = getData.load(path=data_path, just_accs=False)

print(data[0])

logging.info('Data length(all): %d' % len(data))
logging.info('Data length(only acc): %d' % len(acc_data))


# Shuffle
logging.info('Shuffling & split train / test data.')
train_acc_data, test_acc_data = getData.split_train_test_by_user(acc_data, train_test_ratio)
logging.info('Training data length: %d' % len(train_acc_data))
logging.info('Testing data length: %d' % len(test_acc_data))

print(train_acc_data[0])


# Calculating personal data
logging.info('Calculating personal features...')
res = getData.get_personal_data(data,
        count_of_test = True,
        count_of_learn = True,
        count_of_exp = True,
        hours_of_use = True,
        mean_response_time = True,
        mean_learning_time = True,
        freq_all = True,
        freq_duration = True,
        accuracy_after_exposure = True,
        mean_accuracy = True,
        mean_accuracy_each_scoring_model = True,
        mean_accuracy_each_unit=True,
        mean_accuracy_each_unit_module = True,
        school_id = True,
        # acc of different question categories
        mean_accuracy_spot = False,
        mean_accuracy_numbers = False,
        mean_accuracy_phonics = False,
        mean_accuracy_phonemes = False,
        mean_accuracy_singplu = False,
        mean_accuracy_letters = False,
        mean_accuracy_abc = False,
        mean_accuracy_sight = False,
        mean_accuracy_others = False,
        # resp time of different score models / units / unit modules
        mean_resp_time_unit = True,
        mean_resp_time_unit_module = True,
        mean_resp_time_scoring_model = True
    )


logging.info('Transforming features -> pandas.DataFrame')
res_df = pd.DataFrame(data={
    'test_count':[x['test_count'] for x in res],
    'learn_count':[x['learn_count'] for x in res],
    'exp_count':[x['exp_count'] for x in res],
    'hours_use':[x['hours_use'] for x in res],
    'mean_learn_time':[x['mean_learn_time'] for x in res],
    'mean_resp_time':[x['mean_resp_time'] for x in res],
    'freq_all':[x['freq_all'] for x in res],
    'freq_duration':[x['freq_duration'] for x in res],
    'mean_acc':[x['mean_acc'] for x in res],
    'acc_exposure':[x['acc_exposure'] for x in res],
    'mean_acc_score_model':[x['mean_acc_score_model'] for x in res],
    'mean_acc_unit':[x['mean_acc_unit'] for x in res],
    'mean_acc_unit_module':[x['mean_acc_unit_module'] for x in res],
    'school_id':[x['school_id'] for x in res],
    'mean_resp_score_model':[x['mean_resp_score_model'] for x in res],
    'mean_resp_unit':[x['mean_resp_unit'] for x in res],
    'mean_resp_unit_module':[x['mean_resp_unit_module'] for x in res]
    })

temp_df = res_df[(res_df['exp_count'] > 0) & (res_df['mean_acc']>=0)]
temp_df['learn_ratio'] = temp_df.apply(lambda row: row['learn_count'] / row['exp_count'], axis=1)


# # filter
# temp_df = temp_df[(temp_df['mean_resp_time']>=0) & (temp_df['mean_learn_time']>=0)]
# temp_df = temp_df[(temp_df['freq_all']>=0) & (temp_df['freq_duration']>=0)]
# temp_df = temp_df[(temp_df['acc_exposure']>0)]

# bin cut
features = [
    'test_count',
    'learn_count',
    'exp_count',
    'hours_use',
    'mean_learn_time',
    'mean_resp_time',
    'freq_all',
    'freq_duration',
    'acc_exposure',
    'learn_ratio'
    ]
labels = [x for x in range(10)]
for feat_name in features:
    temp_df['cut_'+feat_name] = pd.cut(temp_df[feat_name], 10, labels = labels)

print(temp_df.shape)
print(temp_df.head(3))


# calculate question-based data

logging.info('Calculating question-based features...')
question_res = getData.get_question_data(data)
print(len(question_res['resp_each_scoring_model']))
print(len(question_res['resp_each_unit']))
print(len(question_res['resp_each_unit_module']))
print(len(question_res['acc_each_scoring_model']))
print(len(question_res['acc_each_unit']))
print(len(question_res['acc_each_unit_module']))


# append features - training data
logging.info('Appending features to training data...')
c = 0
spaceNum = 10000
for record in train_acc_data:
    cur_userId = record['user']
    if(cur_userId in temp_df.index):
        row = temp_df.loc[cur_userId]
        # append new personal features
        record['test_count'] = row['test_count']
        record['learn_count'] = row['learn_count']
        record['exp_count'] = row['exp_count']
        record['hours_use'] = row['hours_use']
        record['mean_learn_time'] = row['mean_learn_time']
        record['mean_resp_time'] = row['mean_resp_time']
        record['freq_all'] = row['freq_all']
        record['freq_duration'] = row['freq_duration']
        record['mean_acc'] = row['mean_acc']
        record['acc_exposure'] = row['acc_exposure']
        record['learn_ratio'] = row['learn_ratio']
        record['school_id'] = row['school_id']
        record['mean_acc_score_model'] = row['mean_acc_score_model'][record['scoring_model']]
        record['mean_acc_unit'] = row['mean_acc_unit'][record['unit']]
        record['mean_acc_unit_module'] = row['mean_acc_unit_module'][record['unit_module']]
        record['mean_resp_score_model'] = row['mean_resp_score_model'][record['scoring_model']]
        record['mean_resp_unit'] = row['mean_resp_unit'][record['unit']]
        record['mean_resp_unit_module'] = row['mean_resp_unit_module'][record['unit_module']]
        for feat_name in features:
            record['cut_'+feat_name] = row['cut_'+feat_name]
        # append new question-based features
        record['resp_each_scoring_model'] = question_res['resp_each_scoring_model'][record['scoring_model']]
        record['resp_each_unit'] = question_res['resp_each_unit'][record['unit']]
        record['resp_each_unit_module'] = question_res['resp_each_unit_module'][record['unit_module']]
        record['acc_each_scoring_model'] = question_res['acc_each_scoring_model'][record['scoring_model']]
        record['acc_each_unit'] = question_res['acc_each_unit'][record['unit']]
        record['acc_each_unit_module'] = question_res['acc_each_unit_module'][record['unit_module']]
    c+=1
    if(c%spaceNum==0):
        logging.info('appending data... %d / %d done' % (c, len(train_acc_data)) )


logging.info('Converting training dict to dataframe & filter features.')

# if no exp -> cut_test_count = 0
# train_acc_data_with_exp = list(filter(lambda x: 'test_count' in x.keys(), train_acc_data))

train_df = pd.DataFrame(data={
    'test_count':[x['test_count'] for x in train_acc_data],
    'learn_count':[x['learn_count'] for x in train_acc_data],
    'exp_count':[x['exp_count'] for x in train_acc_data],
    'hours_use':[x['hours_use'] for x in train_acc_data],
    'mean_learn_time':[x['mean_learn_time'] for x in train_acc_data],
    'mean_resp_time':[x['mean_resp_time'] for x in train_acc_data],
    'freq_all':[x['freq_all'] for x in train_acc_data],
    'freq_duration':[x['freq_duration'] for x in train_acc_data],
    'mean_acc':[x['mean_acc'] for x in train_acc_data],
    'acc_exposure':[x['acc_exposure'] for x in train_acc_data],
    'learn_ratio':[x['learn_ratio'] for x in train_acc_data], # too similar with other columns
    'accuracy':[x['accuracy'] for x in train_acc_data],
    'cut_test_count':[x['cut_test_count'] for x in train_acc_data], # too similar with other columns
    'cut_learn_count':[x['cut_learn_count'] for x in train_acc_data], # too similar with other columns
    'cut_exp_count':[x['cut_exp_count'] for x in train_acc_data], # too similar with other columns
    'cut_hours_use':[x['cut_hours_use'] for x in train_acc_data],
    'cut_mean_learn_time':[x['cut_mean_learn_time'] for x in train_acc_data],
    'cut_mean_resp_time':[x['cut_mean_resp_time'] for x in train_acc_data],
    'cut_freq_all':[x['cut_freq_all'] for x in train_acc_data],
    'cut_freq_duration':[x['cut_freq_duration'] for x in train_acc_data],
    'cut_acc_exposure':[x['cut_acc_exposure'] for x in train_acc_data],
    'cut_learn_ratio':[x['cut_learn_ratio'] for x in train_acc_data],
    'school_id':[x['school_id'] for x in train_acc_data],
    'is_preview':[x['is_preview'] for x in train_acc_data],
    'unit_module':[x['unit_module'] for x in train_acc_data],
    'scoring_model':[x['scoring_model'] for x in train_acc_data],
    'level':[x['level'] for x in train_acc_data],
    'teacher':[x['teacher'] for x in train_acc_data],
    'class':[x['class'] for x in train_acc_data],
    'mean_acc_score_model':[x['mean_acc_score_model'] for x in train_acc_data],
    'mean_acc_unit':[x['mean_acc_unit'] for x in train_acc_data],
    'mean_acc_unit_module':[x['mean_acc_unit_module'] for x in train_acc_data],
    'mean_resp_score_model':[x['mean_resp_score_model'] for x in train_acc_data],
    'mean_resp_unit':[x['mean_resp_unit'] for x in train_acc_data],
    'mean_resp_unit_module':[x['mean_resp_unit_module'] for x in train_acc_data],
    'resp_each_scoring_model':[x['resp_each_scoring_model'] for x in train_acc_data],
    'resp_each_unit':[x['resp_each_unit'] for x in train_acc_data],
    'resp_each_unit_module':[x['resp_each_unit_module'] for x in train_acc_data],
    'acc_each_scoring_model':[x['acc_each_scoring_model'] for x in train_acc_data],
    'acc_each_unit':[x['acc_each_unit'] for x in train_acc_data],
    'acc_each_unit_module':[x['acc_each_unit_module'] for x in train_acc_data]
    })

print(train_df.shape)
print(train_df.head(3))

logging.info('Drop rows with no value.')
train_df.dropna(inplace=True)

print(train_df.shape)
print(train_df.head(3))


# append features - testing data
logging.info('Appending features to testing data...')
c = 0
spaceNum = 10000
for record in test_acc_data:
    cur_userId = record['user']
    if(cur_userId in temp_df.index):
        row = temp_df.loc[cur_userId]
        # append new features
        record['test_count'] = row['test_count']
        record['learn_count'] = row['learn_count']
        record['exp_count'] = row['exp_count']
        record['hours_use'] = row['hours_use']
        record['mean_learn_time'] = row['mean_learn_time']
        record['mean_resp_time'] = row['mean_resp_time']
        record['freq_all'] = row['freq_all']
        record['freq_duration'] = row['freq_duration']
        record['mean_acc'] = row['mean_acc']
        record['acc_exposure'] = row['acc_exposure']
        record['learn_ratio'] = row['learn_ratio']
        record['school_id'] = row['school_id']
        record['mean_acc_score_model'] = row['mean_acc_score_model'][record['scoring_model']]
        record['mean_acc_unit'] = row['mean_acc_unit'][record['unit']]
        record['mean_acc_unit_module'] = row['mean_acc_unit_module'][record['unit_module']]
        record['mean_resp_score_model'] = row['mean_resp_score_model'][record['scoring_model']]
        record['mean_resp_unit'] = row['mean_resp_unit'][record['unit']]
        record['mean_resp_unit_module'] = row['mean_resp_unit_module'][record['unit_module']]
        for feat_name in features:
            record['cut_'+feat_name] = row['cut_'+feat_name]
        # append new question-based features
        record['resp_each_scoring_model'] = question_res['resp_each_scoring_model'][record['scoring_model']]
        record['resp_each_unit'] = question_res['resp_each_unit'][record['unit']]
        record['resp_each_unit_module'] = question_res['resp_each_unit_module'][record['unit_module']]
        record['acc_each_scoring_model'] = question_res['acc_each_scoring_model'][record['scoring_model']]
        record['acc_each_unit'] = question_res['acc_each_unit'][record['unit']]
        record['acc_each_unit_module'] = question_res['acc_each_unit_module'][record['unit_module']]
    c+=1
    if(c%spaceNum==0):
        logging.info('appending data... %d / %d done' % (c, len(test_acc_data)) )

logging.info('Converting testing dict to dataframe & filter features.')

# if no exp -> cut_test_count = 0
# test_acc_data_with_exp = list(filter(lambda x: 'test_count' in x.keys(), test_acc_data))

test_df = pd.DataFrame(data={
    'test_count':[x['test_count'] for x in test_acc_data],                    # too similar with other columns
    'learn_count':[x['learn_count'] for x in test_acc_data],                  # too similar with other columns
    'exp_count':[x['exp_count'] for x in test_acc_data],                      # use cut
    'hours_use':[x['hours_use'] for x in test_acc_data],                      # use cut
    'mean_learn_time':[x['mean_learn_time'] for x in test_acc_data],          # use cut
    'mean_resp_time':[x['mean_resp_time'] for x in test_acc_data],            # use cut
    'freq_all':[x['freq_all'] for x in test_acc_data],                        # use cut
    'freq_duration':[x['freq_duration'] for x in test_acc_data],              # use cut
    'mean_acc':[x['mean_acc'] for x in test_acc_data],
    'acc_exposure':[x['acc_exposure'] for x in test_acc_data],
    'learn_ratio':[x['learn_ratio'] for x in test_acc_data],
    'accuracy':[x['accuracy'] for x in test_acc_data],
    'cut_test_count':[x['cut_test_count'] for x in test_acc_data],            # too similar with other columns
    'cut_learn_count':[x['cut_learn_count'] for x in test_acc_data],          # too similar with other columns
    'cut_exp_count':[x['cut_exp_count'] for x in test_acc_data],
    'cut_hours_use':[x['cut_hours_use'] for x in test_acc_data],
    'cut_mean_learn_time':[x['cut_mean_learn_time'] for x in test_acc_data],
    'cut_mean_resp_time':[x['cut_mean_resp_time'] for x in test_acc_data],
    'cut_freq_all':[x['cut_freq_all'] for x in test_acc_data],
    'cut_freq_duration':[x['cut_freq_duration'] for x in test_acc_data],
    'cut_acc_exposure':[x['cut_acc_exposure'] for x in test_acc_data],        # use original
    'cut_learn_ratio':[x['cut_learn_ratio'] for x in test_acc_data],          # use original
    'school_id':[x['school_id'] for x in test_acc_data],
    'is_preview':[x['is_preview'] for x in test_acc_data],
    'unit_module':[x['unit_module'] for x in test_acc_data],
    'scoring_model':[x['scoring_model'] for x in test_acc_data],
    'level':[x['level'] for x in test_acc_data],
    'teacher':[x['teacher'] for x in test_acc_data],
    'class':[x['class'] for x in test_acc_data],
    'mean_acc_score_model':[x['mean_acc_score_model'] for x in test_acc_data],
    'mean_acc_unit':[x['mean_acc_unit'] for x in test_acc_data],
    'mean_acc_unit_module':[x['mean_acc_unit_module'] for x in test_acc_data],
    'mean_resp_score_model':[x['mean_resp_score_model'] for x in test_acc_data],
    'mean_resp_unit':[x['mean_resp_unit'] for x in test_acc_data],
    'mean_resp_unit_module':[x['mean_resp_unit_module'] for x in test_acc_data],
    'resp_each_scoring_model':[x['resp_each_scoring_model'] for x in test_acc_data],
    'resp_each_unit':[x['resp_each_unit'] for x in test_acc_data],
    'resp_each_unit_module':[x['resp_each_unit_module'] for x in test_acc_data],
    'acc_each_scoring_model':[x['acc_each_scoring_model'] for x in test_acc_data],
    'acc_each_unit':[x['acc_each_unit'] for x in test_acc_data],
    'acc_each_unit_module':[x['acc_each_unit_module'] for x in test_acc_data]
    })

print(test_df.shape)
print(test_df.head(3))

logging.info('Drop rows with no value.')
test_df.dropna(inplace=True)

print(test_df.shape)
print(test_df.head(3))


# Convert 'is_preview' bool(true/false) to int(1, 0)
logging.info('Convert bool to int.')
train_df['is_preview'] = train_df['is_preview'].astype(int)
test_df['is_preview'] = test_df['is_preview'].astype(int)


# Final features...
logging.info('Final features:')

# mse 0.0303
feature_list = [
    # 'test_count',
    # 'learn_count',
    # 'exp_count',
    # 'hours_use',
    # 'mean_learn_time',
    # 'mean_resp_time',
    # 'freq_all',
    # 'freq_duration',
    'mean_acc',
    'acc_exposure', 
    'learn_ratio',
    # 'cut_test_count',
    # 'cut_learn_count',
    'cut_exp_count',
    'cut_hours_use',
    'cut_mean_learn_time',
    'cut_mean_resp_time',
    'cut_freq_all', 
    'cut_freq_duration', 
    # 'cut_acc_exposure', 
    # 'cut_learn_ratio',
    'school_id',
    'is_preview',
    'unit_module',
    'scoring_model',
    'level',
    'teacher',
    'class',
    'mean_acc_score_model',
    'mean_acc_unit',
    'mean_acc_unit_module',
    'mean_resp_score_model',
    'mean_resp_unit',
    'mean_resp_unit_module',
    'resp_each_scoring_model',
    'resp_each_unit',
    'resp_each_unit_module',
    'acc_each_scoring_model',
    'acc_each_unit',
    'acc_each_unit_module'
]

target_feature_name = 'accuracy'
print(feature_list)
print(len(feature_list))
print(train_df[feature_list].head(3))


# Feature scaling

# logging.info('Feature scaling.')
# scaled_feature_list = [
#     'mean_resp_score_model',
#     'mean_resp_unit',
#     'mean_resp_unit_module',
#     'resp_each_scoring_model',
#     'resp_each_unit',
#     'resp_each_unit_module'
# ]
# scaler = StandardScaler()
# scaler.fit(train_df[scaled_feature_list])
# train_df[scaled_feature_list] = scaler.transform(train_df[scaled_feature_list])
# test_df[scaled_feature_list] = scaler.transform(test_df[scaled_feature_list])
# print(train_df.head(3))
# print(train_df.shape)
# print(test_df.shape)


# Grid Search !

model_lr = LinearRegression()
model_rf = RandomForestRegressor(criterion='mse')
model_ada = AdaBoostRegressor()
model_gb = GradientBoostingRegressor(criterion='friedman_mse')
model_xgb = xgb.XGBRegressor()

model_info = [
    {
        'name': 'linear regression', # mse 0.0225
        'model': model_lr,
        'param': {}
    },
    {
        'name': 'adaboost', # mse 0.0246
        'model': model_ada,
        'param': {}
    },
    {
        'name': 'gradient boosting regressor', # mse 0.0220
        'model': model_gb,
        'param': {
            'n_estimators': [200, 300, 500], # 200
            'max_depth': [3, 5], # 3
        }
    },
    {
        'name': 'xgboost regressor', # mse 0.0220
        'model': model_xgb,
        'param': {
            'n_estimators': [200, 300, 500], # 200
            'booster': ['dart'], # dart
            'gamma': [0.0001, 0.00001], # 0.0001
        }
    },
    {
        'name': 'random forest', # mse 0.0222
        'model': model_rf,
        'param': {
            'n_estimators': [300, 500, 800], # 300
            'max_depth': [5, 8, 10], # 8
        }
    },
]

train_x = train_df[feature_list]
train_y = train_df[target_feature_name]
test_x = test_df[feature_list]
test_y = test_df[target_feature_name]

for info in model_info:
    grid_search_for_models(
        model = info['model'],
        param = info['param'],
        name = info['name'],
        train_x = train_x,
        train_y = train_y
    )




