import datetime
import logging
import numbers
import random
import sys
from collections import defaultdict

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error

import func.get_data as getData


# Initializing
logging.basicConfig(level=logging.DEBUG,
            format='\n%(asctime)s %(name)-5s === %(levelname)-5s === %(message)s\n')
logging.info('Initializing...')
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.expand_frame_repr', False)

data_path = './data/data.json'
train_test_ratio = 0.7


# Reading data
logging.info('Reading data...')

acc_data = getData.load(path=data_path, just_accs=True)
data = getData.load(path=data_path, just_accs=False)

print(data[0])

logging.info('Finish Reading data.')
logging.info('Data length(all): %d' % len(data))
logging.info('Data length(only acc): %d' % len(acc_data))


# Shuffle
train_acc_data, test_acc_data = getData.split_train_test(acc_data, train_test_ratio)


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
        school_id = True)


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
    'school_id':[x['school_id'] for x in res],
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

# append features - training data
logging.info('Appending features to training data...')
c = 0
spaceNum = 1000
for record in train_acc_data:
    cur_userId = record['user']
    if(cur_userId in temp_df.index):
        row = temp_df.loc[cur_userId]
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
        record['mean_acc_score_model'] = row['mean_acc_score_model']
        record['is_preview'] = row['is_preview']
        record['unit_module'] = row['unit_module']
        record['scoring_model'] = row['scoring_model']
        record['level'] = row['level']
        record['teacher'] = row['teacher']
        record['class'] = row['class']
        for feat_name in features:
            record['cut_'+feat_name] = row['cut_'+feat_name]
    c+=1
    if(c%spaceNum==0):
        logging.info('appending data... %d / %d done' % (c, len(train_acc_data)) )

logging.info('Appending training data done')

train_acc_data_with_exp = list(filter(lambda x: 'test_count' in x.keys(), train_acc_data))

train_df = pd.DataFrame(data={
    # 'test_count':[x['test_count'] for x in train_acc_data_with_exp],
    # 'learn_count':[x['learn_count'] for x in train_acc_data_with_exp],
    # 'exp_count':[x['exp_count'] for x in train_acc_data_with_exp],
    # 'hours_use':[x['hours_use'] for x in train_acc_data_with_exp],
    # 'mean_learn_time':[x['mean_learn_time'] for x in train_acc_data_with_exp],
    # 'mean_resp_time':[x['mean_resp_time'] for x in train_acc_data_with_exp],
    # 'freq_all':[x['freq_all'] for x in train_acc_data_with_exp],
    # 'freq_duration':[x['freq_duration'] for x in train_acc_data_with_exp],
    'mean_acc':[x['mean_acc'] for x in train_acc_data_with_exp],
    'acc_exposure':[x['acc_exposure'] for x in train_acc_data_with_exp],
    'learn_ratio':[x['learn_ratio'] for x in train_acc_data_with_exp], # too similar with other columns
    'accuracy':[x['accuracy'] for x in train_acc_data_with_exp],
    # 'cut_test_count':[x['cut_test_count'] for x in train_acc_data_with_exp], # too similar with other columns
    # 'cut_learn_count':[x['cut_learn_count'] for x in train_acc_data_with_exp], # too similar with other columns
    'cut_exp_count':[x['cut_exp_count'] for x in train_acc_data_with_exp], # too similar with other columns
    'cut_hours_use':[x['cut_hours_use'] for x in train_acc_data_with_exp],
    'cut_mean_learn_time':[x['cut_mean_learn_time'] for x in train_acc_data_with_exp],
    'cut_mean_resp_time':[x['cut_mean_resp_time'] for x in train_acc_data_with_exp],
    'cut_freq_all':[x['cut_freq_all'] for x in train_acc_data_with_exp],
    'cut_freq_duration':[x['cut_freq_duration'] for x in train_acc_data_with_exp],
    # 'cut_acc_exposure':[x['cut_acc_exposure'] for x in train_acc_data_with_exp],
    # 'cut_learn_ratio':[x['cut_learn_ratio'] for x in train_acc_data_with_exp],
    'school_id':[x['school_id'] for x in train_acc_data_with_exp],
    'is_preview':[x['is_preview'] for x in train_acc_data_with_exp],
    'unit_module':[x['unit_module'] for x in train_acc_data_with_exp],
    'scoring_model':[x['scoring_model'] for x in train_acc_data_with_exp],
    'level':[x['level'] for x in train_acc_data_with_exp],
    'teacher':[x['teacher'] for x in train_acc_data_with_exp],
    'class':[x['class'] for x in train_acc_data_with_exp],
    # 'mean_acc_score_model':[x['mean_acc_score_model'] for x in train_acc_data_with_exp] # is a list
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
spaceNum = 1000
for record in test_acc_data:
    cur_userId = record['user']
    if(cur_userId in temp_df.index):
        row = temp_df.loc[cur_userId]
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
        record['mean_acc_score_model'] = row['mean_acc_score_model']
        record['is_preview'] = row['is_preview']
        record['unit_module'] = row['unit_module']
        record['scoring_model'] = row['scoring_model']
        record['level'] = row['level']
        record['teacher'] = row['teacher']
        record['class'] = row['class']
        for feat_name in features:
            record['cut_'+feat_name] = row['cut_'+feat_name]
    c+=1
    if(c%spaceNum==0):
        logging.info('appending data... %d / %d done' % (c, len(test_acc_data)) )

logging.info('Appending testing data done')

test_acc_data_with_exp = list(filter(lambda x: 'test_count' in x.keys(), test_acc_data))

test_df = pd.DataFrame(data={
    # 'test_count':[x['test_count'] for x in test_acc_data_with_exp],                    # too similar with other columns
    # 'learn_count':[x['learn_count'] for x in test_acc_data_with_exp],                  # too similar with other columns
    # 'exp_count':[x['exp_count'] for x in test_acc_data_with_exp],                      # use cut
    # 'hours_use':[x['hours_use'] for x in test_acc_data_with_exp],                      # use cut
    # 'mean_learn_time':[x['mean_learn_time'] for x in test_acc_data_with_exp],          # use cut
    # 'mean_resp_time':[x['mean_resp_time'] for x in test_acc_data_with_exp],            # use cut
    # 'freq_all':[x['freq_all'] for x in test_acc_data_with_exp],                        # use cut
    # 'freq_duration':[x['freq_duration'] for x in test_acc_data_with_exp],              # use cut
    'mean_acc':[x['mean_acc'] for x in test_acc_data_with_exp],
    'acc_exposure':[x['acc_exposure'] for x in test_acc_data_with_exp],
    'learn_ratio':[x['learn_ratio'] for x in test_acc_data_with_exp],
    'accuracy':[x['accuracy'] for x in test_acc_data_with_exp],
    # 'cut_test_count':[x['cut_test_count'] for x in test_acc_data_with_exp],            # too similar with other columns
    # 'cut_learn_count':[x['cut_learn_count'] for x in test_acc_data_with_exp],          # too similar with other columns
    'cut_exp_count':[x['cut_exp_count'] for x in test_acc_data_with_exp],
    'cut_hours_use':[x['cut_hours_use'] for x in test_acc_data_with_exp],
    'cut_mean_learn_time':[x['cut_mean_learn_time'] for x in test_acc_data_with_exp],
    'cut_mean_resp_time':[x['cut_mean_resp_time'] for x in test_acc_data_with_exp],
    'cut_freq_all':[x['cut_freq_all'] for x in test_acc_data_with_exp],
    'cut_freq_duration':[x['cut_freq_duration'] for x in test_acc_data_with_exp],
    # 'cut_acc_exposure':[x['cut_acc_exposure'] for x in test_acc_data_with_exp],        # use original
    # 'cut_learn_ratio':[x['cut_learn_ratio'] for x in test_acc_data_with_exp],          # use original
    'school_id':[x['school_id'] for x in test_acc_data_with_exp],
    'is_preview':[x['is_preview'] for x in test_acc_data_with_exp],
    'unit_module':[x['unit_module'] for x in test_acc_data_with_exp],
    'scoring_model':[x['scoring_model'] for x in test_acc_data_with_exp],
    'level':[x['level'] for x in test_acc_data_with_exp],
    'teacher':[x['teacher'] for x in test_acc_data_with_exp],
    'class':[x['class'] for x in test_acc_data_with_exp],
    # 'mean_acc_score_model':[x['mean_acc_score_model'] for x in test_acc_data_with_exp] # is a list
    })

print(test_df.shape)
print(test_df.head(3))

logging.info('Drop rows with no value.')
test_df.dropna(inplace=True)

print(test_df.shape)
print(test_df.head(3))

# logging.info('calculating correlation')
# numeric_data = final_df[[
#     'test_count',
#     'learn_count',
#     'exp_count',
#     'hours_use',
#     'mean_learn_time',
#     'mean_resp_time',
#     'freq_all',
#     'freq_duration',
#     'mean_acc',
#     'acc_exposure',
#     'learn_ratio',
#     'accuracy']]
# corr = numeric_data.corr()


# model - lr
model_lr = LinearRegression()
model_lr.fit(
    train_df[
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
        # 'mean_acc_score_model'
        ], 
    train_df['accuracy']
)

lr_predict_train_y = model_lr.predict(train_df[
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
        # 'mean_acc_score_model'
        ] )
print('training accuracy:')
print(mean_squared_error(train_df['accuracy'], lr_predict_train_y))

lr_predict_test_y = model_lr.predict(test_df[
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
        # 'mean_acc_score_model'
        ], )
print('\ntesting accuracy:')
print(mean_squared_error(test_df['accuracy'], lr_predict_test_y))
print(lr_predict_test_y)


# model 2 - svr
model_svr = SVR(gamma='scale', C=1.0, epsilon=0.2)
model_svr.fit(
    train_df[
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
        # 'mean_acc_score_model'
        ], 
    train_df['accuracy']
)

svr_predict_train_y = model_svr.predict(train_df[
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
        # 'mean_acc_score_model'
        ])
print('training accuracy:')
print(mean_squared_error(train_df['accuracy'], svr_predict_train_y))

svc_predict_test_y = model_svr.predict(test_df[
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
        # 'mean_acc_score_model'
        ])
print('\ntesting accuracy:')
print(mean_squared_error(test_df['accuracy'], svc_predict_test_y))
print(svc_predict_test_y)

