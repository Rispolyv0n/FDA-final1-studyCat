import datetime
import logging
import numbers
from collections import defaultdict

import pandas as pd
import numpy as np

import func.get_data as getData


# Initializing
logging.basicConfig(level=logging.DEBUG,
            format='\n%(asctime)s %(name)-5s === %(levelname)-5s === %(message)s\n')
logging.info('Initializing...')

data_path = './data/data.json'


# Reading data
logging.info('Reading data...')

acc_data = getData.load(path=data_path, just_accs=True)
data = getData.load(path=data_path, just_accs=False)

logging.info('Finish Reading data.')
logging.info('Data length(all): %d' % len(data))
logging.info('Data length(only acc): %d' % len(acc_data))


# Calculating personal data
logging.info('Calculating personal data...')
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




# not yet
# temp_df = res_df[(res_df['exp_count'] > 0) & (res_df['mean_acc']>=0)]
# temp_df['learn_ratio'] = temp_df.apply(lambda row: row['learn_count'] / row['exp_count'], axis=1)
# temp_df = temp_df[(temp_df['mean_resp_time']>=0) & (temp_df['mean_learn_time']>=0)]
# temp_df = temp_df[(temp_df['freq_all']>=0) & (temp_df['freq_duration']>=0)]
# temp_df = temp_df[(temp_df['acc_exposure']>0)]

# # bin cut
# features = [
#     'test_count',
#     'learn_count',
#     'exp_count',
#     'hours_use',
#     'mean_learn_time',
#     'mean_resp_time',
#     'freq_all',
#     'freq_duration',
#     'acc_exposure',
#     'learn_ratio'
#     ]
# for feat_name in features:
#     temp_df['cut_'+feat_name] = pd.cut(temp_df[feat_name], 10)

# print(temp_df.shape)

# logging.info('appending data')
# c = 0
# spaceNum = 1000
# for record in acc_data:
#     cur_userId = record['user']
#     if(cur_userId in temp_df.index):
#         row = temp_df.loc[cur_userId]
#         record['test_count'] = row['test_count']
#         record['learn_count'] = row['learn_count']
#         record['exp_count'] = row['exp_count']
#         record['hours_use'] = row['hours_use']
#         record['mean_learn_time'] = row['mean_learn_time']
#         record['mean_resp_time'] = row['mean_resp_time']
#         record['freq_all'] = row['freq_all']
#         record['freq_duration'] = row['freq_duration']
#         record['mean_acc'] = row['mean_acc']
#         record['acc_exposure'] = row['acc_exposure']
#         record['learn_ratio'] = row['learn_ratio']
#         for feat_name in features:
#             record['cut_'+feat_name] = row['cut_'+feat_name]
#     c+=1
#     if(c%spaceNum==0):
#         print(c)

# logging.info('append data done')

# acc_data_with_exp = list(filter(lambda x: 'test_count' in x.keys(), acc_data))

# final_df = pd.DataFrame(data={
#     'test_count':[x['test_count'] for x in acc_data_with_exp],
#     'learn_count':[x['learn_count'] for x in acc_data_with_exp],
#     'exp_count':[x['exp_count'] for x in acc_data_with_exp],
#     'hours_use':[x['hours_use'] for x in acc_data_with_exp],
#     'mean_learn_time':[x['mean_learn_time'] for x in acc_data_with_exp],
#     'mean_resp_time':[x['mean_resp_time'] for x in acc_data_with_exp],
#     'freq_all':[x['freq_all'] for x in acc_data_with_exp],
#     'freq_duration':[x['freq_duration'] for x in acc_data_with_exp],
#     'mean_acc':[x['mean_acc'] for x in acc_data_with_exp],
#     'acc_exposure':[x['acc_exposure'] for x in acc_data_with_exp],
#     'learn_ratio':[x['learn_ratio'] for x in acc_data_with_exp],
#     'accuracy':[x['accuracy'] for x in acc_data_with_exp],
#     'cut_test_count':[x['cut_test_count'] for x in acc_data_with_exp],
#     'cut_learn_count':[x['cut_learn_count'] for x in acc_data_with_exp],
#     'cut_exp_count':[x['cut_exp_count'] for x in acc_data_with_exp],
#     'cut_hours_use':[x['cut_hours_use'] for x in acc_data_with_exp],
#     'cut_mean_learn_time':[x['cut_mean_learn_time'] for x in acc_data_with_exp],
#     'cut_mean_resp_time':[x['cut_mean_resp_time'] for x in acc_data_with_exp],
#     'cut_freq_all':[x['cut_freq_all'] for x in acc_data_with_exp],
#     'cut_freq_duration':[x['cut_freq_duration'] for x in acc_data_with_exp],
#     'cut_acc_exposure':[x['cut_acc_exposure'] for x in acc_data_with_exp],
#     'cut_learn_ratio':[x['cut_learn_ratio'] for x in acc_data_with_exp],
#     })

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


