import datetime
import logging
import numbers
from collections import defaultdict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import func.get_data as getData
import func.plot as getPlot

# Initializing
logging.basicConfig(level=logging.DEBUG,
            format='\n%(asctime)s %(name)-5s === %(levelname)-5s === %(message)s\n')
logging.info('Initializing.')

data_path = './data/data.json'

# Reading data
logging.info('Reading data.')

acc_data = getData.load(path=data_path, just_accs=True)
data = getData.load(path=data_path, just_accs=False)
#data = list(filter(lambda x: len(x['experience'])>0, data)) # remove records with no experience

logging.info('Finish Reading data.')
logging.info('Data length: %d' % len(data))


# [Plot 1] - ratio of learn VS mean acc

# res = getData.get_personal_data(data,
#         count_of_test = True,
#         count_of_learn = True,
#         count_of_exp = True,
#         mean_accuracy = True)
# res_df = pd.DataFrame(data={
#     'test_count':[x['test_count'] for x in res],
#     'learn_count':[x['learn_count'] for x in res],
#     'exp_count':[x['exp_count'] for x in res],
#     'mean_acc':[x['mean_acc'] for x in res],
#     })
# res_df = res_df[(res_df['exp_count'] > 0) & (res_df['mean_acc']>=0)]
# res_df['learn_ratio'] = res_df.apply(lambda row: row['learn_count'] / row['exp_count'], axis=1)

# sns.jointplot(x='learn_ratio', y='mean_acc', color = 'darkorange', data=res_df)
# plt.title('Correlation between learn_ratio and mean_acc',size = 15)
# plt.show()


# [Plot 2] - hours of use VS mean acc

# res = getData.get_personal_data(data,
#         hours_of_use = True,
#         mean_accuracy = True)
# res_df = pd.DataFrame(data={
#     'hours_use':[x['hours_use'] for x in res],
#     'mean_acc':[x['mean_acc'] for x in res],
#     })
# res_df = res_df[(res_df['mean_acc']>=0)]

# sns.jointplot(x='hours_use', y='mean_acc', color = 'darkorange', data=res_df)
# plt.title('Correlation between hours_use and mean_acc',size = 15)
# plt.show()


# [Plot 3] - mean_response_time/mean_learning_time VS mean acc

# res = getData.get_personal_data(data,
#         mean_response_time = True,
#         mean_learning_time = True,
#         mean_accuracy = True)
# res_df = pd.DataFrame(data={
#     'mean_learn_time':[x['mean_learn_time'] for x in res],
#     'mean_resp_time':[x['mean_resp_time'] for x in res],
#     'mean_acc':[x['mean_acc'] for x in res],
#     })
# res_df = res_df[(res_df['mean_acc']>=0) & (res_df['mean_resp_time']>=0) & (res_df['mean_learn_time']>=0)]

# sns.jointplot(x='mean_resp_time', y='mean_acc', color = 'darkorange', data=res_df)
# plt.title('Correlation between mean_resp_time and mean_acc')
# plt.show()

# sns.jointplot(x='mean_learn_time', y='mean_acc', color = 'darkorange', data=res_df)
# plt.title('Correlation between mean_learn_time and mean_acc')
# plt.show()


# [Plot 4] - freq_all/freq_duration VS mean acc

# res = getData.get_personal_data(data,
#         freq_all = True,
#         freq_duration = True,
#         mean_accuracy = True)
# res_df = pd.DataFrame(data={
#     'freq_all':[x['freq_all'] for x in res],
#     'freq_duration':[x['freq_duration'] for x in res],
#     'mean_acc':[x['mean_acc'] for x in res],
#     })
# res_df = res_df[(res_df['mean_acc']>=0) & (res_df['freq_all']>=0) & (res_df['freq_duration']>=0)]

# sns.jointplot(x='freq_all', y='mean_acc', color = 'darkorange', data=res_df)
# plt.title('Correlation between freq_all and mean_acc')
# plt.show()

# sns.jointplot(x='freq_duration', y='mean_acc', color = 'darkorange', data=res_df)
# plt.title('Correlation between freq_duration and mean_acc')
# plt.show()


# [Plot 5] - accuracy_after_exposure VS mean acc

# res = getData.get_personal_data(data,
#         accuracy_after_exposure = True,
#         mean_accuracy = True)
# res_df = pd.DataFrame(data={
#     'acc_exposure':[x['acc_exposure'] for x in res],
#     'mean_acc':[x['mean_acc'] for x in res],
#     })
# res_df = res_df[(res_df['mean_acc']>=0) & (res_df['acc_exposure']>0)]

# sns.jointplot(x='acc_exposure', y='mean_acc', color = 'darkorange', data=res_df)
# plt.title('Correlation between acc_exposure and mean_acc')
# plt.show()



# [Plot 6] - heatmap

res = getData.get_personal_data(
        data,
        # count
        count_of_test = True,
        count_of_learn = True,
        count_of_exp = True,
        # time
        hours_of_use = True,
        mean_response_time = True,
        mean_learning_time = True,
        freq_all = True,
        freq_duration = True,
        # acc
        mean_accuracy = True,
        # other acc
        accuracy_after_exposure = True,
    )

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
    })
res_df = res_df[(res_df['exp_count'] > 0) & (res_df['mean_acc']>=0)]
res_df['learn_ratio'] = res_df.apply(lambda row: row['learn_count'] / row['exp_count'], axis=1)
res_df = res_df[(res_df['mean_resp_time']>=0) & (res_df['mean_learn_time']>=0)]
res_df = res_df[(res_df['freq_all']>=0) & (res_df['freq_duration']>=0)]
res_df = res_df[(res_df['acc_exposure']>0)]

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
for feat_name in features:
    res_df['cut_'+feat_name] = pd.cut(res_df[feat_name], 25)


print(res_df.shape)

print(res_df['acc_exposure'].loc[4])
print(res_df.loc[4]['acc_exposure'])

logging.info('appending data')
c = 0
spaceNum = 1000
for record in acc_data:
    cur_userId = record['user']
    if(cur_userId in res_df.index):
        row = res_df.loc[cur_userId]
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
        for feat_name in features:
            record['cut_'+feat_name] = row['cut_'+feat_name]
    c+=1
    if(c%spaceNum==0):
        print(c)

logging.info('append data done')
print(acc_data[0])

acc_data = list(filter(lambda x: 'test_count' in x.keys(), acc_data))

final_df = pd.DataFrame(data={
    'test_count':[x['test_count'] for x in acc_data],
    'learn_count':[x['learn_count'] for x in acc_data],
    'exp_count':[x['exp_count'] for x in acc_data],
    'hours_use':[x['hours_use'] for x in acc_data],
    'mean_learn_time':[x['mean_learn_time'] for x in acc_data],
    'mean_resp_time':[x['mean_resp_time'] for x in acc_data],
    'freq_all':[x['freq_all'] for x in acc_data],
    'freq_duration':[x['freq_duration'] for x in acc_data],
    'mean_acc':[x['mean_acc'] for x in acc_data],
    'acc_exposure':[x['acc_exposure'] for x in acc_data],
    'learn_ratio':[x['learn_ratio'] for x in acc_data],
    'accuracy':[x['accuracy'] for x in acc_data],
    'cut_test_count':[x['cut_test_count'] for x in acc_data],
    'cut_learn_count':[x['cut_learn_count'] for x in acc_data],
    'cut_exp_count':[x['cut_exp_count'] for x in acc_data],
    'cut_hours_use':[x['cut_hours_use'] for x in acc_data],
    'cut_mean_learn_time':[x['cut_mean_learn_time'] for x in acc_data],
    'cut_mean_resp_time':[x['cut_mean_resp_time'] for x in acc_data],
    'cut_freq_all':[x['cut_freq_all'] for x in acc_data],
    'cut_freq_duration':[x['cut_freq_duration'] for x in acc_data],
    'cut_acc_exposure':[x['cut_acc_exposure'] for x in acc_data],
    'cut_learn_ratio':[x['cut_learn_ratio'] for x in acc_data],
    })

logging.info('calculating correlation')
numeric_data = final_df[[
    'test_count',
    'learn_count',
    'exp_count',
    'hours_use',
    'mean_learn_time',
    'mean_resp_time',
    'freq_all',
    'freq_duration',
    'mean_acc',
    'acc_exposure',
    'learn_ratio',
    'accuracy']]
corr = numeric_data.corr()


# Generate a custom colormap
logging.info('plotting')
sns.set()
cmap = sns.diverging_palette(220, 10, as_cmap=True)
fig, heatmap_ax = plt.subplots(figsize=(15,15))
sns.heatmap(corr, cmap=cmap, vmax=1, center=0, square=True, linewidths=0, annot=True, ax=heatmap_ax)
plt.show()


# [Plot 7] - mean accs !
# code of plot 6 required !

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

for feat_name in features:
    getPlot.mean_accs(acc_data, 'cut_'+feat_name)


