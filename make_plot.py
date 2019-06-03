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
logging.info('Data length(all): %d' % len(data))
logging.info('Data length(only acc): %d' % len(acc_data))


# Calculating personal data
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
        mean_accuracy = True)

# [Plot 0] - data count of each user

logging.info('plot 0')

userList = []
for ppl in data:
    userList.append(ppl['user'])
user_df = pd.DataFrame(userList, columns=['user'])
logging.info('User count: %d' % (max(userList)+1) )

user_hist_ax = user_df['user'].hist(bins=len(user_df['user'].unique()))
user_hist_ax.set_xlabel("user count", labelpad=20, size=12)
user_hist_ax.set_ylabel("Count", labelpad=20, size=12)
plt.show()


# [Plot 1] - ratio of learn VS mean acc

logging.info('plot 1')

res_df = pd.DataFrame(data={
    'test_count':[x['test_count'] for x in res],
    'learn_count':[x['learn_count'] for x in res],
    'exp_count':[x['exp_count'] for x in res],
    'mean_acc':[x['mean_acc'] for x in res],
    })
temp_df = res_df[(res_df['exp_count'] > 0) & (res_df['mean_acc']>=0)]
temp_df['learn_ratio'] = temp_df.apply(lambda row: row['learn_count'] / row['exp_count'], axis=1)

sns.jointplot(x='learn_ratio', y='mean_acc', color = 'darkorange', data=temp_df)
plt.title('Correlation between learn_ratio and mean_acc',size = 15)
plt.show()


# [Plot 2] - hours of use VS mean acc

logging.info('plot 2')

res_df = pd.DataFrame(data={
    'hours_use':[x['hours_use'] for x in res],
    'mean_acc':[x['mean_acc'] for x in res],
    })
temp_df = res_df[(res_df['mean_acc']>=0)]

sns.jointplot(x='hours_use', y='mean_acc', color = 'darkorange', data=temp_df)
plt.title('Correlation between hours_use and mean_acc',size = 15)
plt.show()


# [Plot 3] - mean_response_time/mean_learning_time VS mean acc

logging.info('plot 3')

res_df = pd.DataFrame(data={
    'mean_learn_time':[x['mean_learn_time'] for x in res],
    'mean_resp_time':[x['mean_resp_time'] for x in res],
    'mean_acc':[x['mean_acc'] for x in res],
    })
temp_df = res_df[(res_df['mean_acc']>=0) & (res_df['mean_resp_time']>=0) & (res_df['mean_learn_time']>=0)]

sns.jointplot(x='mean_resp_time', y='mean_acc', color = 'darkorange', data=temp_df)
plt.title('Correlation between mean_resp_time and mean_acc')
plt.show()

sns.jointplot(x='mean_learn_time', y='mean_acc', color = 'darkorange', data=temp_df)
plt.title('Correlation between mean_learn_time and mean_acc')
plt.show()


# [Plot 4] - freq_all/freq_duration VS mean acc

logging.info('plot 4')

res_df = pd.DataFrame(data={
    'freq_all':[x['freq_all'] for x in res],
    'freq_duration':[x['freq_duration'] for x in res],
    'mean_acc':[x['mean_acc'] for x in res],
    })
temp_df = res_df[(res_df['mean_acc']>=0) & (res_df['freq_all']>=0) & (res_df['freq_duration']>=0)]

sns.jointplot(x='freq_all', y='mean_acc', color = 'darkorange', data=temp_df)
plt.title('Correlation between freq_all and mean_acc')
plt.show()

sns.jointplot(x='freq_duration', y='mean_acc', color = 'darkorange', data=temp_df)
plt.title('Correlation between freq_duration and mean_acc')
plt.show()


# [Plot 5] - accuracy_after_exposure VS mean acc

logging.info('plot 5')

res_df = pd.DataFrame(data={
    'acc_exposure':[x['acc_exposure'] for x in res],
    'mean_acc':[x['mean_acc'] for x in res],
    })
temp_df = res_df[(res_df['mean_acc']>=0) & (res_df['acc_exposure']>0)]

sns.jointplot(x='acc_exposure', y='mean_acc', color = 'darkorange', data=temp_df)
plt.title('Correlation between acc_exposure and mean_acc')
plt.show()



# [Plot 6] - heatmap

logging.info('plot 6')

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
temp_df = res_df[(res_df['exp_count'] > 0) & (res_df['mean_acc']>=0)]
temp_df['learn_ratio'] = temp_df.apply(lambda row: row['learn_count'] / row['exp_count'], axis=1)
temp_df = temp_df[(temp_df['mean_resp_time']>=0) & (temp_df['mean_learn_time']>=0)]
temp_df = temp_df[(temp_df['freq_all']>=0) & (temp_df['freq_duration']>=0)]
temp_df = temp_df[(temp_df['acc_exposure']>0)]

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
    temp_df['cut_'+feat_name] = pd.cut(temp_df[feat_name], 10)

print(temp_df.shape)

logging.info('appending data')
c = 0
spaceNum = 1000
for record in acc_data:
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
        for feat_name in features:
            record['cut_'+feat_name] = row['cut_'+feat_name]
    c+=1
    if(c%spaceNum==0):
        print(c)

logging.info('append data done')

acc_data_with_exp = list(filter(lambda x: 'test_count' in x.keys(), acc_data))

final_df = pd.DataFrame(data={
    'test_count':[x['test_count'] for x in acc_data_with_exp],
    'learn_count':[x['learn_count'] for x in acc_data_with_exp],
    'exp_count':[x['exp_count'] for x in acc_data_with_exp],
    'hours_use':[x['hours_use'] for x in acc_data_with_exp],
    'mean_learn_time':[x['mean_learn_time'] for x in acc_data_with_exp],
    'mean_resp_time':[x['mean_resp_time'] for x in acc_data_with_exp],
    'freq_all':[x['freq_all'] for x in acc_data_with_exp],
    'freq_duration':[x['freq_duration'] for x in acc_data_with_exp],
    'mean_acc':[x['mean_acc'] for x in acc_data_with_exp],
    'acc_exposure':[x['acc_exposure'] for x in acc_data_with_exp],
    'learn_ratio':[x['learn_ratio'] for x in acc_data_with_exp],
    'accuracy':[x['accuracy'] for x in acc_data_with_exp],
    'cut_test_count':[x['cut_test_count'] for x in acc_data_with_exp],
    'cut_learn_count':[x['cut_learn_count'] for x in acc_data_with_exp],
    'cut_exp_count':[x['cut_exp_count'] for x in acc_data_with_exp],
    'cut_hours_use':[x['cut_hours_use'] for x in acc_data_with_exp],
    'cut_mean_learn_time':[x['cut_mean_learn_time'] for x in acc_data_with_exp],
    'cut_mean_resp_time':[x['cut_mean_resp_time'] for x in acc_data_with_exp],
    'cut_freq_all':[x['cut_freq_all'] for x in acc_data_with_exp],
    'cut_freq_duration':[x['cut_freq_duration'] for x in acc_data_with_exp],
    'cut_acc_exposure':[x['cut_acc_exposure'] for x in acc_data_with_exp],
    'cut_learn_ratio':[x['cut_learn_ratio'] for x in acc_data_with_exp],
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

logging.info('plot 7')

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
    logging.info(feat_name)
    getPlot.mean_accs(acc_data_with_exp, 'cut_'+feat_name)


# [Plot 8] - number of data of each unit module

logging.info('plot 8')

moduleList = []
for ppl in data:
    moduleList.append(ppl['unit_module'])
module_df = pd.DataFrame(moduleList, columns=['unit_module'])
logging.info('Unit_module count: %d' % (max(moduleList)+1) )

module_hist_ax = module_df['unit_module'].hist(bins=len(module_df['unit_module'].unique()))
module_hist_ax.set_xlabel("unit_module count", labelpad=20, size=12)
module_hist_ax.set_ylabel("Count", labelpad=20, size=12)
plt.show()


# [experiment 1] construct experience set

logging.info('experiment 1')

experienceSet = set()
for ppl in data:
    for exp in ppl['experience']:
        if('w' in exp.keys()):
            experienceSet.add(exp['w'])
        if('m' in exp.keys()):
            experienceSet.add(exp['m'])
logging.info('There are %d experience elements.' % len(experienceSet))

# construct experience category dict
expDict = defaultdict(list)
for exp in experienceSet:
    if(isinstance(exp, numbers.Number)):
        expDict['numbers'].append(exp)
    else:
        if('_' in exp):
            ind = exp.find('_')
            expDict[exp[:ind]].append(exp)
        else:
            expDict['others'].append(exp)

logging.info('There are %d categories in experience dict(including wrong categories).' % len(expDict))

# PRINT experience examples in experience category dict
for category, cateList in expDict.items():
    print(category)
    print(len(cateList))
    maxNum = 10
    curNum = 0
    for example in cateList:
        if(curNum >= maxNum):
            break
        curNum += 1
        print(example, sep=' ')
    print('\n')


# [experiment 2] check is_preview / not_preview accuracy

logging.info('experiment 2')

is_preview_no_acc = 0
is_preview_has_acc = 0
not_preview_no_acc = 0
not_preview_has_acc = 0

is_preview_acc_list = []
not_preview_acc_list = []

for record in data:
    if(record['is_preview'] == True):
        if('accuracy' in record.keys()):
            is_preview_has_acc += 1
            is_preview_acc_list.append(record['accuracy'])
        else:
            is_preview_no_acc += 1
    else:
        if('accuracy' in record.keys()):
            not_preview_has_acc += 1
            not_preview_acc_list.append(record['accuracy'])
        else:
            not_preview_no_acc += 1

logging.info('is_preview_no_acc: %d' % is_preview_no_acc)
logging.info('is_preview_has_acc: %d' % is_preview_has_acc)
logging.info('not_preview_no_acc: %d' % not_preview_no_acc)
logging.info('not_preview_has_acc: %d' % not_preview_has_acc)

is_preview_acc_df = pd.DataFrame(is_preview_acc_list, columns=['accuracy'])
not_preview_acc_df = pd.DataFrame(not_preview_acc_list, columns=['accuracy'])

print('is preview acc:')
print(is_preview_acc_df.describe())

print('not preview acc:')
print(not_preview_acc_df.describe())



# [experiment 3-1]
# check duration between testing and learning (correct & wrong)
# - (ver.1) duration between each 2 records

logging.info('experiment 3-1')

# sort data by timestamp
newdata = sorted(data, key=lambda k: k['timestamp'])

correct_mean_larger_count = 0
wrong_mean_larger_count = 0

# for user 0 ~ 9:

for userId in range(1678):
    seen_dict = dict()
    correct_duration = []
    wrong_duration = []

    for record in newdata:
        if(record['user'] == userId):

            for exp in record['experience']:
                if ( (exp['x'] == 'A') and ('m' not in exp.keys()) ): # correct answer
                    if('w' in exp.keys() and exp['w'] in seen_dict):
                        correct_duration.append( round( record['timestamp']/1000/60 ) - seen_dict[exp['w']] )
                    else:
                        correct_duration.append(-1)
                elif ( (exp['x'] == 'A') and ('m' in exp.keys()) ): # wrong answer
                    if('w' in exp.keys() and exp['w'] in seen_dict):
                        wrong_duration.append( round( record['timestamp']/1000/60 ) - seen_dict[exp['w']] )
                    else:
                        wrong_duration.append(-1)
                
                if('w' in exp.keys()):
                    seen_dict[exp['w']] = round(record['timestamp']/1000/60)

    # print('\n\nuser:')
    # print(userId)
    correct_df = pd.DataFrame(correct_duration, columns=['duration'])
    wrong_df = pd.DataFrame(wrong_duration, columns=['duration'])

    # print(len(correct_df))
    # print(len(wrong_df))

    # remove duration <= 0
    correct_df = correct_df[correct_df['duration']>0]
    wrong_df = wrong_df[wrong_df['duration']>0]

    # print('\ncorrect:')
    # print(correct_df.describe())
    # print('\nwrong:')
    # print(wrong_df.describe())

    if(correct_df['duration'].mean() > wrong_df['duration'].mean()):
        correct_mean_larger_count += 1
    else:
        wrong_mean_larger_count += 1
    
    # plot
    # if( len(correct_df['duration'].unique()) > 0 ):
    #     correct_hist_ax = correct_df['duration'].hist(bins=len(correct_df['duration'].unique()))
    #     correct_hist_ax.set_xlabel("correct duration count", labelpad=20, size=12)
    #     correct_hist_ax.set_ylabel("Count(correct)", labelpad=20, size=12)
    #     plt.show()

    # if( len(wrong_df['duration'].unique()) > 0 ):
    #     wrong_hist_ax = wrong_df['duration'].hist(bins=len(wrong_df['duration'].unique()))
    #     wrong_hist_ax.set_xlabel("wrong duration count", labelpad=20, size=12)
    #     wrong_hist_ax.set_ylabel("Count(wrong)", labelpad=20, size=12)
    #     plt.show()

print('correct mean larger count: %d' % correct_mean_larger_count)
print('wrong mean larger count: %d' % wrong_mean_larger_count)



# [experiment 3-2]
# check duration between testing and learning (correct & wrong)
# - (ver.2) duration within each records

logging.info('experiment 3-2')

correct_mean_larger_count = 0
wrong_mean_larger_count = 0

for userId in range(1678):
    correct_duration = []
    wrong_duration = []

    for record in data:
        if(record['user'] == userId):
            
            seen_dict = dict()
            for exp in record['experience']:
                if ( (exp['x'] == 'A') and ('m' not in exp.keys()) ): # correct answer
                    if('w' in exp.keys() and exp['w'] in seen_dict):
                        correct_duration.append( exp['t'] - seen_dict[exp['w']] )
                    elif('w' in exp.keys()):
                        correct_duration.append(-1)
                elif ( (exp['x'] == 'A') and ('m' in exp.keys()) ): # wrong answer
                    if('w' in exp.keys() and exp['w'] in seen_dict):
                        wrong_duration.append( exp['t'] - seen_dict[exp['w']] )
                    elif('w' in exp.keys()):
                        wrong_duration.append(-1)
                
                if('w' in exp.keys()):
                    seen_dict[exp['w']] = exp['t']

    # print('\n\nuser:')
    # print(userId)
    correct_df = pd.DataFrame(correct_duration, columns=['duration'])
    wrong_df = pd.DataFrame(wrong_duration, columns=['duration'])

    # print(len(correct_df))
    # print(len(wrong_df))
    
    # remove duration <= 0
    correct_df = correct_df[correct_df['duration']>0]
    wrong_df = wrong_df[wrong_df['duration']>0]

    # print('\ncorrect:')
    # print(correct_df.describe())
    # print('\nwrong:')
    # print(wrong_df.describe())

    if(correct_df['duration'].mean() > wrong_df['duration'].mean()):
        correct_mean_larger_count += 1
    else:
        wrong_mean_larger_count += 1

    # plot
    # if( len(correct_df['duration'].unique()) > 0 ):
    #     correct_hist_ax = correct_df['duration'].hist(bins=len(correct_df['duration'].unique()))
    #     correct_hist_ax.set_xlabel("correct duration count", labelpad=20, size=12)
    #     correct_hist_ax.set_ylabel("Count(correct)", labelpad=20, size=12)
    #     plt.show()

    # if( len(wrong_df['duration'].unique()) > 0 ):
    #     wrong_hist_ax = wrong_df['duration'].hist(bins=len(wrong_df['duration'].unique()))
    #     wrong_hist_ax.set_xlabel("wrong duration count", labelpad=20, size=12)
    #     wrong_hist_ax.set_ylabel("Count(wrong)", labelpad=20, size=12)
    #     plt.show()

print('correct mean larger count: %d' % correct_mean_larger_count)
print('wrong mean larger count: %d' % wrong_mean_larger_count)



# [Plot 9] accuracy distribution of different scoring_models

logging.info('plot 9')

scoring_model_count = max( [ x['scoring_model'] for x in data ] ) + 1
record_count_of_each_model = [0 for _ in range(scoring_model_count)]
acc_sum_of_each_model = [0 for _ in range(scoring_model_count)]

for record in data:
    scoring_model_id = record['scoring_model']
    if('accuracy' in record.keys()):
        record_count_of_each_model[scoring_model_id] += 1
        acc_sum_of_each_model[scoring_model_id] += record['accuracy']

record_count_of_each_model = np.array(record_count_of_each_model)
acc_sum_of_each_model = np.array(acc_sum_of_each_model)

mean_acc_of_each_model = np.nan_to_num(acc_sum_of_each_model / record_count_of_each_model)

plt.plot([i for i in range(scoring_model_count)], mean_acc_of_each_model, 'bo')
plt.show()



# [Plot 10] accuracy distribution of different unit_modules

logging.info('plot 10')

module_count = max( [ x['unit_module'] for x in data ] ) + 1
record_count_of_each_module = [0 for _ in range(module_count)]
acc_sum_of_each_module = [0 for _ in range(module_count)]

for record in data:
    module_id = record['unit_module']
    if('accuracy' in record.keys()):
        record_count_of_each_module[module_id] += 1
        acc_sum_of_each_module[module_id] += record['accuracy']

record_count_of_each_module = np.array(record_count_of_each_module)
acc_sum_of_each_module = np.array(acc_sum_of_each_module)

mean_acc_of_each_module = np.nan_to_num(acc_sum_of_each_module / record_count_of_each_module)

plt.plot([i for i in range(module_count)], mean_acc_of_each_module, 'bo')
plt.show()



# [Plot 11] - using frequency & mean accuracy

logging.info('plot 11')

res_df = pd.DataFrame(res, columns=['mean_acc'])

freq_df = pd.DataFrame(getData.get_using_frequency(data), columns=['frequency'])

combined_df = pd.concat([res_df, freq_df], axis=1)
combined_df = combined_df[combined_df['mean_acc'] > -1]

plt.figure(figsize = (10,10))
sns.jointplot(x="frequency", y="mean_acc", color = 'darkorange', data=combined_df)
plt.title('using frequency V.S. mean accuracy',size = 15)
plt.show()



# [Plot 12] - using frequency & mean accuracy

logging.info('plot 12')

res_df = pd.DataFrame([[rec['mean_acc'], rec['freq_all']] for rec in res], columns=['mean_acc', 'freq_all'])
res_df = res_df[res_df['freq_all']>=0]
res_df = res_df[res_df['mean_acc']>=0]

res_df.sort_values(by=['freq_all'], inplace=True)

mean_freq_lst = []
mean_acc_lst = []
spaceNum = round(res_df.shape[0]/10)
for g, df in res_df.groupby(np.arange(len(res_df)) // spaceNum):
    mean_freq_lst.append(df['freq_all'].mean())
    mean_acc_lst.append(df['mean_acc'].mean())

# plot
plt.figure(figsize = (10,8))
plt.scatter(
    mean_freq_lst, 
    mean_acc_lst,
    s=20)
plt.xlabel('mean_frequency', fontsize=14)
plt.ylabel('mean_accuracy', fontsize=14)
plt.title('mean_freq VS mean_acc', fontsize=17)
plt.show()



# [Plot 13] - unit_module VS score_model VS accuracy
# !! should extract data with accuracy only !!

logging.info('plot 13')

old_res = getData.get_personal_old_data(acc_data, ['unit_module', 'accuracy', 'scoring_model'])

old_res_lst = []
for one_res in old_res:
    if(len(one_res)==3):
        old_res_lst.append([one_res['unit_module'], one_res['accuracy'], one_res['scoring_model']])
    else:
        old_res_lst.append([-1,-1,-1])

temp_np = np.array(old_res_lst)
old_res_pd = pd.DataFrame(old_res_lst, columns=['unit_module', 'accuracy', 'scoring_model'])
old_res_pd = old_res_pd[old_res_pd['accuracy']>=0]
print(old_res_pd.shape)

# plot
plt.figure(figsize = (10,8))
plt.scatter(
    old_res_pd['unit_module'], 
    old_res_pd['scoring_model'] ,
    c=old_res_pd['accuracy'], 
    cmap = 'hot', 
    s=20)
ax = plt.gca()
ax.set_facecolor('lightslategray')
plt.colorbar().set_label('accuracy', fontsize=14)
plt.xlabel('unit_module_id', fontsize=14)
plt.ylabel('scoring_model_id', fontsize=14)
plt.title('accuracy with unit_module & scoring_model', fontsize=17)
plt.show()



# [Plot 14] question category VS mean accuracy

logging.info('plot 14')

correct_count = dict()
all_count = dict()
for record in data:
    for exp in record['experience']:
        if(exp['x'] == 'A' and 'w' in exp.keys()):
            cur_word = exp['w']
            if(isinstance(cur_word, numbers.Number)):
                if('numbers' not in all_count.keys()):
                    all_count['numbers'] = 1
                else:
                    all_count['numbers'] += 1
                if('m' not in exp.keys()):
                    if('numbers' not in correct_count.keys()):
                        correct_count['numbers'] = 1
                    else:
                        correct_count['numbers'] += 1
            else:
                if('_' in cur_word):
                    ind = cur_word.find('_')
                    cur_category = cur_word[:ind]
                    cur_category = cur_category.lower()
                    if(cur_category not in all_count.keys()):
                        all_count[cur_category] = 1
                    else:
                        all_count[cur_category] += 1
                    if('m' not in exp.keys()):
                        if(cur_category not in correct_count.keys()):
                            correct_count[cur_category] = 1
                        else:
                            correct_count[cur_category] += 1
print(all_count)
print(correct_count)

category_name = []
category_acc = []
question_categories = ['spot', 'numbers', 'phonics', 'phonemes', 'singular', 'plural', 'letters', 'abc', 'sight']

for key_name in all_count.keys():
    if(key_name in correct_count.keys() and key_name in question_categories):
        category_name.append(key_name)
        category_acc.append(correct_count[key_name]/all_count[key_name])

# plot
plt.bar(np.arange(len(category_acc)), category_acc)
plt.xticks(np.arange(len(category_name)), category_name)
plt.show()



