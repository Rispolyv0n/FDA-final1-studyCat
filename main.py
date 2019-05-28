import datetime
import logging
import numbers
from collections import defaultdict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import func.get_data as getData


logging.basicConfig(level=logging.DEBUG,
            format='\n%(asctime)s %(name)-5s === %(levelname)-5s === %(message)s\n')

logging.info('Initializing.')
data_path = './data/data.json'


logging.info('Reading data.')
data = getData.load(path=data_path, just_accs=False)


logging.info('Finish Reading data.')
logging.info('Data length: %d' % len(data))


# print data
"""
for i in range(5):
    print(data[i])
    print(len(data[i]['experience']))

print(data[3])
for exp in data[3]['experience']:
    print(exp)
"""

# [experiment 0] construct list of each user id (may repeat)
"""
userList = []
for ppl in data:
    userList.append(ppl['user'])
user_df = pd.DataFrame(userList, columns=['user'])
logging.info('User count: %d' % (max(userList)+1) )


# PLOT data count of each user
user_hist_ax = user_df['user'].hist(bins=len(user_df['user'].unique()))
user_hist_ax.set_xlabel("user count", labelpad=20, size=12)
user_hist_ax.set_ylabel("Count", labelpad=20, size=12)
plt.show()
"""

# [experiment 1] construct experience set
"""
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
"""

# [experiment 2] check is_preview / not_preview accuracy
"""
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
print(is_preview_acc_df.info())
print(is_preview_acc_df.describe())

print('not preview acc:')
print(not_preview_acc_df.info())
print(not_preview_acc_df.describe())
"""

# check unit_module
"""
moduleList = []
for ppl in data:
    moduleList.append(ppl['unit_module'])
module_df = pd.DataFrame(moduleList, columns=['unit_module'])
logging.info('Unit_module count: %d' % (max(moduleList)+1) )

# PLOT data count of each unit_module
module_hist_ax = module_df['unit_module'].hist(bins=len(module_df['unit_module'].unique()))
module_hist_ax.set_xlabel("unit_module count", labelpad=20, size=12)
module_hist_ax.set_ylabel("Count", labelpad=20, size=12)
plt.show()
"""


# [experiment 3-1]
# check duration between testing and learning (correct & wrong)
# - (ver.1) duration between each 2 records

# sort data by timestamp
"""

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

    print('\n\nuser:')
    print(userId)
    #print('\ndict:')
    #print(seen_dict)
    correct_df = pd.DataFrame(correct_duration, columns=['duration'])
    wrong_df = pd.DataFrame(wrong_duration, columns=['duration'])

    print(len(correct_df))
    print(len(wrong_df))

    # remove duration <= 0
    correct_df = correct_df[correct_df['duration']>0]
    wrong_df = wrong_df[wrong_df['duration']>0]

    print('\ncorrect:')
    print(correct_df.describe())
    print('\nwrong:')
    print(wrong_df.describe())

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
"""



# [experiment 3-2]
# check duration between testing and learning (correct & wrong)
# - (ver.2) duration within each records
"""
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

    print('\n\nuser:')
    print(userId)
    correct_df = pd.DataFrame(correct_duration, columns=['duration'])
    wrong_df = pd.DataFrame(wrong_duration, columns=['duration'])

    print(len(correct_df))
    print(len(wrong_df))
    
    # remove duration <= 0
    correct_df = correct_df[correct_df['duration']>0]
    wrong_df = wrong_df[wrong_df['duration']>0]

    print('\ncorrect:')
    print(correct_df.describe())
    print('\nwrong:')
    print(wrong_df.describe())

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
"""


# [Test Function] - get_data/get_personal_data

logging.info('Start getting features')
# testData = data[:5]
res = getData.get_personal_data(
    data, 
    count_of_test = False,
    count_of_learn = False,
    count_of_exp = False,
    hours_of_use = False,
    mean_response_time = False,
    mean_learning_time = False,
    mean_accuracy = True,
    # acc of different question categories
    mean_accuracy_spot = True,
    mean_accuracy_numbers = True,
    mean_accuracy_phonics = False,
    mean_accuracy_phonemes = False,
    mean_accuracy_singplu = False,
    mean_accuracy_letters = False,
    mean_accuracy_abc = False,
    mean_accuracy_sight = False,
    mean_accuracy_others = False,
    # acc of different score models
    mean_accuracy_each_scoring_model = True,
    # others
    school_id = False)

logging.info('Getting features done')
print_num = len(res)
print_num = 200
for i in range(print_num):
    print(res[i])



# [Experiment 4] accuracy distribution of different scoring_models
"""
scoring_model_count = max( [ x['scoring_model'] for x in data ] ) + 1
record_count_of_each_model = [0 for _ in range(scoring_model_count)]
acc_sum_of_each_model = [0 for _ in range(scoring_model_count)]

print(scoring_model_count)

for record in data:
    scoring_model_id = record['scoring_model']
    if('accuracy' in record.keys()):
        record_count_of_each_model[scoring_model_id] += 1
        acc_sum_of_each_model[scoring_model_id] += record['accuracy']

print(record_count_of_each_model)
print(acc_sum_of_each_model)

record_count_of_each_model = np.array(record_count_of_each_model)
acc_sum_of_each_model = np.array(acc_sum_of_each_model)

mean_acc_of_each_model = np.nan_to_num(acc_sum_of_each_model / record_count_of_each_model)
print(mean_acc_of_each_model)

plt.plot([i for i in range(scoring_model_count)], mean_acc_of_each_model, 'bo')
plt.show()
"""

# [Experiment 5] accuracy distribution of different unit_modules
"""
module_count = max( [ x['unit_module'] for x in data ] ) + 1
record_count_of_each_module = [0 for _ in range(module_count)]
acc_sum_of_each_module = [0 for _ in range(module_count)]

print(module_count)

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
"""

# [Test Function] - get_data/get_school_ordinal
"""
a = getData.get_school_ordinal(data)
print('outside:')
print(a)
"""


