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

# construct list of each user id (may repeat)
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

# construct experience set
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

# check is_preview / not_preview accuracy
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

# check duration between testing and learning (correct & wrong)

# sort data by timestamp

newdata = sorted(data, key=lambda k: k['timestamp']) 


# for user 0 ~ 9:

for userId in range(10):
    seen_dict = dict()
    correct_duration = []
    wrong_duration = []

    for record in newdata:
        if(record['user'] == userId):

            for exp in record['experience']:
                if ( (exp['x'] == 'A') and ('m' not in exp.keys()) ): # correct answer
                    if(exp['w'] in seen_dict):
                        correct_duration.append( round( record['timestamp']/1000/60 ) - seen_dict[exp['w']] )
                    else:
                        correct_duration.append(-1)
                elif ( (exp['x'] == 'A') and ('m' in exp.keys()) ): # wrong answer
                    if(exp['w'] in seen_dict):
                        wrong_duration.append( round( record['timestamp']/1000/60 ) - seen_dict[exp['w']] )
                    else:
                        wrong_duration.append(-1)
                seen_dict[exp['w']] = round(record['timestamp']/1000/60)

    print('\n\nuser:')
    print(userId)
    #print('\ndict:')
    #print(seen_dict)
    correct_df = pd.DataFrame(correct_duration, columns=['duration'])
    wrong_df = pd.DataFrame(wrong_duration, columns=['duration'])

    print('\ncorrect:')
    print(correct_df.describe())
    print('\nwrong:')
    print(wrong_df.describe())

    if( len(correct_df['duration'].unique()) > 0 ):
        correct_hist_ax = correct_df['duration'].hist(bins=len(correct_df['duration'].unique()))
        correct_hist_ax.set_xlabel("correct duration count", labelpad=20, size=12)
        correct_hist_ax.set_ylabel("Count(correct)", labelpad=20, size=12)
        plt.show()

    if( len(wrong_df['duration'].unique()) > 0 ):
        wrong_hist_ax = wrong_df['duration'].hist(bins=len(wrong_df['duration'].unique()))
        wrong_hist_ax.set_xlabel("wrong duration count", labelpad=20, size=12)
        wrong_hist_ax.set_ylabel("Count(wrong)", labelpad=20, size=12)
        plt.show()
    



