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

# get each user id into list
userList = []
for ppl in data:
    userList.append(ppl['user'])
user_df = pd.DataFrame(userList, columns=['user'])
logging.info('User count: %d' % (max(userList)+1) )


# plot data count of each user
"""
user_hist_ax = user_df['user'].hist(bins=len(user_df['user'].unique()))
user_hist_ax.set_xlabel("user count", labelpad=20, size=12)
user_hist_ax.set_ylabel("Count", labelpad=20, size=12)
plt.show()
"""

# construct experience set
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

# print experience examples in experience category dict
"""
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

