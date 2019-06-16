import json
import numbers
import random
from statistics import mean
import datetime
from datetime import date

import numpy as np
import pandas as pd


# ==========
# data processing for training
# ==========

def load(path, just_accs=False):
    with open(path) as f:
        data = json.loads(f.read())
    if just_accs:
        data = filter_accs(data)
    return data

def filter_accs(data):
    """Filters out just the records with accuracies."""
    return [x for x in data if 'accuracy' in x.keys()]

def split_train_test(data, train_ratio):
    data_count = len(data)
    random.shuffle(data)
    train_max_id = round(data_count*train_ratio)
    train_data = []
    test_data = []
    train_data = data[:train_max_id]
    test_data = data[train_max_id:]
    return train_data, test_data

def split_train_test_by_user(data, train_ratio):
    user_count = max( [ x['user'] for x in data ] ) + 1
    everyone_record = [[] for _ in range(user_count)]
    for record in data:
        everyone_record[record['user']].append(record)
    train_data = []
    test_data = []
    for user_record in everyone_record:
        random.shuffle(user_record)
        train_max_id = round(len(user_record)*train_ratio)
        train_data += user_record[:train_max_id]
        test_data += user_record[train_max_id:]
    random.shuffle(train_data)
    random.shuffle(test_data)
    return train_data, test_data



def generate_data_pair_for(acc_data, personal_data):
    n_teachers = max([x['teacher'] for x in acc_data]) + 1
    n_classes = max([x['class'] for x in acc_data]) + 1
    n_schools = max([x['school'] for x in acc_data]) + 1 # here!!!
    n_features = n_teachers + n_classes + n_schools + len(personal_data[i])
    for record in acc_data:
        break # here !!!

    return


# ==========
# get one feature
# ==========

def get_school_count_of_each_group(data):
    everyone_data = []
    for record in data:
        temp = [record['school_group'], record['school']]
        everyone_data.append(temp)
    
    school_group_count = max([x[0] for x in everyone_data]) + 1
    everyone_data_df = pd.DataFrame(everyone_data, columns=['school_group', 'school'])

    school_count_list = []
    for group_id in range(school_group_count):
        group_df = everyone_data_df[everyone_data_df['school_group'] == group_id]
        school_count_list.append(group_df['school'].max() + 1)
    return school_count_list

def get_using_frequency(data):
    user_count = max( [ x['user'] for x in data ] ) + 1
    everyone_timestamp = [[] for _ in range(user_count)]
    for record in data:
        everyone_timestamp[record['user']].append(round(record['timestamp']/1000))
    # sort
    for i in range(user_count):
        everyone_timestamp[i] = sorted(everyone_timestamp[i])
    # all
    freq_all = []
    for one_timestamp in everyone_timestamp:
        end_time = date.fromtimestamp(one_timestamp[-1])
        start_time = date.fromtimestamp(one_timestamp[0])
        use_day_duration = (end_time - start_time).days
        if(use_day_duration>0):
            freq_all.append(len(one_timestamp)/use_day_duration)
        else:
            freq_all.append(-1)
    return freq_all

# correctness within short time(a record) to memorize
def get_correctness_after_exposure(data):
    user_count = max( [ x['user'] for x in data ] ) + 1
    user_words_rec = [ dict.fromkeys(['count_all','count_correct'], 0) for _ in range(user_count)]
    for record in data:
        cur_userId = record['user']
        seen_set = set()
        for exp in record['experience']:
            if('w' in exp.keys()):
                cur_word = exp['w']
                if(exp['x'] == 'X'):
                    # exposure
                    seen_set.add(cur_word)
                else:
                    # test
                    if(cur_word in seen_set):
                        user_words_rec[cur_userId]['count_all'] += 1
                        if('m' not in exp.keys()):
                            user_words_rec[cur_userId]['count_correct'] += 1

    correct_np = np.array([x['count_correct'] for x in user_words_rec])
    count_np = np.array([x['count_all'] for x in user_words_rec])

    return correct_np / count_np

# duration of each time using (hour)
def mean_duration_each_record(data):
    user_count = max( [ x['user'] for x in data ] ) + 1
    user_timestamp_rec = [ [] for _ in range(user_count)]
    for record in data:
        cur_userId = record['user']
        user_timestamp_rec[cur_userId].append(datetime.datetime.fromtimestamp(round(record['timestamp']/1000)))
    user_mean_duration_rec = []
    for one_user_timestamp in user_timestamp_rec:
        one_user_timestamp.sort()
        cur_count = 0
        cur_duration_sum = datetime.timedelta()
        for i in range(len(one_user_timestamp)-1):
            cur_duration_sum += (one_user_timestamp[i+1] - one_user_timestamp[i])
            cur_count += 1
        if(cur_count == 0):
            user_mean_duration_rec.append(-1)
        else:
            user_mean_duration_rec.append( round( (cur_duration_sum/cur_count).total_seconds() )/(60*60) )
    return user_mean_duration_rec



# ==========
# extract features of each user
# ==========


def get_personal_old_data( data, col_name_list ):
    user_count = max( [ x['user'] for x in data ] ) + 1
    res = [dict() for _ in range(user_count)]

    for record in data:
        cur_userId = record['user']
        for col_name in col_name_list:
            res[cur_userId][col_name] = record[col_name]
    return res

def get_personal_data(
        data,
        # count
        count_of_test = False,
        count_of_learn = False,
        count_of_exp = False,
        # time
        hours_of_use = False,
        mean_response_time = False,
        mean_learning_time = False,
        freq_all = False,
        freq_week = False, #
        freq_month = False, #
        freq_duration = False,
        # acc
        mean_accuracy = False,
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
        # acc of different score models
        mean_accuracy_each_scoring_model = False,
        mean_accuracy_each_unit = False,
        # other acc
        accuracy_after_exposure = False,
        # others
        school_id = False
    ):

    user_count = max( [ x['user'] for x in data ] ) + 1
    res = [dict() for _ in range(user_count)]

    school_count_of_groups_list = get_school_count_of_each_group(data)
    using_frequency_list = get_using_frequency(data)
    duration_between_records_list = mean_duration_each_record(data)
    question_categories = ['spot', 'numbers', 'phonics', 'phonemes', 'singular', 'plural', 'letters', 'abc', 'sight']
    scoring_model_count = max([x['scoring_model'] for x in data]) + 1
    unit_count = max([x['unit'] for x in data]) + 1

    # count
    count_of_record_list = [0 for _ in range(user_count)]
    count_of_exp_list = [0 for _ in range(user_count)]
    count_of_test_list = [0 for _ in range(user_count)]
    count_of_learn_list = [0 for _ in range(user_count)]

    # time
    hours_of_use_list = [0 for _ in range(user_count)]
    response_time_list = [0 for _ in range(user_count)]

    # acc
    count_of_accuracy_list = [0 for _ in range(user_count)]
    accuracy_list = [0 for _ in range(user_count)]
    
    # acc - question category
    count_of_spot_list = [0 for _ in range(user_count)]
    count_of_numbers_list = [0 for _ in range(user_count)]
    count_of_phonics_list = [0 for _ in range(user_count)]
    count_of_phonemes_list = [0 for _ in range(user_count)]
    count_of_singplu_list = [0 for _ in range(user_count)]
    count_of_letters_list = [0 for _ in range(user_count)]
    count_of_abc_list = [0 for _ in range(user_count)]
    count_of_sight_list = [0 for _ in range(user_count)]
    count_of_others_list = [0 for _ in range(user_count)]

    correct_spot_list = [0 for _ in range(user_count)]
    correct_numbers_list = [0 for _ in range(user_count)]
    correct_phonics_list = [0 for _ in range(user_count)]
    correct_phonemes_list = [0 for _ in range(user_count)]
    correct_singplu_list = [0 for _ in range(user_count)]
    correct_letters_list = [0 for _ in range(user_count)]
    correct_abc_list = [0 for _ in range(user_count)]
    correct_sight_list = [0 for _ in range(user_count)]
    correct_others_list = [0 for _ in range(user_count)]

    # acc - scoring_model
    count_of_accuracy_of_each_scoring_model = [[[] for _ in range(scoring_model_count)] for _ in range(user_count)]

    # acc - unit
    count_of_accuracy_of_each_unit = [[[] for _ in range(unit_count)] for _ in range(user_count)]

    # acc - exposure
    accuracy_exposure_list = get_correctness_after_exposure(data)

    # others
    id_of_school_id_list = [0 for _ in range(user_count)]


    zero_exp_rec_count = 0
    for record in data:
        cur_userId = record['user']
        count_of_record_list[cur_userId] += 1

        # mean_accuracy
        if('accuracy' in record.keys()):
            count_of_accuracy_list[cur_userId] += 1
            accuracy_list[cur_userId] += record['accuracy']
            count_of_accuracy_of_each_scoring_model[cur_userId][record['scoring_model']].append(record['accuracy'])
            count_of_accuracy_of_each_unit[cur_userId][record['unit']].append(record['accuracy'])

        # hours_of_use
        if(len(record['experience']) > 0):
            hours_of_use_list[cur_userId] += record['experience'][-1]['t']
        else:
            zero_exp_rec_count += 1

        # school_id
        ordinal_id = 0
        for group_id in range(record['school_group']):
            ordinal_id += school_count_of_groups_list[group_id]
        ordinal_id += record['school']
        id_of_school_id_list[cur_userId] = ordinal_id

        for exp in record['experience']:
            count_of_exp_list[cur_userId] += 1

            if(exp['x'] == 'A'):
                # count_of_test
                count_of_test_list[cur_userId] += 1
                # mean_response_time
                if('s' in exp.keys()):
                    response_time_list[cur_userId] += exp['s']
            elif(exp['x'] == 'X'):
                # count_of_learn
                count_of_learn_list[cur_userId] += 1
            
            if('w' in exp.keys()):
                cur_word = exp['w']
                if(isinstance(cur_word, numbers.Number)):
                    # mean_acc_numbers
                    count_of_numbers_list[cur_userId] += 1
                    if('m' not in exp.keys()):
                        correct_numbers_list[cur_userId] += 1
                else:
                    if('_' in cur_word):
                        ind = cur_word.find('_')
                        cur_category = cur_word[:ind]
                        cur_category = cur_category.lower()
                        if(cur_category == 'spot'):
                            # mean_acc_spot
                            count_of_spot_list[cur_userId] += 1
                            if('m' not in exp.keys()):
                                correct_spot_list[cur_userId] += 1
                        elif(cur_category == 'phonics'):
                            # mean_acc_phonics
                            count_of_phonics_list[cur_userId] += 1
                            if('m' not in exp.keys()):
                                correct_phonics_list[cur_userId] += 1
                        elif(cur_category == 'singular' or cur_category == 'plural'):
                            # mean_acc_singplu
                            count_of_singplu_list[cur_userId] += 1
                            if('m' not in exp.keys()):
                                correct_singplu_list[cur_userId] += 1
                        elif(cur_category == 'phonemes'):
                            # mean_acc_phonemes
                            count_of_phonemes_list[cur_userId] += 1
                            if('m' not in exp.keys()):
                                correct_phonemes_list[cur_userId] += 1
                        elif(cur_category == 'letters'):
                            # mean_acc_letters
                            count_of_letters_list[cur_userId] += 1
                            if('m' not in exp.keys()):
                                correct_letters_list[cur_userId] += 1
                        elif(cur_category == 'abc'):
                            # mean_acc_abc
                            count_of_abc_list[cur_userId] += 1
                            if('m' not in exp.keys()):
                                correct_abc_list[cur_userId] += 1
                        elif(cur_category == 'sight'):
                            # mean_acc_sight
                            count_of_sight_list[cur_userId] += 1
                            if('m' not in exp.keys()):
                                correct_sight_list[cur_userId] += 1
                        else:
                            # mean_acc_others
                            count_of_others_list[cur_userId] += 1
                            if('m' not in exp.keys()):
                                correct_others_list[cur_userId] += 1
                    else:
                        # mean_acc_others
                        count_of_others_list[cur_userId] += 1
                        if('m' not in exp.keys()):
                            correct_others_list[cur_userId] += 1

    # integrate data of users
    for id in range(user_count):
        if(count_of_test == True):
            res[id]['test_count'] = count_of_test_list[id]
        if(count_of_learn == True):
            res[id]['learn_count'] = count_of_learn_list[id]
        if(count_of_exp == True):
            res[id]['exp_count'] = count_of_exp_list[id]
        if(hours_of_use == True):
            res[id]['hours_use'] = hours_of_use_list[id]/60/60
        if(mean_accuracy == True):
            if(count_of_accuracy_list[id] == 0):
                res[id]['mean_acc'] = -1
            else:
                res[id]['mean_acc'] = accuracy_list[id] / count_of_accuracy_list[id]
        if(mean_response_time == True):
            if(count_of_test_list[id] == 0):
                res[id]['mean_resp_time'] = -1
            else:
                res[id]['mean_resp_time'] = response_time_list[id] / count_of_test_list[id]
        if(mean_learning_time == True):
            if(count_of_learn_list[id] == 0):
                res[id]['mean_learn_time'] = -1
            else:
                res[id]['mean_learn_time'] = (hours_of_use_list[id] - response_time_list[id]) / count_of_learn_list[id]
        if(mean_accuracy_spot == True):
            if(count_of_spot_list[id] == 0):
                res[id]['mean_acc_spot'] = -1
            else:
                res[id]['mean_acc_spot'] = correct_spot_list[id] / count_of_spot_list[id]
        if(mean_accuracy_numbers == True):
            if(count_of_numbers_list[id] == 0):
                res[id]['mean_acc_numbers'] = -1
            else:
                res[id]['mean_acc_numbers'] = correct_numbers_list[id] / count_of_numbers_list[id]
        if(mean_accuracy_phonics == True):
            if(count_of_phonics_list[id] == 0):
                res[id]['mean_acc_phonics'] = -1
            else:
                res[id]['mean_acc_phonics'] = correct_phonics_list[id] / count_of_phonics_list[id]
        if(mean_accuracy_phonemes == True):
            if(count_of_phonemes_list[id] == 0):
                res[id]['mean_acc_phonemes'] = -1
            else:
                res[id]['mean_acc_phonemes'] = correct_phonemes_list[id] / count_of_phonemes_list[id]
        if(mean_accuracy_singplu == True):
            if(count_of_singplu_list[id] == 0):
                res[id]['mean_acc_singplu'] = -1
            else:
                res[id]['mean_acc_singplu'] = correct_singplu_list[id] / count_of_singplu_list[id]
        if(mean_accuracy_abc == True):
            if(count_of_abc_list[id] == 0):
                res[id]['mean_acc_abc'] = -1
            else:
                res[id]['mean_acc_abc'] = correct_abc_list[id] / count_of_abc_list[id]
        if(mean_accuracy_letters == True):
            if(count_of_letters_list[id] == 0):
                res[id]['mean_acc_letters'] = -1
            else:
                res[id]['mean_acc_letters'] = correct_letters_list[id] / count_of_letters_list[id]
        if(mean_accuracy_sight == True):
            if(count_of_sight_list[id] == 0):
                res[id]['mean_acc_sight'] = -1
            else:
                res[id]['mean_acc_sight'] = correct_sight_list[id] / count_of_sight_list[id]
        if(mean_accuracy_others == True):
            if(count_of_others_list[id] == 0):
                res[id]['mean_acc_others'] = -1
            else:
                res[id]['mean_acc_others'] = correct_others_list[id] / count_of_others_list[id]
        if(mean_accuracy_each_scoring_model == True):
            if(count_of_accuracy_list[id] == 0):
                res[id]['mean_acc_score_model'] = -1
            else:
                temp = []
                for lst in count_of_accuracy_of_each_scoring_model[id]:
                    if(len(lst) == 0):
                        temp.append(-1)
                    else:
                        temp.append(mean(lst))
                res[id]['mean_acc_score_model'] = temp.copy()
        if(mean_accuracy_each_unit == True):
            if(count_of_accuracy_list[id] == 0):
                res[id]['mean_acc_unit'] = -1
            else:
                temp = []
                for lst in count_of_accuracy_of_each_unit[id]:
                    if(len(lst) == 0):
                        temp.append(-1)
                    else:
                        temp.append(mean(lst))
                res[id]['mean_acc_unit'] = temp.copy()
        if(accuracy_after_exposure == True):
            res[id]['acc_exposure'] = accuracy_exposure_list[id]
        if(school_id == True):
            res[id]['school_id'] = id_of_school_id_list[id]
        if(freq_all == True):
            res[id]['freq_all'] = using_frequency_list[id]
        if(freq_duration == True):
            res[id]['freq_duration'] = duration_between_records_list[id]
    
    return res

