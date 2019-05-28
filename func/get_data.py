import json
import numbers

import numpy as np
import pandas as pd


def load(path, just_accs=False):
    with open(path) as f:
        data = json.loads(f.read())
    if just_accs:
        data = filter_accs(data)
    return data

def filter_accs(data):
    """Filters out just the records with accuracies."""
    return [x for x in data if 'accuracy' in x.keys()]

def generate_data_pair(acc_data, personal_data):
    n_teachers = max([x['teacher'] for x in acc_data]) + 1
    n_classes = max([x['class'] for x in acc_data]) + 1
    n_levels = max([x['level'] for x in acc_data]) + 1
    n_schools = max([x['school'] for x in acc_data]) + 1 # here!!!
    n_features = n_teachers + n_classes + n_levels + n_schools + len(personal_data[i])
    for record in acc_data:
        break # here !!!

    return

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


def get_personal_data(
        data,
        count_of_test = False,
        count_of_learn = False,
        count_of_exp = False,
        hours_of_use = False,
        mean_response_time = False,
        mean_learning_time = False,
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
        mean_accuracy_model_a = False, #
        # others
        school_id = False
    ):

    user_count = max( [ x['user'] for x in data ] ) + 1
    res = [dict() for _ in range(user_count)]

    school_count_of_groups_list = get_school_count_of_each_group(data)
    question_categories = ['spot', 'numbers', 'phonics', 'phonemes', 'singular', 'plural', 'letters', 'abc', 'sight']
    

    count_of_record_list = [0 for _ in range(user_count)]
    count_of_exp_list = [0 for _ in range(user_count)]
    count_of_test_list = [0 for _ in range(user_count)]
    count_of_learn_list = [0 for _ in range(user_count)]
    hours_of_use_list = [0 for _ in range(user_count)]
    response_time_list = [0 for _ in range(user_count)]
    count_of_accuracy_list = [0 for _ in range(user_count)]
    accuracy_list = [0 for _ in range(user_count)]
    
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

    id_of_school_id_list = [0 for _ in range(user_count)]


    zero_exp_rec_count = 0
    for record in data:
        cur_userId = record['user']
        count_of_record_list[cur_userId] += 1

        # mean_accuracy
        if('accuracy' in record.keys()):
            count_of_accuracy_list[cur_userId] += 1
            accuracy_list[cur_userId] += record['accuracy']

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
            res[id]['hours_use'] = hours_of_use_list[id]
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
        if(school_id == True):
            res[id]['school_id'] = id_of_school_id_list[id]
    
    return res

