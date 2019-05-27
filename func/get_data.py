import json
import numbers

def load(path, just_accs=False):
    with open(path) as f:
        data = json.loads(f.read())
    if just_accs:
        data = filter_accs(data)
    return data

def filter_accs(data):
    """Filters out just the records with accuracies."""
    return [x for x in data if 'accuracy' in x.keys()]

def get_personal_data(
    data,
    count_of_test = False,
    count_of_learn = False,
    count_of_exp = False,
    hours_of_use = False,
    mean_response_time = False,
    mean_learning_time = False, #
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
    # acc of different score models (NOT DONE YET!!)
    mean_accuracy_model_a = False): #

    user_count = max( [ x['user'] for x in data ] ) + 1
    res = [dict() for _ in range(user_count)]
    
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

    question_categories = ['spot', 'numbers', 'phonics', 'phonemes', 'singular', 'plural', 'letters', 'abc', 'sight']

    for record in data:
        cur_userId = record['user']
        count_of_record_list[cur_userId] += 1

        if('accuracy' in record.keys()):
            if(mean_accuracy == True):
                count_of_accuracy_list[cur_userId] += 1
                accuracy_list[cur_userId] += record['accuracy']

        if(hours_of_use == True):
            hours_of_use_list[cur_userId] += record['experience'][-1]['t']

        for exp in record['experience']:
            count_of_exp_list[cur_userId] += 1

            if(exp['x'] == 'A'):
                if(count_of_test == True):
                    count_of_test_list[cur_userId] += 1
                if(mean_response_time == True and 's' in exp.keys()):
                    response_time_list[cur_userId] += exp['s']
            elif(exp['x'] == 'X'):
                if(count_of_learn == True):
                    count_of_learn_list[cur_userId] += 1
            
            if('w' in exp.keys()):
                cur_word = exp['w']
                if(isinstance(cur_word, numbers.Number)):
                    if(mean_accuracy_numbers == True):
                        count_of_numbers_list[cur_userId] += 1
                        if('m' not in exp.keys()):
                            correct_numbers_list[cur_userId] += 1
                else:
                    if('_' in cur_word):
                        ind = cur_word.find('_')
                        cur_category = cur_word[:ind]
                        cur_category = cur_category.lower()
                        if(cur_category == 'spot'):
                            if(mean_accuracy_spot == True):
                                count_of_spot_list[cur_userId] += 1
                                if('m' not in exp.keys()):
                                    correct_spot_list[cur_userId] += 1
                        elif(cur_category == 'phonics'):
                            if(mean_accuracy_phonics == True):
                                count_of_phonics_list[cur_userId] += 1
                                if('m' not in exp.keys()):
                                    correct_phonics_list[cur_userId] += 1
                        elif(cur_category == 'singular' or cur_category == 'plural'):
                            if(mean_accuracy_singplu == True):
                                count_of_singplu_list[cur_userId] += 1
                                if('m' not in exp.keys()):
                                    correct_singplu_list[cur_userId] += 1
                        elif(cur_category == 'phonemes'):
                            if(mean_accuracy_phonemes == True):
                                count_of_phonemes_list[cur_userId] += 1
                                if('m' not in exp.keys()):
                                    correct_phonemes_list[cur_userId] += 1
                        elif(cur_category == 'letters'):
                            if(mean_accuracy_letters == True):
                                count_of_letters_list[cur_userId] += 1
                                if('m' not in exp.keys()):
                                    correct_letters_list[cur_userId] += 1
                        elif(cur_category == 'abc'):
                            if(mean_accuracy_abc == True):
                                count_of_abc_list[cur_userId] += 1
                                if('m' not in exp.keys()):
                                    correct_abc_list[cur_userId] += 1
                        elif(cur_category == 'sight'):
                            if(mean_accuracy_sight == True):
                                count_of_sight_list[cur_userId] += 1
                                if('m' not in exp.keys()):
                                    correct_sight_list[cur_userId] += 1
                        else:
                            if(mean_accuracy_others == True):
                                count_of_others_list[cur_userId] += 1
                                if('m' not in exp.keys()):
                                    correct_others_list[cur_userId] += 1
                    else:
                        if(mean_accuracy_others == True):
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
    
    return res

