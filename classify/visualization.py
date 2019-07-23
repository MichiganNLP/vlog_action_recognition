from __future__ import print_function, absolute_import, unicode_literals, division

from nltk import word_tokenize
import nltk
from tabulate import tabulate
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from collections import OrderedDict
import json
from scipy import stats
from nltk.stem.porter import *
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.style as style
import matplotlib.cm as cm
sns.set_context('talk')


class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def get_relative_increase(old, new):
    print(100.0 * (new - old) / old)


def get_ttest_significance(results_method1, results_method2):
    if type(results_method1) == np.ndarray:
        results_method1 = results_method1.tolist()
    if type(results_method2) == np.ndarray:
        results_method2 = results_method2.tolist()
    # verify is list is nested + flatten it if it is
    if any(isinstance(i, list) for i in results_method1):
        results_method1 = [item for sublist in results_method1 for item in sublist]
    if any(isinstance(i, list) for i in results_method2):
        results_method2 = [item for sublist in results_method2 for item in sublist]

    results_method1 = map(int, results_method1)
    results_method2 = map(int, results_method2)
    # print(results_method1, results_method2)
    print(stats.ttest_rel(results_method1, results_method2))


def get_list_actions_for_label(dict_video_actions, miniclip, label_type):
    list_type_actions = []
    list_action_labels = dict_video_actions[miniclip]
    for [action, label] in list_action_labels:
        if label == label_type:
            list_type_actions.append(action)
    return list_type_actions


def get_nb_visible_not_visible(dict_video_actions):
    nb_visible_actions = 0
    nb_not_visible_actions = 0
    for miniclip in dict_video_actions.keys():
        nb_visible_actions += len(get_list_actions_for_label(dict_video_actions, miniclip, 0))
        nb_not_visible_actions += len(get_list_actions_for_label(dict_video_actions, miniclip, 1))
    return nb_visible_actions, nb_not_visible_actions


def calculate_metrics(dict_results):
    dict_results_method = {}
    for method in dict_results.keys():
        print(color.UNDERLINE + "Results for " + color.PURPLE + color.BOLD + method + color.END)
        test_accuracy = []
        train_accuracy = []
        val_accuracy = []
        test_precision = []
        test_recall = []
        test_f1 = []
        for results in dict_results[method]:
            [acc_train, acc_val, acc_test, recall, precision, f1] = results
            test_accuracy.append(acc_test)
            train_accuracy.append(acc_train)
            val_accuracy.append(acc_val)
            test_precision.append(precision)
            test_recall.append(recall)
            test_f1.append(f1)

        n_repeats = len(dict_results[method])
        mean_test_accuracy = sum(test_accuracy) / float(n_repeats)
        mean_train_accuracy = sum(train_accuracy) / float(n_repeats)
        mean_val_accuracy = sum(val_accuracy) / float(n_repeats)
        mean_test_recall = sum(test_recall) / float(n_repeats)
        mean_test_precision = sum(test_precision) / float(n_repeats)
        mean_test_f1 = sum(test_f1) / float(n_repeats)
        standard_error = np.std(test_accuracy) / np.sqrt(np.ma.count(test_accuracy))
        interval = standard_error * 1.96  # the interval of 95% is (1.96 * standard_error) around the mean results.

        lower_interval = mean_test_accuracy - interval
        upper_interval = mean_test_accuracy + interval

        create_table(['Mean Test Acc', 'Std error', 'Lower interval', 'Upper interval'],
                     [[mean_test_accuracy, standard_error, lower_interval, upper_interval]])

        dict_results_method[method] = [[mean_train_accuracy, mean_val_accuracy, mean_test_accuracy, mean_test_recall,
                                        mean_test_precision, mean_test_f1, standard_error, lower_interval,
                                        upper_interval]]
    return dict_results_method


def calculate_significance_between_2models(dict_mean_results_method):
    model1 = dict_mean_results_method.keys()[0]
    model2 = dict_mean_results_method.keys()[1]

    [[_, _, _, _, _, _, _, lower_interval1, upper_interval1]] = dict_mean_results_method[
        model1]
    [[_, _, _, _, _, _, _, lower_interval2, upper_interval2]] = dict_mean_results_method[
        model2]

    if lower_interval1 <= upper_interval2 and lower_interval2 <= upper_interval1:
        # intervals overlap
        print(
            "Models " + color.PURPLE + color.BOLD + model1 + color.END + " and " + color.PURPLE + color.BOLD + model2 + color.END + " are NOT significantly different")
    else:
        print(
            "Models " + color.BLUE + color.BOLD + model1 + color.END + " and " + color.BLUE + color.BOLD + model2 + color.END + " are significantly different")


def print_scores_per_method(dict_results):
    headers = ['Method', 'Train Accuracy', 'Val Accuracy', 'Test Accuracy', 'Test Recall', 'Test Precision', 'Test F1']
    list_results = []
    for method in dict_results.keys():
        for results in dict_results[method]:
            if len(results) == 6:
                [acc_train, acc_val, acc_test, recall, precision, f1] = results
            else:
                [acc_train, acc_val, acc_test, recall, precision, f1, _, _, _] = results
            list_results.append([method, acc_train, acc_val, acc_test, recall, precision, f1])

    create_table(headers, list_results)


def print_t_test_significance(dict_significance):
    method1 = dict_significance.keys()[0]
    method2 = dict_significance.keys()[1]
    print("T-test Significance for {0} and {1}:".format(method1, method2))
    predicted_method1 = dict_significance[method1]
    predicted_method2 = dict_significance[method2]
    get_ttest_significance(predicted_method1, predicted_method2)


def group_action_by_verb(unique_visibile_actions):
    lemma = nltk.wordnet.WordNetLemmatizer()
    stemmer = PorterStemmer()
    dict_verb_actions = {}
    for action in unique_visibile_actions:
        tokens = word_tokenize(action)
        pos_text = nltk.pos_tag(tokens)

        for (word, pos) in pos_text:
            if 'VB' in pos:
                word = lemma.lemmatize(word)
                if word in ['i', 'oh', 'red', 'bed']:
                    continue
                if word[-3:] == 'ing':
                    word = stemmer.stem(word)
                if word == 'ad':
                    word = 'add'
                if word == 'drizzl':
                    word = 'drizzle'
                if word == 'tri':
                    word = 'try'
                if word == 'saut':
                    word = 'saute'
                if word == 'cooked':
                    word = 'cook'
                if word == 'fri':
                    word = 'fry'
                if word == 'danc':
                    word = 'dance'
                if word == 'hydrat':
                    word = 'hydrate'

                if word not in dict_verb_actions.keys():
                    dict_verb_actions[word] = []
                dict_verb_actions[word].append(action)

    ordered_d = OrderedDict(sorted(dict_verb_actions.viewitems(), key=lambda x: len(x[1])))
    with open("data/dict_verb_actions.json", 'w') as f:
        json.dump(ordered_d, f)
    print("For visibile actions, nb of different verbs: {0}".format(len(ordered_d.keys())))


def find_miniclip_by_not_visible_action(dict_video_actions, label_type, action_to_search):
    list_miniclips = []
    for miniclip in dict_video_actions.keys():
        list_action_labels = dict_video_actions[miniclip]
        for [action, label] in list_action_labels:
            if label == label_type and action == action_to_search:
                list_miniclips.append(miniclip)
                continue
    return list_miniclips


def measure_nb_unique_actions(dict_video_actions):
    all_visibile_actions = []
    all_not_visibile_actions = []

    for miniclip in dict_video_actions.keys():
        visibile_actions = get_list_actions_for_label(dict_video_actions, miniclip, 0)
        not_visibile_actions = get_list_actions_for_label(dict_video_actions, miniclip, 1)

        all_visibile_actions = all_visibile_actions + visibile_actions
        all_not_visibile_actions = all_not_visibile_actions + not_visibile_actions

    for action in all_visibile_actions:
        if action in all_not_visibile_actions:
            # print (action)
            visibile_in_miniclips = find_miniclip_by_not_visible_action(dict_video_actions, 0, action)
            not_visibile_in_miniclips = find_miniclip_by_not_visible_action(dict_video_actions, 1, action)

            both_miniclips_visibile_not_visibile = set(visibile_in_miniclips).intersection(
                set(not_visibile_in_miniclips))
            # if len(both_miniclips_visibile_not_visibile):
            #     print(action)
            #     print ("visible and not in miniclips: %", both_miniclips_visibile_not_visibile)

            # print ("visible in miniclips: %", visibile_in_miniclips)
            # print ("not visible in miniclips: %", not_visibile_in_miniclips)

    unique_all_actions = set(all_not_visibile_actions + all_visibile_actions)
    unique_visibile_actions = set(all_visibile_actions)
    unique_not_visibile_actions = set(all_not_visibile_actions)

    group_action_by_verb(unique_visibile_actions)

    # for visibile_action in all_visibile_actions:
    #     print(visibile_action)

    both_visibile_not_visibile = unique_not_visibile_actions.intersection(unique_visibile_actions)
    # print(both_visibile_not_visibile)
    print("Number unique visible actions that can be not-visibile: {0}".format(len(both_visibile_not_visibile)))
    print("Number unique visible actions: {0}, not visibile: {1}, both: {2}".format(len(unique_visibile_actions),
                                                                                    len(unique_not_visibile_actions),
                                                                                    len(unique_all_actions)))


def print_nb_actions_miniclips_train_test_eval(dict_train_data, dict_test_data, dict_val_data):
    nb_train_actions_visible, nb_train_actions_not_visible = get_nb_visible_not_visible(dict_train_data)
    nb_train_actions = nb_train_actions_visible + nb_train_actions_not_visible

    nb_test_actions_visible, nb_test_actions_not_visible = get_nb_visible_not_visible(dict_test_data)
    nb_test_actions = nb_test_actions_visible + nb_test_actions_not_visible

    nb_val_actions_visible, nb_val_actions_not_visible = get_nb_visible_not_visible(dict_val_data)
    nb_val_actions = nb_val_actions_visible + nb_val_actions_not_visible

    print(tabulate([['nb_actions', nb_train_actions, nb_test_actions, nb_val_actions],
                    ['nb_miniclips', len(dict_train_data.keys()), len(dict_test_data.keys()),
                     len(dict_val_data.keys())]], headers=['', 'Train', 'Test', 'Eval'], tablefmt='grid'))


def call_print(balance, before_balance, string_to_print):
    if before_balance:
        string_to_print = "# --- Before Balance: " + str(string_to_print)
    if balance == "upsample":
        string_to_print = "# --- After upsample: " + str(string_to_print)
    elif balance == "downsample":
        string_to_print = "# --- After downsample: " + str(string_to_print)

    return string_to_print


def print_bar_plots():
    # set width of bar
    barWidth = 0.15

    # set height of bar
    bars6 = [65.4] # yolo
    bars7 = [71.2]  # concreteness
    bars1 = [74.5, 75.5, 76.7] # action
    # bars1 = [74.5, 75.5, 76.7, 76.1] # action
    bars0 = [76.1]
    bars2 = [73.0, 74.6, 75.7, 75.8] # action + pos
    bars3 = [75.2, 75.7, 75.9, 76.1] # action + context
    bars4 = [74.7, 75.4, 75.6, 75.9] # action + concreteness
    bars5 = [74.3, 74.4, 75.6, 76.4] # action + all

    error6 = [1.7]# yolo
    error7 = [2.6]# concreteness
    error1 = [2.1, 1.4, 1.6]  # action
    error0 = [1.8]
    error2 = [2.4, 1.6, 1.7, 1.4]  # action + pos
    error3 = [2.2, 1.7, 1.9, 1.7]  # action + context
    error4 = [2.1, 1.4, 1.6, 1.5]  # action + concreteness
    error5 = [2.2, 1.4, 1.6, 1.5]  # action + all


    # Set position of bar on X axis
    r1 = np.arange(len(bars1))
    rr = np.arange(len(bars1) + 1)
    r2 = [x + barWidth for x in rr]
    r3 = [x + barWidth for x in r2]
    r4 = [x + barWidth for x in r3]
    r5 = [x + barWidth for x in r4]

    r6 = [r5[-1] + 2 * barWidth]
    r7 = [r6[-1] + barWidth]

    r0 = [r1[-1] + 6.7 * barWidth]

    # Make the plot
    plt.axhline(y=71.1, color='black', linestyle='--')

    x = np.arange(8)
    ys = [i + x + (i * x) ** 2 for i in range(8)]

    colors = cm.rainbow(np.linspace(0, 1, len(ys)))


    plt.bar(r1, bars1, yerr=error1, color=colors[0], width=barWidth, edgecolor='white',capsize=10, label='Action')
    plt.bar(r0, bars0, yerr=error0, color=colors[1], width=barWidth, edgecolor='white', capsize=10, label='Action + Visual Feat.')
    plt.bar(r2, bars2, yerr=error2, color=colors[2], width=barWidth, edgecolor='white',capsize=10, label='+POS')
    plt.bar(r3, bars3, yerr=error3, color=colors[3], width=barWidth, edgecolor='white',capsize=10, label='+Context')
    plt.bar(r4, bars4, yerr=error4, color=colors[4], width=barWidth, edgecolor='white',capsize=10, label='+Concreteness')
    plt.bar(r5, bars5, yerr=error5, color=colors[5], width=barWidth, edgecolor='white',capsize=10, label='+All')
    plt.bar(r6, bars6, yerr=error6, color=colors[6], width=barWidth, edgecolor='white', capsize=10,  label='Object Detection')
    # plt.bar(r7, bars7, yerr=error7, color=colors[7], width=barWidth, edgecolor='white',capsize=10,  label='Concreteness')

    # Add xticks on the middle of the group bars
    plt.xlabel('Method', fontweight='bold')
    plt.ylabel('Accuracy', fontweight='bold')
    # plt.xticks([0], ['YOLO'])
    plt.xticks([r + barWidth for r in range(len(bars2))], ['SVM', 'LSTM', 'ELMO', 'MULTIMODAL', 'Object Detection', 'Concreteness'])
    plt.tick_params(axis='both', which='major', labelsize=20)
    # Create legend & Show graphic
    plt.legend(loc=4, prop={'size': 20})
    #plt.savefig('data/plots/bar7.pdf', format='svg', dpi=1200)

    # for i, v in enumerate(bars1):
    #     plt.text(i - .15,
    #               v + 3,
    #               bars1[i],
    #             fontsize=20,
    #               color='black')
    # for i, v in enumerate(bars0):
    #     plt.text(i + 2.9,
    #              v + 3,
    #              bars0[i],
    #              fontsize=20,
    #              color='black')

    #
    # for i, v in enumerate(bars2):
    #     plt.text(i + 0.05,
    #               v + 3,
    #               bars2[i],
    #             fontsize=18,
    #               color='black')
    #
    # for i, v in enumerate(bars3):
    #     plt.text(i + .23,
    #               v + 3,
    #               bars3[i],
    #             fontsize=18,
    #               color='black')
    #
    # for i, v in enumerate(bars4):
    #     plt.text(i + .43,
    #               v + 3,
    #               bars4[i],
    #             fontsize=18,
    #               color='black')
    # for i, v in enumerate(bars5):
    #     plt.text(i + .55,
    #               v + 3,
    #               bars5[i],
    #             fontsize=20,
    #               color='black')
    #
    # for i, v in enumerate(bars6):
    #     plt.text(i + .75,
    #               v + 3,
    #               bars5[i],
    #             fontsize=20,
    #               color='black')
    #
    # for i, v in enumerate(bars7):
    #     plt.text(i + .85,
    #               v + 3,
    #               bars5[i],
    #             fontsize=20,
    #               color='black')


    plt.show()

    #  fig, ax = plt.subplots()
    #  # labels = ['MAJORITY', 'SVM', 'LSTM', 'ELMo', 'MULTIMODAL']
    #  labels = ['MAJORITY', 'SVM']
    #  x = np.arange(len(labels))
    #  y = [71.1, 75.2]
    #  # y = [71.1, 75.2, 75.5, 76.7, 76.4]
    #  error = [1.3, 2.2]
    #  # error = [1.3, 2.2, 1.4, 1.6, 1.5]
    #
    #  width = 0.35  # the width of the bars
    #
    #  # color_dict = {'MAJORITY': 'aquamarine', 'SVM': 'cadetblue', 'LSTM': 'darkcyan', 'ELMO': 'darkolivegreen', 'MULTIMODAL': 'darkgreen'}
    #  # rect_main = ax.bar(x - width / 4, y, yerr=error, align='center', alpha=0.9, ecolor='black',color=[color_dict[r] for r in labels], capsize=10, label = 'action')
    #
    #  rect_main = ax.bar(x - width / 2, y, yerr=error,color='aquamarine', capsize=10, label = 'Action')
    #  rect_pos = ax.bar(x + width / 2, y, yerr=error, color='cadetblue', capsize=10, label = 'POS')
    #
    #  #rect_context = ax.bar(x, y, yerr=error, align='center', alpha=0.9, ecolor='black',color='darkcyan', capsize=10, label = 'Context')
    # # rect_concretness = ax.bar(x + width / 2, y, yerr=error, align='center', alpha=0.9, ecolor='black',color='darkolivegreen', capsize=10, label = 'Concreteness')
    #
    #  ax.set_ylabel('Accuracy')
    #  ax.set_xticks(x)
    #  ax.set_xticklabels(labels)
    #  ax.legend()

    # ax.yaxis.grid(True)
    # for i, v in enumerate(y):
    #     ax.text(i - .25,
    #               # v / y[i] + 75,
    #               v + 3,
    #               y[i],
    #             fontsize=17,
    #               color='black')


def print_action_balancing_stats(balance, before_balance, dict_video_actions, dict_train_data, dict_test_data,
                                 dict_val_data, test_data):
    string_to_print = "in total there are {0} visible actions and {1} not visible"
    string_to_print = call_print(balance, before_balance, string_to_print)
    nb_visible_actions, nb_not_visible_actions = get_nb_visible_not_visible(dict_video_actions)
    print(string_to_print.format(nb_visible_actions, nb_not_visible_actions))

    string_to_print = "in train there are {0} visible actions and {1} not visible"
    string_to_print = call_print(balance, before_balance, string_to_print)
    train_nb_visible_actions, train_nb_not_visible_actions = get_nb_visible_not_visible(dict_train_data)
    print(string_to_print.format(train_nb_visible_actions, train_nb_not_visible_actions))

    string_to_print = "in test there are {0} visible actions and {1} not visible"
    string_to_print = call_print(balance, before_balance, string_to_print)
    test_nb_visible_actions, test_nb_not_visible_actions = get_nb_visible_not_visible(dict_test_data)
    print(string_to_print.format(test_nb_visible_actions, test_nb_not_visible_actions))

    string_to_print = "in val there are {0} visible actions and {1} not visible"
    string_to_print = call_print(balance, before_balance, string_to_print)
    val_nb_visible_actions, val_nb_not_visible_actions = get_nb_visible_not_visible(dict_val_data)
    print(string_to_print.format(val_nb_visible_actions, val_nb_not_visible_actions))

    most_common_label = 0 if train_nb_visible_actions > train_nb_not_visible_actions else 1

    predicted = (test_nb_visible_actions + test_nb_not_visible_actions) * [most_common_label]
    test_labels = [label for (video, action, label) in test_data]

    acc = accuracy_score(test_labels, predicted)
    f1 = f1_score(test_labels, predicted)
    recall = recall_score(test_labels, predicted)
    precision = precision_score(test_labels, predicted)

    print("# Most common label Test: Acc: {0}, Precision:{1}, Recall:{2}, F1:{3} ".format(acc, precision, recall, f1))

    # print("# Baseline Acc Most common label Train: {0} ".format(1 - 1.0 * nb_visible_actions / (nb_not_visible_actions + nb_visible_actions)))
    # print("# Baseline Acc Most common label Val: {0} ".format(1 - 1.0 * val_nb_visible_actions / (val_nb_not_visible_actions + val_nb_visible_actions)))
    # print("# Baseline Acc Most common label Test: {0} ".format(1 - 1.0 * test_nb_visible_actions / (test_nb_not_visible_actions + test_nb_visible_actions)))


def create_table(headers, list_all_results):
    final_list = []
    for elem in list_all_results:
        list_results = []
        list_results += elem
        final_list.append(list_results)

    print(tabulate(final_list, headers=headers, tablefmt='orgtbl'))


def create_table_concreteness():
    # Visible actions with low concreteness score
    final_list = [['give them a really nice full look', 2.96, 'look'],
                  ['put the instructions on how', 2.5, 'put'],
                  ['give them a really nice full look', 2.96, 'look'],
                  ['make my markings', 2.67, 'make'],
                  ['using this', 2.78, 'using'],
                  ['make something hearty', 2.78, 'make'],
                  ['making diy', 2.34, 'making'],
                  ['do this in their home', 2.46, 'do'],
                  ]
    print(' Examples visible actions with low concreteness score:')
    print(tabulate(final_list, headers=['Action', 'Score', 'Verb / Noun'], tablefmt='orgtbl'))

    # Not visible actions with high concreteness score:

    final_list = [['making a rustic necklace', 4.96, 'necklace'],
                  ['throw away', 4.04, 'throw'],
                  ['found this great piece of wood', 4.85, 'wood'],
                  ['chopping the wood i', 4.85, 'wood'],
                  ['use this method daily shower cleaner even though i', 4.89, 'shower'],
                  ['get to bed sometimes i', 5.0, 'bed'],
                  ['throw my hair in braids', 4.97, 'hair'],
                  ['do every single night depending on how tired i', 5.0, 'tired'],
                  ['making a diy dog toy this', 4.93, 'toy'],
                  ]
    print("\n")
    print(' Examples of not visible actions with high concreteness score:')
    print(tabulate(final_list, headers=['Action', 'Score', 'Verb / Noun'], tablefmt='orgtbl'))


if __name__ == '__main__':
    print_bar_plots()
    # headers = ['', 'Method', 'Test Accuracy', 'Test Recall', 'Test Precision', 'Test F1']
    # dict_results = [['SVM + GloVe', 0.69, 0.7, 0.69, 0.67],
    #                 ['SVM + GloVe + pos_tag', 0.69, 0.67, 0.71, 0.69],
    #                 ['SVM + GloVe + context', 0.67, 0.65, 0.69, 0.67],
    #                 ['SVM + GloVe + pos_tag + context', 0.67, 0.65, 0.7, 0.67],
    #                 ['LSTM + GloVe', 0.64, 0.61, 0.76, 0.67],
    #                 ['ELMO', 0.62, 0.6, 0.67, 0.63],
    #                 ['3 Dense layers + GloVe', 0.67, 0.64, 0.74, 0.68],
    #                 ['concreteness score', 0.61, 0.79, 0.59, 0.68],
    #                 ['YOLO v3 + wup-similarity', 0.57, 0.53, 0.64, 0.58],
    #                 ['Inception V3 features + GloVe', '-', '-', '-', '-']
    #                 ]
    # create_table(headers, dict_results)
    #
    # create_table_concreteness()
