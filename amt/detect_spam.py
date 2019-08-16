from __future__ import print_function, absolute_import, unicode_literals, division

from collections import OrderedDict
import csv
from sklearn.metrics import accuracy_score

from amt.Test_data import get_miniclips_video_stats, print_stats_actions_miniclips, get_actions_stats

from amt.settings import *

import pandas as pd
import numpy as np


def compare_with_GT(dict_actions_hit, threshold):
    csv_file = open(PATH_output_file_name, 'r')
    reader = csv.DictReader(csv_file)

    with open(PATH_output_results_csv, 'w+') as csvfile3:
        fieldnames = ['HIT_nb', 'Video_name', 'Actions', 'Worker_1', 'Worker_2', 'Worker_3', 'Worker_GT']
        writer = csv.DictWriter(csvfile3, fieldnames=fieldnames)
        writer.writeheader()
        HIT_nb = ""
        Actions = ""
        Video_name = ""
        Worker_1 = ""
        Worker_2 = ""
        Worker_3 = ""
        worker_GT = ""
        for row in reader:
            ok_write = 0
            for (column_name, value) in row.items():
                if column_name == 'Video_name' and 'p' not in value.split("_")[0]:
                    Video_name = value
                    for (column_name, value) in row.items():
                        if column_name == 'Actions':
                            Actions = value
                        if column_name == 'HIT_nb':
                            HIT_nb = value
                        if column_name == 'Worker_1':
                            Worker_1 = value
                        if column_name == 'Worker_2':
                            Worker_2 = value
                        if column_name == 'Worker_3':
                            Worker_3 = value
                    ok_write = 1

                if ok_write == 1:
                    break
            if ok_write == 1:
                csv_file2 = open(PATH_GT_AMT, 'r')
                reader2 = csv.DictReader(csv_file2)
                for row2 in reader2:
                    ok = 0
                    if ('Video_name', Video_name) in row2.items() and ('Actions', Actions) in row2.items():
                        for (column_name, value) in row2.items():
                            if column_name == 'Worker_GT':
                                worker_GT = value
                                ok = 1
                                break
                    if ok == 1:
                        break
                csv_file2.close()
                writer.writerow({'HIT_nb': HIT_nb, 'Video_name': Video_name, 'Actions': Actions, 'Worker_1': Worker_1,
                                 'Worker_2': Worker_2, 'Worker_3': Worker_3, 'Worker_GT': worker_GT})

    # compute agreement GT vs. Worker1,  GT vs. Worker2,  GT vs. Worker3
    csv_file = open(PATH_output_results_csv, 'r')
    reader = csv.DictReader(csv_file)
    ok_1_equal_2 = 1
    list_results = []
    hit_nbs = set()
    for row in reader:
        for (column_name, value) in row.items():
            if "Worker_1" == column_name:
                value1 = int(value)
                if value1 == 2 and ok_1_equal_2 == 1:
                    value1 = 1
            if "Worker_2" == column_name:
                value2 = int(value)
                if value2 == 2 and ok_1_equal_2 == 1:
                    value2 = 1
            if "Worker_3" == column_name:
                value3 = int(value)
                if value3 == 2 and ok_1_equal_2 == 1:
                    value3 = 1
            if "Worker_GT" == column_name:
                value4 = int(value)
                if value4 == 2 and ok_1_equal_2 == 1:
                    value4 = 1
            if "HIT_nb" == column_name:
                hit_nb = value
                hit_nbs.add(hit_nb)

        list_results.append((hit_nb, value1, value2, value3, value4))

    per_hit_val_rater = dict()
    per_hit_val_rater[0] = []
    per_hit_val_rater[1] = []
    per_hit_val_rater[2] = []
    per_hit_val_rater['GT'] = []

    values_GT = OrderedDict()
    potential_spammers = OrderedDict()
    spammers_same_val = OrderedDict()
    spammers_low_GT_acc = OrderedDict()
    for key in hit_nbs:
        potential_spammers[key] = set()
        spammers_same_val[key] = set()
        spammers_low_GT_acc[key] = set()

    # ----- Verifiy in all the results if a worker put the same value everywhere --> spammer
    for hit in dict_actions_hit.keys():
        set_results_worker_1 = set()
        set_results_worker_2 = set()
        set_results_worker_3 = set()
        dict_video_names = dict_actions_hit[hit]
        for video_name in dict_video_names.keys():
            for action_result_list in dict_video_names[video_name]:
                result = action_result_list[1]
                if len(result) == 3:
                    set_results_worker_1.add(result[0])
                    set_results_worker_2.add(result[1])
                    set_results_worker_3.add(result[2])

        if len(set_results_worker_1) == 1:
            potential_spammers[hit].add(0)
            spammers_same_val[hit].add(0)
        if len(set_results_worker_2) == 1:
            potential_spammers[hit].add(1)
            spammers_same_val[hit].add(1)
        if len(set_results_worker_3) == 1:
            potential_spammers[hit].add(2)
            spammers_same_val[hit].add(2)

    hit = list_results[0][0]
    for (hit_nb, value1, value2, value3, value4) in list_results:
        if (hit_nb == hit):
            per_hit_val_rater[0].append(value1)
            per_hit_val_rater[1].append(value2)
            per_hit_val_rater[2].append(value3)
            per_hit_val_rater['GT'].append(value4)
        else:
            for worker in range(0, 3):
                if worker in potential_spammers[hit]:
                    continue
                else:
                    accuracy_with_GT = accuracy_score(per_hit_val_rater[worker], per_hit_val_rater['GT'])

                    if accuracy_with_GT < threshold:
                        potential_spammers[hit].add(worker)
                        spammers_low_GT_acc[hit].add(worker)
                        values_GT[hit] = per_hit_val_rater['GT']

            per_hit_val_rater[0] = [value1]
            per_hit_val_rater[1] = [value2]
            per_hit_val_rater[2] = [value3]
            per_hit_val_rater['GT'] = [value4]

            hit = hit_nb

    # compute for the last HIT
    for worker in range(0, 3):
        if worker in potential_spammers[hit_nb]:
            continue
        else:
            accuracy_with_GT = accuracy_score(per_hit_val_rater[worker], per_hit_val_rater['GT'])
            if accuracy_with_GT < 0.2:
                values_GT[hit_nb] = per_hit_val_rater['GT']
                potential_spammers[hit_nb].add(worker)
                spammers_low_GT_acc[hit_nb].add(worker)

    list_keys = dict()
    with open(PATH_input_file_name, 'r') as csvinput:
        for row in csv.reader(csvinput):
            key = row[0]
            if key == 'HITId':
                continue
            if key not in list_keys.keys():
                list_keys[key] = 0
            list_keys[key] += 1

    csvinput.close()

    list_bad_keys = []
    for key in list_keys.keys():
        if list_keys[key] != 3:
            list_bad_keys.append(key)

    index = 0
    index_row = 0
    with open(PATH_input_file_name, 'r') as csvinput:
        with open(PATH_spammers_files, 'w+') as csvoutput:
            writer = csv.writer(csvoutput)
            for row in csv.reader(csvinput):
                if index_row == 0:
                    writer.writerow(row)
                    index_row += 1
                    continue

                key = row[0]
                if key not in potential_spammers.keys() or potential_spammers[key] == set() or key in list_bad_keys:
                    row.append('x')
                    row.append('')

                else:
                    set_workers = potential_spammers[key]

                    if index in set_workers:
                        row.append('')
                        row.append('spammer')
                    else:
                        row.append('x')
                        row.append('')

                if key not in list_bad_keys:
                    index += 1
                if index == 3:
                    index = 0
                writer.writerow(row)

    csvinput.close()
    csvoutput.close()

    return potential_spammers


def get_compromised_hits(potential_spammers):
    compromised_hits = []  # hits with more than 1 spammer
    nb_spammers = 0
    nb_workers = 0
    for hit_nb in potential_spammers.keys():
        nb_workers += 3
        nb_spammers += len(potential_spammers[hit_nb])
        if len(potential_spammers[hit_nb]) > 1:
            compromised_hits.append(hit_nb)

    return compromised_hits, nb_spammers, nb_workers


def create_after_spam_filtered_results(dict_output, print_stats, threshold):
    potential_spammers = compare_with_GT(dict_output, threshold)

    compromised_hits, nb_spammers, nb_workers = get_compromised_hits(potential_spammers)

    csv_file = open(PATH_output_file_name, 'r')
    reader = csv.DictReader(csv_file)

    with open(PATH_after_spam_filter_csv, 'w+') as csvfile2:
        fieldnames = ['HIT_nb', 'Video_name', 'Actions', 'Worker_1', 'Worker_2', 'Worker_3', 'All_Yes_actions',
                      'Majority_Yes_actions', 'All_No_actions', 'Majority_No_actions']
        writer = csv.DictWriter(csvfile2, fieldnames=fieldnames)
        writer.writeheader()
        HIT_nb = ""
        Actions = ""
        Video_name = ""
        Worker_1 = ""
        Worker_2 = ""
        Worker_3 = ""
        All_Yes_actions = ""
        Majority_Yes_actions = ""
        All_No_actions = ""
        for row in reader:
            for (column_name, value) in row.items():
                if column_name == 'Actions':
                    Actions = value
                if column_name == 'Video_name':
                    Video_name = value
                if column_name == 'HIT_nb':
                    HIT_nb = value
                if column_name == 'Worker_1':
                    Worker_1 = value
                if column_name == 'Worker_2':
                    Worker_2 = value
                if column_name == 'Worker_3':
                    Worker_3 = value
                if column_name == 'All_Yes_actions':
                    All_Yes_actions = value
                if column_name == 'Majority_Yes_actions':
                    Majority_Yes_actions = value
                if column_name == 'Majority_No_actions':
                    Majority_No_actions = value

            if len(potential_spammers[HIT_nb]) == 0:
                writer.writerow({'HIT_nb': HIT_nb, 'Video_name': Video_name, 'Actions': Actions, 'Worker_1': Worker_1,
                                 'Worker_2': Worker_2, 'Worker_3': Worker_3, 'All_Yes_actions': All_Yes_actions,
                                 'Majority_Yes_actions': Majority_Yes_actions,
                                 'Majority_No_actions': Majority_No_actions})
            elif HIT_nb in compromised_hits:
                continue
            else:
                spammer = int(list(potential_spammers[HIT_nb])[0])
                if spammer == 0:
                    Worker_1 = -1
                    if Worker_2 != '0' or Worker_3 != '0':
                        Majority_No_actions = Actions
                        Majority_Yes_actions = ''
                elif spammer == 1:
                    Worker_2 = -1
                    if Worker_1 != '0' or Worker_3 != '0':
                        Majority_No_actions = Actions
                        Majority_Yes_actions = ''
                else:
                    Worker_3 = -1
                    if Worker_2 != '0' or Worker_1 != '0':
                        Majority_No_actions = Actions
                        Majority_Yes_actions = ''

                writer.writerow({'HIT_nb': HIT_nb, 'Video_name': Video_name, 'Actions': Actions, 'Worker_1': Worker_1,
                                 'Worker_2': Worker_2, 'Worker_3': Worker_3, 'All_Yes_actions': All_Yes_actions,
                                 'Majority_Yes_actions': Majority_Yes_actions,
                                 'Majority_No_actions': Majority_No_actions})

    csv_file.close()
    csvfile2.close()

    visible_actions, all_visible_actions, not_visible_actions = get_visible_and_not_visible_actions(
        PATH_after_spam_filter_csv, PATH_visible_not_visible_actions_csv)

    if print_stats == 1:
        print("There are {0} spammers out of {1} total workers".format(nb_spammers, nb_workers))
        print("There are {0} compromised hits (more than 1 spammer) out of a total of {1} hits.".format(
            len(compromised_hits), len(potential_spammers)))
        # print "Compromised hits: " + str(compromised_hits)

        nb_non_gt_videos, nb_non_gt_miniclips, nb_gt_videos, nb_gt_miniclips = get_miniclips_video_stats(
            PATH_after_spam_filter_csv)
        nb_total_actions, nb_majority_visible_actions, nb_all_visible_actions = get_actions_stats(
            PATH_after_spam_filter_csv)

        print("##--------------- After Spam Check --------------------------##")
        print_stats_actions_miniclips(nb_total_actions, nb_majority_visible_actions, nb_all_visible_actions,
                                      nb_non_gt_videos, nb_non_gt_miniclips, nb_gt_videos, nb_gt_miniclips)
    return potential_spammers


def get_visible_and_not_visible_actions(after_spam_filter_csv, visible_not_visible_actions_csv):
    df_data = pd.read_csv(after_spam_filter_csv)
    unique_data = df_data.drop_duplicates(subset=['Video_name', 'Actions'])

    video_names = unique_data['Video_name'].tolist()
    visible_actions = unique_data['Majority_Yes_actions'].tolist()
    all_visible_actions = unique_data['All_Yes_actions'].tolist()
    not_visible_actions = unique_data['Majority_No_actions'].tolist()

    df = pd.DataFrame({'Video_name': video_names,
                       'Visible Actions': visible_actions,
                       'Not Visible Actions': not_visible_actions})
    df.to_csv(visible_not_visible_actions_csv, sep=',', encoding='utf-8', index=False, header=True,
              columns=["Video_name", "Visible Actions", "Not Visible Actions"])

    return visible_actions, all_visible_actions, not_visible_actions
