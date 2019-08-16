from __future__ import print_function, absolute_import, unicode_literals, division

import os
import csv
from collections import OrderedDict
import collections
from sklearn.metrics import cohen_kappa_score, accuracy_score
import subprocess
import pandas as pd

GLOBAL_NB_NONGT_VIDEOS = 0
GLOBAL_NB_NONGT_MINICLIPS = 0
GLOBAL_NB_GT_VIDEOS = 0
GLOBAL_NB_GT_MINICLIPS = 0
list_results_hit = []


def getLength(filename):
    result = subprocess.Popen(["ffprobe", filename],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return [x.split(',')[0].split('Duration: ')[1] for x in result.stdout.readlines() if "Duration" in x][0]


def compute_agreement(rater1, rater2, rater3=None):
    if (rater3 == None):
        kappa_ratingtask = cohen_kappa_score(rater1, rater2)
    else:

        n = 3  # nb of raters
        N = len(rater1)  # nb of subjects
        k = 2  # nb of categories (0 or 1)
        p = [0, 0]  # the proportion of all assignments which were to the j-th category
        P = []  # the extent to which raters agree for the i-th subject (i.e., compute how many rater--rater pairs are in agreement, relative to the number of all possible rater--rater pairs):
        # with open('test.csv', 'wb+') as csvfile:
        #     spamwriter = csv.writer(csvfile)
        for subject in range(0, N):
            nb_raters_0 = 0
            nb_raters_1 = 0
            if rater1[subject] == 0:
                nb_raters_0 += 1
            else:
                nb_raters_1 += 1
            if rater2[subject] == 0:
                nb_raters_0 += 1
            else:
                nb_raters_1 += 1
            if rater3[subject] == 0:
                nb_raters_0 += 1
            else:
                nb_raters_1 += 1

            # spamwriter.writerow([nb_raters_0, nb_raters_1])
            nb = 1.0 * (nb_raters_0 * nb_raters_0 + nb_raters_1 * nb_raters_1 - n) / (n * (n - 1))
            P.append(nb)
            p[0] += nb_raters_0
            p[1] += nb_raters_1

        p[0] = 1.0 * p[0] / (N * n)
        p[1] = 1.0 * p[1] / (N * n)

        P_mean = 1.0 * sum(P) / len(P)
        p_e = p[0] * p[0] + p[1] * p[1]

        # variation is 0 (division by 0) if all turkers answered the same
        if p_e == 1:
            kappa_ratingtask = 1
        else:
            kappa_ratingtask = 1.0 * (P_mean - p_e) / (1 - p_e)

    if (kappa_ratingtask <= 0):
        interpretation = "less than chance"
    elif (kappa_ratingtask > 0 and kappa_ratingtask <= 0.2):
        interpretation = "slight"
    elif (kappa_ratingtask > 0.2 and kappa_ratingtask <= 0.40):
        interpretation = "fair"
    elif (kappa_ratingtask > 0.4 and kappa_ratingtask <= 0.60):
        interpretation = "moderate"
    elif (kappa_ratingtask > 0.6 and kappa_ratingtask <= 0.80):
        interpretation = "substantial"
    else:
        interpretation = "almost perfect"

    print("kappa " + str(kappa_ratingtask) + ", " + interpretation)

    return interpretation


def get_list_interpretations(csv_file_name):
    csv_file = open(csv_file_name, 'r')
    reader = csv.DictReader(csv_file)

    hit_id = "3GMLHYZ0LDWP3ZYFSPKOU3VB7XAYUY"
    index_hit = 1
    rater_answers = []
    index = 0

    list_interpretations = []
    for row in reader:
        index += 1
        answer = []
        for (column_name, value) in row.items():
            if column_name == "HITId":
                if value != hit_id:
                    hit_id = value
                    index_hit += 1
                    rater_answers = []

            if "Answer.name" in column_name:
                if value != "nothing" and value != '':
                    val, video_id, action_id, result = value.split('_')
                    video_id = int(video_id)
                    action_id = int(action_id)
                    result = int(result)
                    answer.append(result)

        rater_answers.append(answer)

        if (index % 3 == 0):
            print("\n------------------")
            print("Agreement for HIT " + str(index_hit) + ":")
            interpretation = compute_agreement(rater_answers[0], rater_answers[1], rater_answers[2])
            list_interpretations.append(interpretation)

    counter = collections.Counter(list_interpretations)
    print(counter)
    csv_file.close()


# Random sampling for verification
def TEST_results_input_AMT(list_hits_to_verify, dict_actions_output):
    print("\n----------------------- Random Sampling ---------------------\n")
    for hit in list_hits_to_verify:
        print(hit)
        list_values = dict_actions_output[hit][1]
        index_video = 0
        for video_name in dict_actions_output[hit][0].keys():
            for sublist_actions in dict_actions_output[hit][0][video_name]:
                index_action = 0
                for action in sublist_actions:
                    results_per_worker = []
                    for index_worker in range(0, len(list_values)):
                        results_worker = list_values[index_worker]
                        for result in results_worker:
                            if result != "":
                                _, result_index_video, result_index_action, result_value = result.split("_")
                                if str(result_index_video) == str(index_video) and str(result_index_action) == str(
                                        index_action):
                                    results_per_worker.append(result_value)
                                    break

                    print(str(index_video), str(index_action), results_per_worker, video_name, action)
                    index_action += 1
                index_video += 1
        print("\n-----------------------------------------------\n")


def read_results_from_AMT(csv_file_name):
    csv_file = open(csv_file_name, 'r')
    reader = csv.DictReader(csv_file)

    dict_actions_output = OrderedDict()

    hit_id = ""
    list_triples_values = []
    list_video_names = [''] * 5
    list_actions = [''] * 5
    list_triples_values = []

    index_row = 1
    num_lines = sum(1 for line in open(csv_file_name))

    for row in reader:
        index_row += 1
        list_values = []

        for (column_name, value) in row.items():

            if "Answer.name" in column_name:
                list_values.append(value)

            if column_name == "HITId":
                hit_new_id = value
                if index_row == 2:
                    hit_id = hit_new_id
                if hit_new_id != hit_id or index_row == 2:
                    for (column_name, value) in row.items():
                        if "Input.video_url" in column_name:
                            video_id = int(column_name[-1]) - 1
                            list_video_names[video_id] = value

                    for (column_name, value) in row.items():
                        if "Input.actions" in column_name:
                            action_id = int(column_name[-1]) - 1
                            list_actions[action_id] = value.split(";")

                    if hit_id not in dict_actions_output.keys():
                        dict_actions_output[hit_id] = []
                    if list_triples_values != []:
                        dict_actions_output[hit_id].append(list_triples_values)

                    dict_video_names = OrderedDict()
                    if hit_new_id not in dict_actions_output.keys():
                        dict_actions_output[hit_new_id] = []

                    for video_index in range(0, len(list_video_names)):
                        sublist_actions = list_actions[video_index]
                        video_name = list_video_names[video_index]

                        if video_name not in dict_video_names.keys():
                            dict_video_names[video_name] = []
                        dict_video_names[video_name].append(sublist_actions)

                    dict_actions_output[hit_new_id].append(dict_video_names)

                    if index_row != num_lines:
                        list_triples_values = []

                hit_id = hit_new_id

        list_triples_values.append(list_values)

    if hit_id not in dict_actions_output.keys():
        dict_actions_output[hit_id] = []
    dict_actions_output[hit_id].append(list_triples_values)

    ## Save all in a dictionary
    dict_output = OrderedDict()
    for hit_id in dict_actions_output.keys():
        list_values = dict_actions_output[hit_id][1]
        index_video = 0
        dict_video_names = OrderedDict()
        for video_name in dict_actions_output[hit_id][0].keys():
            for sublist_actions in dict_actions_output[hit_id][0][video_name]:
                index_action = 0
                for action in sublist_actions:
                    results_per_worker = []
                    for index_worker in range(0, len(list_values)):
                        results_worker = list_values[index_worker]
                        for result in results_worker:
                            if result != "":
                                _, result_index_video, result_index_action, result_value = result.split("_")
                                if str(result_index_video) == str(index_video) and str(result_index_action) == str(
                                        index_action):
                                    results_per_worker.append(int(result_value))
                                    break

                    if video_name not in dict_video_names.keys():
                        dict_video_names[video_name] = []
                    dict_video_names[video_name].append([action, results_per_worker])
                    index_action += 1

                index_video += 1

        dict_output[hit_id] = dict_video_names

        ## Random sampling for verification
    # list_hits_to_verify = ['3VIVIU06FJBK0RE1F57ZH5KT8XRMIK', '37YYO3NWHCPGE2GAA6HZ36HM1E3CCQ','3KL228NDMULBEC8345UGHGF94F4KGL','3S829FDFT10EMSXJ1Y8X1PKENW3DXT',
    # '3PKVGQTFIGJXN76YOVPJPS8R43NYRP', '3IWA71V4THFF1JI4RB0JVJAM6AG6XU', '3UUIU9GZC44C1Y96HIPHEABGXJH5TU']
    # TEST_results_input_AMT(list_hits_to_verify, dict_actions_output)

    # for hit_id in dict_output:
    #     if hit_id in list_hits_to_verify:
    #         print "-----------------------------------------------\n"
    #         dict_video_names = dict_output[hit_id]
    #         print "\n---------------" + hit_id
    #         for video_name in dict_video_names.keys():
    #             print "\n---------------" + video_name
    #             for action_result_list in dict_video_names[video_name]:
    #                 action = action_result_list[0]
    #                 result = action_result_list[1]
    #                 print action, result
    return dict_output, dict_actions_output


def get_action_names_per_video(csv_file_name):
    csv_file = open(csv_file_name, 'r')
    reader = csv.DictReader(csv_file)

    list_actions = OrderedDict()
    list_video_names = []
    set_videos = set()
    set_miniclips = set()
    set_videos_GT = set()
    set_miniclips_GT = set()

    for row in reader:

        for (column_name, value) in row.items():

            for i in range(1, 6):
                if "Input.video_url" + str(i) in column_name and value != "nothing":
                    if i < 5:
                        channel_id, video_id, miniclip_id = value.split("_")
                        set_videos.add(channel_id + "_" + video_id)
                        set_miniclips.add(value)
                    else:
                        video_id, miniclip_id = value.split("_")
                        set_videos_GT.add(video_id)
                        set_miniclips_GT.add(value)

    csv_file.close()
    GLOBAL_NB_NONGT_VIDEOS = len(set_videos)
    GLOBAL_NB_NONGT_MINICLIPS = len(set_miniclips)
    GLOBAL_NB_GT_VIDEOS = len(set_videos_GT)
    GLOBAL_NB_GT_MINICLIPS = len(set_miniclips_GT)

    print("There are " + str(GLOBAL_NB_NONGT_VIDEOS) + " NON-GT videos and " + str(
        GLOBAL_NB_NONGT_MINICLIPS) + " miniclips.")
    print("There are " + str(GLOBAL_NB_GT_VIDEOS) + " GT videos and " + str(GLOBAL_NB_GT_MINICLIPS) + " miniclips.")
    print("In total, there are " + str(GLOBAL_NB_NONGT_VIDEOS + GLOBAL_NB_GT_VIDEOS) + " videos and " + str(
        GLOBAL_NB_NONGT_MINICLIPS + GLOBAL_NB_GT_MINICLIPS) + " miniclips.")

    return GLOBAL_NB_NONGT_VIDEOS, GLOBAL_NB_NONGT_MINICLIPS, GLOBAL_NB_GT_VIDEOS, GLOBAL_NB_GT_MINICLIPS


def write_output_file(output_file_name, dict_output):
    hit_id = dict_output.keys()[0]
    video_name = dict_output[hit_id].keys()[0]
    result = dict_output[hit_id][video_name]
    nb_turks_per_hit = len(result[0][1])

    with open(output_file_name, 'w+') as csvfile:
        if (nb_turks_per_hit == 3):
            fieldnames = ['HIT_nb', 'Video_name', 'Actions', 'Worker_1', 'Worker_2', 'Worker_3', 'All_Yes_actions',
                          'Majority_Yes_actions']
        elif (nb_turks_per_hit == 1):
            fieldnames = ['HIT_nb', 'Video_name', 'Actions', 'Worker_1', 'All_Yes_actions']
        else:
            print("Different number of turkers!!")

        nb_all_visible_actions = 0
        nb_majority_visible_actions = 0
        nb_total_actions = 0

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for hit_id in dict_output.keys():
            dict_video_names = dict_output[hit_id]
            # get only non-gt videos : dict_video_names.keys()[:-1]
            for video_name in dict_video_names.keys():
                for action_result_list in dict_video_names[video_name]:
                    action = action_result_list[0]
                    result_list = action_result_list[1]
                    nb_turks_per_hit = len(result_list)

                    if nb_turks_per_hit == 1:
                        result_1 = result_list[0]
                        if (result_1 == 0):
                            writer.writerow(
                                {'HIT_nb': hit_id, 'Video_name': video_name, 'Actions': action, 'Worker_1': result_1,
                                 'All_Yes_actions': action})
                        else:
                            writer.writerow(
                                {'HIT_nb': hit_id, 'Video_name': video_name, 'Actions': action, 'Worker_1': result_1})

                    elif nb_turks_per_hit == 3:
                        if video_name != dict_video_names.keys()[-1]:
                            nb_total_actions += 1
                        [result_1, result_2, result_3] = result_list
                        if result_1 == result_2 == result_3 == 0:
                            if video_name != dict_video_names.keys()[-1]:
                                nb_majority_visible_actions += 1
                                nb_all_visible_actions += 1
                            writer.writerow(
                                {'HIT_nb': hit_id, 'Video_name': video_name, 'Actions': action, 'Worker_1': result_1,
                                 'Worker_2': result_2, 'Worker_3': result_3, 'All_Yes_actions': action,
                                 'Majority_Yes_actions': action})
                        elif (result_1 == result_2 == 0 or result_1 == result_3 == 0 or result_3 == result_2 == 0):
                            if video_name != dict_video_names.keys()[-1]:
                                nb_majority_visible_actions += 1
                            writer.writerow(
                                {'HIT_nb': hit_id, 'Video_name': video_name, 'Actions': action, 'Worker_1': result_1,
                                 'Worker_2': result_2, 'Worker_3': result_3, 'Majority_Yes_actions': action})
                        else:
                            writer.writerow(
                                {'HIT_nb': hit_id, 'Video_name': video_name, 'Actions': action, 'Worker_1': result_1,
                                 'Worker_2': result_2, 'Worker_3': result_3})


def compare_results(csv_file_name1, csv_file_name2):
    csv_file1 = open(csv_file_name1, 'r')
    reader = csv.DictReader(csv_file1)

    set_videos1 = set()
    for row in reader:
        for (column_name, value) in row.items():
            if ('Input.video_url' in column_name and value != "nothing"):
                set_videos1.add(value)
    csv_file1.close()

    csv_file2 = open(csv_file_name2, 'r')
    reader = csv.DictReader(csv_file2)

    set_videos2 = set()
    for row in reader:
        for (column_name, value) in row.items():
            if ('Input.video_url' in column_name and value != "nothing"):
                set_videos2.add(value)
    csv_file2.close()

    print(len(set_videos1), len(set_videos2))
    print(set_videos2 - set_videos1)


def compute_statistics(csv_file_name, spammers, no_spammers, double_spammers):
    csv_file1 = open(csv_file_name, 'r')
    reader = csv.DictReader(csv_file1)
    results = []
    hit_numbers = set()
    for row in reader:
        for (column_name, value) in row.items():
            if ('Ground_All_Yes_actions' == column_name):
                if (value != ''):
                    value_ground = 1
                else:
                    value_ground = 0
            if ('Majority_Yes_actions' == column_name):
                if (value != ''):
                    value_majority = 1
                else:
                    value_majority = 0
            if ('All_Yes_actions' == column_name):
                if (value != ''):
                    value_all = 1
                else:
                    value_all = 0

            if ('HIT_nb' == column_name):
                hit_nb = int(value)
        if (no_spammers == 0 or hit_nb not in spammers):  # or value_majority == 1) and hit_nb not in double_spammers):
            hit_numbers.add(hit_nb)
            results.append((value_ground, value_majority, value_all))

    nb_TP = 0
    nb_FP = 0
    nb_FN = 0
    nb_TN = 0
    # compute for majority of YES
    for (ground_truth, majority, all_yes) in results:
        if (ground_truth == majority == 1):
            nb_TP += 1
        if (ground_truth == majority == 0):
            nb_TN += 1
        if (ground_truth == 1 and majority == 0):
            nb_FN += 1
        if (ground_truth == 0 and majority == 1):
            nb_FP += 1

    accuracy = (nb_TP + nb_TN) / float(nb_FN + nb_FP + nb_TN + nb_TP)
    precision = nb_TP / float(nb_TP + nb_FP)
    recall = nb_TP / float(nb_FN + nb_TP)
    F1_measure = 2 * precision * recall / (precision + recall)
    print("FOR MAJORITY YES (2 out of 3)")
    print("accuracy is : " + str(accuracy))
    print("precision is : " + str(precision) + " = how many selected items are relevant")
    print("recall is : " + str(recall) + " = how many relevant items are selected")
    print("F1 score is : " + str(F1_measure))
    print("total nb of actions: " + str(nb_FN + nb_FP + nb_TN + nb_TP))
    print("total nb of HITS: " + str(len(hit_numbers)))

    # compute for all of YES
    for (ground_truth, majority, all_yes) in results:
        if (ground_truth == all_yes == 1):
            nb_TP += 1
        if (ground_truth == all_yes == 0):
            nb_TN += 1
        if (ground_truth == 1 and all_yes == 0):
            nb_FN += 1
        if (ground_truth == 0 and all_yes == 1):
            nb_FP += 1

    accuracy = (nb_TP + nb_TN) / float(nb_FN + nb_FP + nb_TN + nb_TP)
    precision = nb_TP / float(nb_TP + nb_FP)
    recall = nb_TP / float(nb_FN + nb_TP)
    F1_measure = 2 * precision * recall / (precision + recall)
    print("FOR ALL YES (3 out of 3)")
    print("accuracy is : " + str(accuracy))
    print("precision is : " + str(precision) + " = how many selected items are relevant")
    print("recall is : " + str(recall) + " = how many relevant items are selected")
    print("F1 score is : " + str(F1_measure))


def compute_agreement_ok(csv_file_name, ok_1_equal_2, spammer_hits, no_spammers, nb_turks_agree):
    csv_file = open(csv_file_name, 'r')
    reader = csv.DictReader(csv_file)

    list_results = []
    for row in reader:
        for (column_name, value) in row.items():

            if "Worker_1" == column_name:
                value1 = int(value)
                if (value1 == 2 and ok_1_equal_2 == 1):
                    value1 = 1
            if "Worker_2" == column_name:
                value2 = int(value)
                if (value2 == 2 and ok_1_equal_2 == 1):
                    value2 = 1
            if (nb_turks_agree == 3):
                if "Worker_3" == column_name:
                    value3 = int(value)
                    if (value3 == 2 and ok_1_equal_2 == 1):
                        value3 = 1
            if "HIT_nb" == column_name:
                hit_nb = int(value)

        if (nb_turks_agree == 3):
            if (value1 != -1 and value2 != -1 and value3 != -1):
                if (no_spammers == 0 or hit_nb not in spammer_hits):
                    list_results.append((hit_nb, value1, value2, value3))
        else:
            if (value1 != -1 and value2 != -1):
                if (no_spammers == 0 or hit_nb not in spammer_hits):
                    list_results.append((hit_nb, value1, value2))

    val_rater1 = []
    val_rater2 = []
    if (nb_turks_agree == 3):
        val_rater3 = []
        per_hit_val_rater3 = []
    hit = 0
    per_hit_val_rater1 = []
    per_hit_val_rater2 = []

    interpreations_per_hit = []

    if (nb_turks_agree == 3):
        for (hit_nb, value1, value2, value3) in list_results:
            val_rater1.append(value1)
            val_rater2.append(value2)
            val_rater3.append(value3)
            if (hit_nb == hit):
                per_hit_val_rater1.append(value1)
                per_hit_val_rater2.append(value2)
                per_hit_val_rater3.append(value3)
            else:
                print("------------------")
                print("Agreement for HIT " + str(hit) + ":")
                interpretation = compute_agreement(per_hit_val_rater1, per_hit_val_rater2, per_hit_val_rater3)
                interpreations_per_hit.append(interpretation)
                per_hit_val_rater1 = []
                per_hit_val_rater2 = []
                per_hit_val_rater3 = []
                per_hit_val_rater1.append(value1)
                per_hit_val_rater2.append(value2)
                per_hit_val_rater3.append(value3)

                hit = hit_nb
        # compute for last HIT
        print("------------------")
        print("Agreement for HIT " + str(hit) + ":")
        interpretation = compute_agreement(per_hit_val_rater1, per_hit_val_rater2, per_hit_val_rater3)
        interpreations_per_hit.append(interpretation)
        print("------------------")
        print("For 3 Workers Overall Agreement: ")

        overall_interpretation = compute_agreement(val_rater1, val_rater2, val_rater3)

    # if nb turkers = 2
    else:
        for (hit_nb, value1, value2) in list_results:
            val_rater1.append(value1)
            val_rater2.append(value2)
            if (hit_nb == hit):
                per_hit_val_rater1.append(value1)
                per_hit_val_rater2.append(value2)
            else:
                print("------------------")
                print("Agreement for HIT " + str(hit) + ":")
                interpretation = compute_agreement(per_hit_val_rater1, per_hit_val_rater2)
                interpreations_per_hit.append(interpretation)
                per_hit_val_rater1 = []
                per_hit_val_rater2 = []
                per_hit_val_rater1.append(value1)
                per_hit_val_rater2.append(value2)

                hit = hit_nb
        # compute for last HIT
        print("------------------")
        print("Agreement for HIT " + str(hit) + ":")
        interpretation = compute_agreement(per_hit_val_rater1, per_hit_val_rater2)
        interpreations_per_hit.append(interpretation)
        print("------------------")
        print("For 2 workers Overall Agreement:")
        overall_interpretation = compute_agreement(val_rater1, val_rater2)

    # for i in range(0,len(interpreations_per_hit)):
    #         print i, ',', interpreations_per_hit[i]

    counter = collections.Counter(interpreations_per_hit)
    print(counter)


def filter_results_for_spam(worker_file_name, filtered_file_name):
    csv_file = open(worker_file_name, 'r')
    reader = csv.DictReader(csv_file)

    with open(filtered_file_name, 'w+') as csvfile2:
        fieldnames = ['Actions', 'Worker_1', 'Worker_2', 'Worker_3', 'All_Yes_actions', 'Majority_Yes_actions']
        writer = csv.DictWriter(csvfile2, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            if ('Video_nb', '4') in row.items():
                for (column_name, value) in row.items():
                    if column_name == 'Actions':
                        action = value
                    elif column_name == 'Worker_1':
                        result_1 = value
                    elif column_name == 'Worker_2':
                        result_2 = value
                    elif column_name == 'Worker_3':
                        result_3 = value
                    elif column_name == 'All_Yes_actions':
                        action_all = value
                    elif column_name == 'Majority_Yes_actions':
                        action_maj = value

                writer.writerow({'Actions': action, 'Worker_1': result_1, 'Worker_2': result_2, 'Worker_3': result_3,
                                 'All_Yes_actions': action_all, 'Majority_Yes_actions': action_maj})


def get_channels_with_nb_visible_actions(output_file_name, nb_turks_per_hit, path_miniclips):
    csv_file = open(output_file_name, 'r')
    reader = csv.DictReader(csv_file)

    dict_channels_actions = dict()
    dict_miniclip_actions = dict()

    # divide to the total nb of actions because the number of actions per channel varies ..
    for row in reader:
        for (column_name, value) in row.items():
            if "Video_name" == column_name:
                channel, _, _ = value.split("_")
                dict_channels_actions[channel] = []
                dict_miniclip_actions[value] = []
    csv_file.close()

    csv_file2 = open(output_file_name, 'r')
    reader2 = csv.DictReader(csv_file2)

    for row in reader2:
        for (column_name, value) in row.items():
            if "Video_name" == column_name:
                channel, _, _ = value.split("_")
                miniclip = value

        for (column_name, value) in row.items():
            if (nb_turks_per_hit == 1):
                if "Worker_1" == column_name:
                    dict_channels_actions[channel].append(value)
                    dict_miniclip_actions[miniclip].append(value)
            else:
                if "Majority_Yes_actions" == column_name:
                    if value != '':
                        dict_channels_actions[channel].append('0')
                        dict_miniclip_actions[miniclip].append('0')
                    else:
                        dict_channels_actions[channel].append('1')
                        dict_miniclip_actions[miniclip].append('1')

    list_miniclip_nb_visible_actions = []
    for miniclip in dict_miniclip_actions.keys():
        nb_total_actions = len(dict_miniclip_actions[miniclip])
        nb_visible_actions = 0

        miniclip_length = getLength(path_miniclips + miniclip)
        for value in dict_miniclip_actions[miniclip]:
            if value == '0':
                nb_visible_actions += 1

        #     print channel, nb_total_actions, nb_visible_actions, 1.0 * nb_visible_actions/nb_total_actions
        list_miniclip_nb_visible_actions.append((miniclip, miniclip_length, nb_total_actions, nb_visible_actions))

    return list_miniclip_nb_visible_actions


def write_stats_in_csv(output_file_name, list_miniclip_nb_visible_actions):
    with open(output_file_name, 'w+') as csvfile:
        fieldnames = ['Miniclip', 'miniclip_length', 'nb_total_actions', 'nb_visible_actions',
                      'nb_visible_actions / nb_total_actions', 'nb_visible_actions / miniclip_length']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for (miniclip, miniclip_length, nb_total_actions, nb_visible_actions) in list_miniclip_nb_visible_actions:
            miniclip_length = float(miniclip_length.split(":")[2])
            writer.writerow(
                {'Miniclip': miniclip, 'miniclip_length': miniclip_length, 'nb_total_actions': nb_total_actions,
                 'nb_visible_actions': nb_visible_actions,
                 'nb_visible_actions / nb_total_actions': 1.0 * nb_visible_actions / nb_total_actions,
                 'nb_visible_actions / miniclip_length': 1.0 * nb_visible_actions / miniclip_length})


def get_stats_no_GT(csv_file_name):
    csv_file = open(csv_file_name, 'r')
    reader = csv.DictReader(csv_file)

    nb_visible_actions_maj = 0
    nb_visible_actions_all = 0

    for row in reader:
        for (column_name, value) in row.items():
            if column_name == 'Video_name' and 'p' in value.split("_")[0]:
                for (column_name, value) in row.items():
                    if column_name == 'Majority_Yes_actions':
                        if value != "":
                            nb_visible_actions_maj += 1
                    if column_name == 'All_Yes_actions':
                        if value != "":
                            nb_visible_actions_all += 1

    print("There are", str(nb_visible_actions_maj), "visible actions labeled by majority of workers (2 out of 3)")
    print("There are", str(nb_visible_actions_all), "visible actions labeled by all workers (1 out of 3)")


def compare_with_GT(dict_actions_hit, AMT_input_csv, GT_csv, output_results_csv, spammers_files, csv_file_name):
    csv_file = open(AMT_input_csv, 'r')
    reader = csv.DictReader(csv_file)

    with open(output_results_csv, 'w+') as csvfile3:
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
                csv_file2 = open(GT_csv, 'r')
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
    csv_file = open(output_results_csv, 'r')
    reader = csv.DictReader(csv_file)
    ok_1_equal_2 = 1
    rater_answers = []
    list_results = []
    hit_nbs = set()
    for row in reader:
        for (column_name, value) in row.items():
            if "Worker_1" == column_name:
                value1 = int(value)
                if (value1 == 2 and ok_1_equal_2 == 1):
                    value1 = 1
            if "Worker_2" == column_name:
                value2 = int(value)
                if (value2 == 2 and ok_1_equal_2 == 1):
                    value2 = 1
            if "Worker_3" == column_name:
                value3 = int(value)
                if (value3 == 2 and ok_1_equal_2 == 1):
                    value3 = 1
            if "Worker_GT" == column_name:
                value4 = int(value)
                if (value4 == 2 and ok_1_equal_2 == 1):
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

                    if accuracy_with_GT < 0.2:
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
    with open(csv_file_name, 'r') as csvinput:
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
    with open(csv_file_name, 'r') as csvinput:
        with open(spammers_files, 'w+') as csvoutput:
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

    # # remove keys with empty sets
    # for k in potential_spammers.keys():
    #     if potential_spammers[k] == set():
    #         del potential_spammers[k]

    return potential_spammers


def create_after_spam_filtered_results(output_file_name, potential_spammers, compromised_hits, after_spam_filter_csv):
    csv_file = open(output_file_name, 'r')
    reader = csv.DictReader(csv_file)

    with open(after_spam_filter_csv, 'w+') as csvfile2:
        fieldnames = ['HIT_nb', 'Video_name', 'Actions', 'Worker_1', 'Worker_2', 'Worker_3']
        writer = csv.DictWriter(csvfile2, fieldnames=fieldnames)
        writer.writeheader()
        HIT_nb = ""
        Actions = ""
        Video_name = ""
        Worker_1 = ""
        Worker_2 = ""
        Worker_3 = ""
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

            if len(potential_spammers[HIT_nb]) == 0:
                writer.writerow({'HIT_nb': HIT_nb, 'Video_name': Video_name, 'Actions': Actions, 'Worker_1': Worker_1,
                                 'Worker_2': Worker_2, 'Worker_3': Worker_3})
            elif HIT_nb in compromised_hits:
                continue
            else:
                spammer = int(list(potential_spammers[HIT_nb])[0])
                if spammer == 0:
                    Worker_1 = -1
                elif spammer == 1:
                    Worker_2 = -1
                else:
                    Worker_3 = -1
                writer.writerow({'HIT_nb': HIT_nb, 'Video_name': Video_name, 'Actions': Actions, 'Worker_1': Worker_1,
                                 'Worker_2': Worker_2, 'Worker_3': Worker_3})

    csv_file.close()
    csvfile2.close()


def get_visible_and_not_visible_actions(after_spam_filter_csv, visible_not_visible_actions_csv, no_GT_FILE):
    visible_actions = dict()
    not_visible_actions = dict()
    all_visible_actions = dict()

    csv_file = open(after_spam_filter_csv, 'r')
    reader = csv.DictReader(csv_file)
    video_action_pair = []

    with open(visible_not_visible_actions_csv, 'w+') as csvfile2:
        fieldnames = ['Video_name', 'Visible Actions', 'Not Visible Actions']
        writer = csv.DictWriter(csvfile2, fieldnames=fieldnames)
        writer.writeheader()

        Actions = ""
        Video_name = ""
        Worker_1 = ""
        Worker_2 = ""
        Worker_3 = ""

        for row in reader:
            for (column_name, value) in row.items():
                if column_name == 'Actions':
                    Actions = value
                if column_name == 'Video_name':
                    Video_name = value
                if column_name == 'HIT_nb':
                    HIT_nb = value
                if column_name == 'Worker_1':
                    if int(value) == 2:
                        value = 1
                    Worker_1 = int(value)
                if column_name == 'Worker_2':
                    if int(value) == 2:
                        value = 1
                    Worker_2 = int(value)
                if column_name == 'Worker_3':
                    if int(value) == 2:
                        value = 1
                    Worker_3 = int(value)

            if no_GT_FILE == True:
                if 'p0' not in Video_name and 'p1' not in Video_name:
                    continue

            if (Video_name, Actions) not in video_action_pair:
                video_action_pair.append((Video_name, Actions))
            else:
                # repeated actions (probably from GT data)
                continue
            # !!! I had to remove the GT data from video 3( it was duplicated with all the other videos from last batch)
            # if 'p' not in Video_name[0:-3]:
            #     print Video_name
            #     continue
            if Worker_1 == Worker_2 == 0 or Worker_1 == Worker_3 == 0 or Worker_2 == Worker_3 == 0:
                if Video_name not in visible_actions.keys():
                    visible_actions[Video_name] = []
                visible_actions[Video_name].append(Actions)
                if Worker_1 == Worker_2 == Worker_3 == 0:
                    if Video_name not in all_visible_actions.keys():
                        all_visible_actions[Video_name] = []
                    all_visible_actions[Video_name].append(Actions)

                writer.writerow({'Video_name': Video_name, 'Visible Actions': Actions, 'Not Visible Actions': ''})

            else:
                if Video_name not in not_visible_actions.keys():
                    not_visible_actions[Video_name] = []
                not_visible_actions[Video_name].append(Actions)
                writer.writerow({'Video_name': Video_name, 'Visible Actions': '', 'Not Visible Actions': Actions})
    csv_file.close()
    csvfile2.close()

    return visible_actions, not_visible_actions


def compare_spammers_files(bad_spammers, good_spammers):
    csv_file = open(bad_spammers, 'r')
    reader = csv.DictReader(csv_file)

    dict_bad_spammers = OrderedDict()
    for row in reader:
        for (column_name, value) in row.items():
            if column_name == "WorkerId":
                WorkerId = value
            if column_name == "AssignmentId":
                AssignmentId = value
            if column_name == "SubmitTime":
                DateSubmitted = value  # Made an error (offseted columns)
            if column_name == "AssignmentStatus":
                Reject = value
        if Reject == 'Rejected':
            dict_bad_spammers[AssignmentId] = [WorkerId, DateSubmitted]

    csv_file = open(good_spammers, 'r')
    reader = csv.DictReader(csv_file)

    dict_good_spammers = OrderedDict()
    for row in reader:
        for (column_name, value) in row.items():
            if column_name == "WorkerId":
                WorkerId = value
            if column_name == "AssignmentId":
                AssignmentId = value
            if column_name == "SubmitTime":
                DateSubmitted = value  # Made an error (offseted columns)
            if column_name == "Reject":
                Reject = value
        if 'spammer' in Reject:
            dict_good_spammers[AssignmentId] = [WorkerId, DateSubmitted]

    list_spammers_wrongly_labeled = []
    list_spammers_ok_labeled = []
    for key_bad_spammer in dict_bad_spammers.keys():
        if key_bad_spammer not in dict_good_spammers.keys():
            list_spammers_wrongly_labeled.append(key_bad_spammer)
        else:
            list_spammers_ok_labeled.append(key_bad_spammer)

    dict_workers_labeled_wrong = OrderedDict()
    for key_bad_spammer in list_spammers_wrongly_labeled:
        WorkerID = dict_bad_spammers[key_bad_spammer][0]
        AssignmentId = key_bad_spammer
        DateSubmitted = dict_bad_spammers[key_bad_spammer][1]
        if WorkerID not in dict_workers_labeled_wrong.keys():
            dict_workers_labeled_wrong[WorkerID] = []
        dict_workers_labeled_wrong[WorkerID].append([AssignmentId, DateSubmitted])
    for key_bad_spammer in dict_workers_labeled_wrong:
        # print "WorkerId:", key_bad_spammer
        for l in dict_workers_labeled_wrong[key_bad_spammer]:
            print("WorkerId:", key_bad_spammer, "AssignmentId:", l[0], "DateSubmitted:", l[1])
    # print "There were {0} workers labeled wrong and {1} total assignments".format(len(dict_workers_labeled_wrong.keys()), len(list_spammers_wrongly_labeled)) 
    # print "There were {0} workers labeled okay and {1} total assignments".format(len(dict_workers_labeled_ok.keys()), len(list_spammers_ok_labeled)) 


if __name__ == '__main__':

    index_batch = 4

    ROOT_PATH = "data/AMT/"
    # bad_spammers_PATH = ROOT_PATH + "Batch" + str(index_batch) + "/bad_spammers_file" + str(index_batch) + ".csv" 
    # good_spammers_PATH = ROOT_PATH + "Batch" + str(index_batch) + "/spammers_file" + str(index_batch) + ".csv" 
    # bad_spammers_PATH = ROOT_PATH + "Batch_3377775_batch_results.csv" 
    # compare_spammers_files(bad_spammers_PATH, good_spammers_PATH)

    PATH_input_batch = ROOT_PATH + "Input/" + "Batch" + str(index_batch)
    PATH_output_batch = ROOT_PATH + "Output/" + "Batch" + str(index_batch)
    if not os.path.exists(PATH_input_batch):
        os.makedirs(PATH_input_batch)

    if not os.path.exists(PATH_output_batch):
        os.makedirs(PATH_output_batch)

    csv_file_name = PATH_input_batch + "/Batch_video" + str(index_batch) + "_batch_results.csv"  # sys.argv[1]
    output_file_name = PATH_output_batch + "/results_batch_video" + str(index_batch) + ".csv"  # sys.argv[2]
    dict_output, dict_actions_output = read_results_from_AMT(csv_file_name)

    GLOBAL_NB_NONGT_VIDEOS, GLOBAL_NB_NONGT_MINICLIPS, GLOBAL_NB_GT_VIDEOS, GLOBAL_NB_GT_MINICLIPS = get_action_names_per_video(
        csv_file_name)
    write_output_file(output_file_name, dict_output)

    # ------------------ Spam check ----------------
    GT_csv = ROOT_PATH + "Input/" + "BOTH_Laura_Oana - Copy.csv"  # sys.argv[2]
    output_results_csv = PATH_output_batch + "/results_BOTH_GT_video" + str(index_batch) + ".csv"  # sys.argv[2]
    spammers_files = PATH_output_batch + "/spammers_file" + str(index_batch) + ".csv"

    potential_spammers = compare_with_GT(dict_output, output_file_name, GT_csv, output_results_csv, spammers_files,
                                         csv_file_name)

    # TEST_results_input_AMT(potential_spammers.keys(), dict_actions_output)

    compromised_hits = []  # hits with more than 1 spammer
    for hit_nb in potential_spammers.keys():
        if len(potential_spammers[hit_nb]) > 1:
            compromised_hits.append(hit_nb)
    spammer_hits = potential_spammers.keys()
    print("Compromised hits: " + str(compromised_hits))

    after_spam_filter_csv = PATH_output_batch + "/results_after_spam_filter_video" + str(index_batch) + ".csv"
    create_after_spam_filtered_results(output_file_name, potential_spammers, compromised_hits, after_spam_filter_csv)

    visible_not_visible_actions_noGT_csv = PATH_output_batch + "/no_GT_visible_not_visible_actions_video_after_spam" + str(
        index_batch) + ".csv"
    visible_not_visible_actions_csv = PATH_output_batch + "/visible_not_visible_actions_video_after_spam" + str(
        index_batch) + ".csv"
    visible_actions, not_visible_actions = get_visible_and_not_visible_actions(after_spam_filter_csv,
                                                                               visible_not_visible_actions_noGT_csv,
                                                                               no_GT_FILE=True)
    visible_actions, not_visible_actions = get_visible_and_not_visible_actions(after_spam_filter_csv,
                                                                               visible_not_visible_actions_csv,
                                                                               no_GT_FILE=False)

    # make a whole file from all batches
    PATH_all_visible_not_visible_after_spam = ROOT_PATH + "Output/" + "visible_not_visible_actions_video_after_spam_total.csv"
    if not os.path.exists(PATH_all_visible_not_visible_after_spam):
        file_object = open(PATH_all_visible_not_visible_after_spam, 'w+')
        file_object.close()
    results = pd.DataFrame([])
    a = pd.read_csv(
        ROOT_PATH + "Output/" + "Batch" + str(3) + "/no_GT_visible_not_visible_actions_video_after_spam" + str(
            3) + ".csv")
    b = pd.read_csv(
        ROOT_PATH + "Output/" + "Batch" + str(4) + "/visible_not_visible_actions_video_after_spam" + str(4) + ".csv")
    a = a.append(b)
    a.to_csv(PATH_all_visible_not_visible_after_spam, index=False)
