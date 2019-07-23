from __future__ import print_function

import csv
import pandas as pd
import numpy as np


def get_miniclips_video_stats(csv_file_name):
    set_videos = set()
    set_miniclips = set()
    set_videos_GT = set()
    set_miniclips_GT = set()

    df_data = pd.read_csv(csv_file_name)
    video_name = df_data['Video_name']
    list_video_names = video_name.tolist()
    for value in list_video_names:
        if 'p0' in value or 'p1' in value:
            channel_id, video_id, miniclip_id = value.split("_")
            set_videos.add(channel_id + "_" + video_id)
            set_miniclips.add(value)
        else:
            video_id, miniclip_id = value.split("_")
            set_videos_GT.add(video_id)
            set_miniclips_GT.add(value)

    nb_non_gt_videos = len(set_videos)
    nb_non_gt_miniclips = len(set_miniclips)
    nb_gt_videos = len(set_videos_GT)
    nb_gt_miniclips = len(set_miniclips_GT)

    return nb_non_gt_videos, nb_non_gt_miniclips, nb_gt_videos, nb_gt_miniclips


def get_actions_stats(csv_file_name):
    df_data = pd.read_csv(csv_file_name)
    unique_data = df_data.drop_duplicates(subset=['Video_name', 'Actions'])

    unique_data['Majority_Yes_actions'].replace('  ', np.nan, inplace=True)
    df_maj = unique_data.dropna(subset=['Majority_Yes_actions'])
    majority_yes = df_maj['Majority_Yes_actions']
    nb_majority_visible_actions = len(majority_yes.tolist())

    unique_data['All_Yes_actions'].replace('  ', np.nan, inplace=True)
    df_all = unique_data.dropna(subset=['All_Yes_actions'])

    all_yes = df_all['All_Yes_actions']
    nb_all_visible_actions = len(all_yes.tolist())

    unique_data['Actions'].replace('  ', np.nan, inplace=True)
    df_total = unique_data.dropna(subset=['Actions'])
    total_actions = df_total['Actions']
    nb_total_actions = len(total_actions.tolist())

    return nb_total_actions, nb_majority_visible_actions, nb_all_visible_actions


def print_stats_actions_miniclips(nb_total_actions, nb_majority_visible_actions, nb_all_visible_actions,
                                  nb_non_gt_videos,
                                  nb_non_gt_miniclips, nb_gt_videos, nb_gt_miniclips):
    print(
        "There are " + str(nb_non_gt_videos) + " NON-GT videos and " + str(nb_non_gt_miniclips) + " NON-GT miniclips.")
    print("There are " + str(nb_gt_videos) + " GT videos and " + str(nb_gt_miniclips) + " GT miniclips.")
    print("In total, there are " + str(nb_non_gt_videos + nb_gt_videos) + " videos and " + str(
        nb_non_gt_miniclips + nb_gt_miniclips) + " miniclips.")

    print(
        "There are " + str(nb_total_actions) + " total actions in " + str(nb_non_gt_videos + nb_gt_videos) + " videos")
    print("There are " + str(
        nb_majority_visible_actions) + " majority (labeled by at least 2/3 workers ) visible actions in " + str(
        nb_non_gt_videos + nb_gt_videos) + " videos")
    print("There are " + str(nb_all_visible_actions) + " all visible (labeled by 3/3 workers) actions in " + str(
        nb_non_gt_videos + nb_gt_videos) + " videos")
    print("majority_visible / total actions = " + str(1.0 * nb_majority_visible_actions / nb_total_actions))
    print("all_visible / total actions = " + str(1.0 * nb_all_visible_actions / nb_total_actions))
    print("majority_visible / # VIDEOS = " + str(1.0 * nb_majority_visible_actions / (nb_non_gt_videos + nb_gt_videos)))
    print("all_visible / # VIDEOS = " + str(1.0 * nb_all_visible_actions / (nb_non_gt_videos + nb_gt_videos)))
    print("majority_visible / # MINICLIPS = " + str(
        1.0 * nb_majority_visible_actions / (nb_non_gt_miniclips + nb_gt_miniclips)))
    print("all_visible / # MINICLIPS = " + str(1.0 * nb_all_visible_actions / (nb_non_gt_miniclips + nb_gt_miniclips)))
    print("##-------------------------------------------------##")


def compute_statistics(csv_file_name, spammers, no_spammers, double_spammers):
    csv_file1 = open(csv_file_name, 'r')
    reader = csv.DictReader(csv_file1)
    results = []
    hit_numbers = set()
    for row in reader:
        for (column_name, value) in row.items():
            if 'Ground_All_Yes_actions' == column_name:
                if value != '':
                    value_ground = 1
                else:
                    value_ground = 0
            if 'Majority_Yes_actions' == column_name:
                if value != '':
                    value_majority = 1
                else:
                    value_majority = 0
            if 'All_Yes_actions' == column_name:
                if value != '':
                    value_all = 1
                else:
                    value_all = 0

            if 'HIT_nb' == column_name:
                hit_nb = int(value)
        if no_spammers == 0 or hit_nb not in spammers:  # or value_majority == 1) and hit_nb not in double_spammers):
            hit_numbers.add(hit_nb)
            results.append((value_ground, value_majority, value_all))

    nb_TP = 0
    nb_FP = 0
    nb_FN = 0
    nb_TN = 0
    # compute for majority of YES
    for (ground_truth, majority, all_yes) in results:
        if ground_truth == majority == 1:
            nb_TP += 1
        if ground_truth == majority == 0:
            nb_TN += 1
        if ground_truth == 1 and majority == 0:
            nb_FN += 1
        if ground_truth == 0 and majority == 1:
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
    print("(total nb of HITS: " + str(len(hit_numbers)))

    # compute for all of YES
    for (ground_truth, majority, all_yes) in results:
        if ground_truth == all_yes == 1:
            nb_TP += 1
        if ground_truth == all_yes == 0:
            nb_TN += 1
        if ground_truth == 1 and all_yes == 0:
            nb_FN += 1
        if ground_truth == 0 and all_yes == 1:
            nb_FP += 1

    accuracy = (nb_TP + nb_TN) / float(nb_FN + nb_FP + nb_TN + nb_TP)
    precision = nb_TP / float(nb_TP + nb_FP)
    recall = nb_TP / float(nb_FN + nb_TP)
    F1_measure = 2 * precision * recall / (precision + recall)
    print("For ALL YES (3 out of 3)")
    print("accuracy is : " + str(accuracy))
    print("precision is : " + str(precision) + " = how many selected items are relevant")
    print("recall is : " + str(recall) + " = how many relevant items are selected")
    print("F1 score is : " + str(F1_measure))
