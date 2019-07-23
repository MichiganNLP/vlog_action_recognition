from __future__ import print_function

import string
from collections import OrderedDict
import csv

from Test_data import print_stats_actions_miniclips, get_miniclips_video_stats, get_actions_stats
from amt.settings import PATH_input_file_name, PATH_output_file_name



def read_results_from_amt(csv_file_name):
    csv_file = open(csv_file_name, 'r')
    reader = csv.DictReader(csv_file)

    dict_actions_output = OrderedDict()

    hit_id = ""
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
                    if list_triples_values:
                        dict_actions_output[hit_id].append(list_triples_values)

                    dict_video_names = OrderedDict()
                    if hit_new_id not in dict_actions_output.keys():
                        dict_actions_output[hit_new_id] = []

                    for video_index in range(0, len(list_video_names)):
                        sublist_actions = list_actions[video_index]
                        video_name = list_video_names[video_index]

                        if video_name not in dict_video_names.keys():
                            dict_video_names[video_name] = []

                        #sublist_actions = clean_list_actions(sublist_actions)
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


def write_output_file(output_file_name, dict_output):
    fieldnames = []
    hit_id = dict_output.keys()[0]
    video_name = dict_output[hit_id].keys()[0]
    result = dict_output[hit_id][video_name]
    nb_turks_per_hit = len(result[0][1])

    with open(output_file_name, 'w+') as csvfile:
        if nb_turks_per_hit == 3:
            fieldnames = ['HIT_nb', 'Video_name', 'Actions', 'Worker_1', 'Worker_2', 'Worker_3', 'All_Yes_actions',
                          'Majority_Yes_actions', 'Majority_No_actions']
        elif nb_turks_per_hit == 1:
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
                        if result_1 == 0:
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
                        elif result_1 == result_2 == 0 or result_1 == result_3 == 0 or result_3 == result_2 == 0:
                            if video_name != dict_video_names.keys()[-1]:
                                nb_majority_visible_actions += 1
                            writer.writerow(
                                {'HIT_nb': hit_id, 'Video_name': video_name, 'Actions': action, 'Worker_1': result_1,
                                 'Worker_2': result_2, 'Worker_3': result_3, 'Majority_Yes_actions': action})
                        else:
                            writer.writerow(
                                {'HIT_nb': hit_id, 'Video_name': video_name, 'Actions': action, 'Worker_1': result_1,
                                 'Worker_2': result_2, 'Worker_3': result_3, 'Majority_No_actions': action})

    csvfile.close()


def process_results_amt(print_script=False):
    dict_output, dict_actions_output = read_results_from_amt(PATH_input_file_name)
    write_output_file(PATH_output_file_name, dict_output)
    if print_script:
        nb_non_gt_videos, nb_non_gt_miniclips, nb_gt_videos, nb_gt_miniclips = get_miniclips_video_stats(
            PATH_output_file_name)
        nb_total_actions, nb_majority_visible_actions, nb_all_visible_actions = get_actions_stats(PATH_output_file_name)
        print("##--------------- Before Spam Check --------------------------##")
        print_stats_actions_miniclips(nb_total_actions, nb_majority_visible_actions, nb_all_visible_actions,
                                      nb_non_gt_videos, nb_non_gt_miniclips, nb_gt_videos, nb_gt_miniclips)
    return dict_output, dict_actions_output
