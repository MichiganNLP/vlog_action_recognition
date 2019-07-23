#!/usr/bin/env python2

import os
import csv
from collections import defaultdict
import ffmpy
from datetime import datetime
import glob
import copy
import string
import shutil


def write_html(in_file_name, out_file_name):
    input_file = open(in_file_name, 'r')
    output_file = open(out_file_name, 'w+')

    lines = input_file.readlines()
    input_file.close()

    index_line = 0
    new_input = []
    for line in lines:
        sth, action = line.split(":")
        output_file.write(
            "<input id=\"" + sth + "\" name=\"" + sth + "\" type=\"checkbox\" value=\"" + sth + "\" /><label for=\"" + sth + "\">" + " " + action + "</label><br />" + "\n")

    output_file.close()
    input_file.close()

    return


def get_actions_time_from_csv(csv_file_name):
    video_dict = dict()
    set_index_video = set()

    csv_file = open(csv_file_name, 'r')
    reader = csv.DictReader(csv_file)
    for row in reader:
        for (k, v) in row.items():
            if (k == 'key' and v != ""):
                set_index_video.add(v)
            # if(k == 'transcript_id' and v!= ""):
            #     set_index_video.add(v)
    csv_file.close()

    for key in set_index_video:

        csv_file = open(csv_file_name, 'r')
        reader = csv.DictReader(csv_file)

        video_dict[key] = []
        columns = defaultdict(list)

        for row in reader:
            for (k, v) in row.items():
                if (k == 'key' and v == key):
                    for (k, v) in row.items():
                        if (v != ""):
                            columns[k].append(v)
                #  if(k == 'transcript_id' and v == key):
                #     for (k,v) in row.items():
                #         if( v!=""):
                #             columns[k].append(v)

        actions_column = columns['actions']
        start_column = columns['start_time']
        end_column = columns['end_time']

        # filter actions of weird characters as & (replace with and)
        filtered_actions = []
        for x in actions_column:
            if "&" in x:
                x = string.replace(x, '&', 'and')
            filtered_actions.append(x)

        # actions_column = [x if ("&" not in x) else 'and' for x in actions_column]

        video_dict[key].append(filtered_actions)
        video_dict[key].append(start_column)
        video_dict[key].append(end_column)

    # for key in list_index_video:

    #     csv_file = open(csv_file_name,'r')
    #     reader = csv.DictReader(csv_file)

    #     video_dict[key] = []
    #     columns = defaultdict(list)

    #     for row in reader: 
    #         for (k,v) in row.items():
    #             if(k == 'key' and v == key):
    #                 for (k,v) in row.items():
    #                     if( v!=""):
    #                         columns[k].append(v)

    #     actions_column = columns['actions']
    #     start_column = columns['start_time']
    #     end_column = columns['end_time']

    #     video_dict[key].append(actions_column)
    #     video_dict[key].append(start_column)
    #     video_dict[key].append(end_column)

    # return actions_column, start_column, end_column
    return video_dict


def resize_time_per_action(video_dict, time_window):
    # - 3s for start_column
    # + 3s for end_column
    # time_window = 3
    resized_video_dict = dict()
    for key in video_dict.keys():
        resized_video_dict[key] = []
        start_column = video_dict[key][1]
        end_column = video_dict[key][2]

        new_start_column = []
        for time in start_column:
            hour, minute, sec = time.split(':')
            int_sec = int(sec)
            int_min = int(minute)
            if (int_sec < time_window and int_min == 0):
                int_sec = 0
            elif (int_sec < time_window and int_min != 0):
                int_min = int_min - 1
                int_sec = 60 - (time_window - int_sec)
            else:
                int_sec = int_sec - time_window

            if int_sec < 10:
                sec = '0' + str(int_sec)
            else:
                sec = str(int_sec)
            minute = str(int_min)

            new_start_column.append(minute + ":" + sec)

        new_end_column = []
        for time in end_column:
            if (time != 'END'):
                hour, minute, sec = time.split(':')
                int_sec = int(sec)
                int_min = int(minute)
                if (int_sec + time_window > 59):
                    int_min = int_min + 1
                    int_sec = time_window + int_sec - 60
                else:
                    int_sec = int_sec + time_window

                if int_sec < 10:
                    sec = '0' + str(int_sec)
                else:
                    sec = str(int_sec)
                minute = str(int_min)
                new_end_column.append(minute + ":" + sec)
            else:
                new_start_column = new_start_column[:-1]

        list_tuples_time = []
        for i in range(0, len(new_start_column)):
            resized_video_dict[key].append((new_start_column[i], new_end_column[i]))
    #       list_tuples_time.append((new_start_column[i], new_end_column[i]))
    #  resized_video_dict[key].append(list_tuples_time)
    return resized_video_dict


'''
return -1, if time_1 < time_2
        1, if time_1 > time_2
        0, if equal
'''


def compare_time(time_1, time_2):
    minute_1_s, sec_1_s = time_1.split(":")
    minute_2_s, sec_2_s = time_2.split(":")

    if int(minute_1_s) == int(minute_2_s) and int(sec_1_s) == int(sec_2_s):
        return 0
    if int(minute_1_s) > int(minute_2_s) or (int(minute_1_s) == int(minute_2_s) and int(sec_1_s) >= int(sec_2_s)):
        return 1
    else:
        return -1


def get_time_difference(time_1, time_2):
    if compare_time(time_1, time_2) == -1:
        return 100
    FMT = '%M:%S'
    tdelta = datetime.strptime(time_1, FMT) - datetime.strptime(time_2, FMT)
    tdelta_h, tdelta_min, tdelta_seconds = str(tdelta).split(":")
    tdelta_total = int(tdelta_seconds) + int(tdelta_min) * 60 + int(tdelta_h) * 360
    return tdelta_total


def process_time_per_action(resized_video_dict, video_dict):
    clip_actions_time = defaultdict(list)

    for channel_video_index in resized_video_dict.keys():

        list_tuples_time = resized_video_dict[channel_video_index]
        list_actions = video_dict[channel_video_index][0]
        miniclip_index = 0
        i = 0

        while (i < len(list_tuples_time) - 1):
            mini_clip_list_action = set()
            miniclip_index += 1
            start_time_1 = list_tuples_time[i][0]
            end_time_1 = list_tuples_time[i][1]
            minute_1_s, sec_1_s = start_time_1.split(":")
            minute_1_e, sec_1_e = end_time_1.split(":")

            mini_clip_list_action.add(list_actions[i])
            final_end_time = end_time_1
            final_start_time = start_time_1

            for j in range(i + 1, len(list_tuples_time)):

                start_time_2 = list_tuples_time[j][0]
                end_time_2 = list_tuples_time[j][1]
                minute_2_s, sec_2_s = start_time_2.split(":")
                minute_2_e, sec_2_e = end_time_2.split(":")

                # same end and start time
                if compare_time(final_start_time, start_time_2) == 0 and compare_time(final_end_time, end_time_2) == 0:
                    mini_clip_list_action.add(list_actions[j])
                    i = j + 1

                # action 2 time included in action 1 time
                elif compare_time(final_start_time, start_time_2) == -1 and compare_time(final_end_time,
                                                                                         end_time_2) == 1:
                    mini_clip_list_action.add(list_actions[j])
                    i = j + 1

                # action 1 time included in action 2 time
                elif compare_time(final_start_time, start_time_2) == 1 and compare_time(final_end_time,
                                                                                        end_time_2) == -1:
                    final_start_time = start_time_2
                    final_end_time = end_time_2
                    mini_clip_list_action.add(list_actions[j])
                    i = j + 1

                # intersection 1 between A1 and A2 while total time < 60s
                elif compare_time(end_time_2, final_start_time) == 1 and get_time_difference(end_time_2,
                                                                                             final_start_time) <= 60 and (
                        compare_time(final_start_time, start_time_2) == -1 or compare_time(final_start_time,
                                                                                           start_time_2) == 0) and (
                        compare_time(final_end_time, end_time_2) == -1 or compare_time(final_end_time,
                                                                                       end_time_2) == 0):
                    final_end_time = end_time_2
                    mini_clip_list_action.add(list_actions[j])
                    i = j + 1

                # intersection 2 between A1 and A2 while total time < 60s
                elif compare_time(final_end_time, start_time_2) == 1 and get_time_difference(final_end_time,
                                                                                             start_time_2) <= 60 and (
                        compare_time(final_start_time, start_time_2) == 1 or compare_time(final_start_time,
                                                                                          start_time_2) == 0) and compare_time(
                    final_end_time, end_time_2) == 1:
                    final_start_time = start_time_2
                    mini_clip_list_action.add(list_actions[j])
                    i = j + 1
                else:
                    i = j
                    break

                # to save it as json
            # clip_actions_time[str(miniclip_index) + channel_video_index].append([final_start_time, final_end_time])
            # clip_actions_time[str(miniclip_index) + channel_video_index].append(list(mini_clip_list_action))

            clip_actions_time[(miniclip_index, channel_video_index)].append([final_start_time, final_end_time])
            clip_actions_time[(miniclip_index, channel_video_index)].append(list(mini_clip_list_action))

    return clip_actions_time


def make_mini_clips(clip_actions_time, p_input_video, path_output_video):
    for (mini_index, channel_video_index) in clip_actions_time.keys():
        channel_index, video_index = channel_video_index.split(",")
        channel_index = channel_index[1:]
        video_index = video_index[1:-1]

        # list_videos = glob.glob(p_input_video+"*mp4")
        path_input_video = p_input_video + channel_index + "video_" + video_index + '.mp4'
        # path_input_video_mkv = p_input_video + channel_index + "video_" + video_index  +'.mkv'
        extension = '.mp4'
        # if path_input_video_mp4 in list_videos:
        #     path_input_video = path_input_video_mp4
        #     extension = '.mp4'
        # elif path_input_video_mkv in list_videos:
        #     path_input_video = path_input_video_mkv
        #     extension = '.mkv'
        # else:
        #     print "ERROR"
        #     print path_input_video_mkv
        #     path_input_video = ""
        #     extension = ""

        time_start = clip_actions_time[(mini_index, channel_video_index)][0][0]
        time_end = clip_actions_time[(mini_index, channel_video_index)][0][1]

        # if (mini_index == 0 and video_index == "2"):
        #     print time_start, time_end
        # if (mini_index == 20 and video_index == "2"):
        #     print time_start, time_end
        #  ff = ffmpy.FFmpeg(inputs={path_input_video : None}, outputs={path_output_video + video_index + 'mini_' + str(mini_index) + '.mp4': '-vcodec -copy  -acodec copy -ss ' + str(time_start) + ' -to ' + str(time_end) + ' -c copy -copyts '})

        # ff = ffmpy.FFmpeg(inputs={path_input_video : None}, outputs={path_output_video +  channel_index + "_" + video_index  + 'mini_' + str(mini_index) + '.mp4': ' -ss ' + str(time_start) + ' -to ' + str(time_end) + ' -c copy -copyts '})

        # ff.run()

        # command = 'ffmpeg -i ' + path_input_video + ' -ss ' + time_start + ' -to ' + time_end + ' -c:a aac -strict -2 ' + path_output_video + channel_index + "_" + video_index  + 'mini_' + str(mini_index) + '.mp4'
        # command = 'ffmpeg -i ' + path_input_video + ' -ss ' + time_start + ' -to ' + time_end + ' -c:a aac -strict -2 -vcodec libx264 -r 15 -preset ultrafast -s 800x600 ' + path_output_video + channel_index + "_" + video_index  + 'mini_' + str(mini_index) + '.mp4'
        command = 'ffmpeg -i ' + path_input_video + ' -ss ' + time_start + ' -to ' + time_end + ' -c copy -an ' + path_output_video + channel_index + "_" + video_index + 'mini_' + str(
            mini_index) + extension
        os.system(command)

    return


# def write_in_csv_file(csv_file_name, clip_actions_time, max_nb_actions_per_video):
#     with open(csv_file_name, "wb+") as csv_file:
#         writer = csv.writer(csv_file, delimiter=',')
#         header = ["video_url","actions"]
#         writer.writerow(header)
#         for key in clip_actions_time.keys():
#             video_url = 'mini_' + str(key) + '.mp4'
#             actions = clip_actions_time[key][1]
#             cont = 1
#             nb_actions = 0
#             while cont == 1:
#                 string_actions = ""
#                 for a in actions:
#                     string_actions += a + ";"
#                     nb_actions += 1
#                     if(nb_actions % 7 == 0):
#                         break

#                 line = []
#                 line.append(video_url)
#                 line.append(string_actions[:-1])
#                 writer.writerow(line)
#                 if(nb_actions % 7 == 0):
#                     cont = 1
#                     actions = actions[nb_actions:]
#                 else:
#                     cont = 0
#     return

def filter_miniclips_max_60_minutes(clip_actions_time, PATH_miniclips):
    list_videos = glob.glob(PATH_miniclips + "*.mp4")
    list_good_keys = []

    for video_file in list_videos:
        video_name = video_file.split('/')[-1:][0]
        new_video_name, _ = video_name.split('.')
        index_channel, video_index_more, index_miniclip = new_video_name.split('_')
        index_video, _ = video_index_more.split('mini')
        # if index_channel not in ['6', '10', '11']:
        list_good_keys.append((int(index_miniclip), '(' + index_channel + ', ' + index_video + ')'))

    filtered_clip_actions_time = copy.deepcopy(clip_actions_time)

    # TO DO: SOLVE PROBLEM WITH LARGE MINICLIPS
    list_large_miniclips = []
    with open("AMT2/Batch3/large_miniclips.csv", "r") as csv_file2:
        reader = csv.DictReader(csv_file2)
        for row in reader:
            for (k, v) in row.items():
                index_channel, video_index_more, index_miniclip = v.split('_')
                index_video, _ = video_index_more.split('mini')
                list_large_miniclips.append((int(index_miniclip), '(' + index_channel + ', ' + index_video + ')'))
        csv_file2.close()

    for key in filtered_clip_actions_time.keys():
        if key not in list_good_keys or key in list_large_miniclips:
            filtered_clip_actions_time.pop(key, None)

    return filtered_clip_actions_time


def create_input_AMT_WHOLE(PATH_INPUT_AMT, filtered_clip_actions_time, PARAM_nb_video_id,
                           PARAM_MAX_NB_ACTIONS_PER_MINICLIP):
    # put #nb_video_id video for each playlist (10 channels -> 20 playlists; #nb_video_id video from each playlist -> 20 videos)
    list_playlists = []
    for i in range(1, 11):
        list_playlists.append(str(i) + "p0")
        list_playlists.append(str(i) + "p1")
    nb_miniclips = 0
    set_videos = set()
    set_playlist = set()

    index_key = 0
    line = []
    max_nb_actions_per_video = 7

    with open(PATH_INPUT_AMT, "wb+") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        header = ["video_url1", "actions1", "video_url2", "actions2", "video_url3", "actions3", "video_url4",
                  "actions4", "video_url5", "actions5"]
        writer.writerow(header)
        index_key = 0
        line = []
        break_all_loops = 0

        for key in filtered_clip_actions_time.keys():
            playlist_id = key.split(", '(")[1].split(",")[0]
            video_id = str(key.split(", '(")[1].split(", ")[1][:-3])
            if playlist_id in list_playlists and video_id in PARAM_nb_video_id:
                nb_miniclips += 1
                set_videos.add(playlist_id + "_" + video_id)
                set_playlist.add(playlist_id)

                actions = filtered_clip_actions_time[key][1]
                mini_index = key.split(", '(")[0][1:]
                video_url = playlist_id + "_" + video_id + "mini_" + str(mini_index) + '.mp4'

                cont = 1
                nb_actions = 0

                while cont == 1:
                    index_key += 1
                    string_actions = ""
                    for a in actions:
                        string_actions += a + ";"
                        nb_actions += 1
                        if (nb_actions % PARAM_MAX_NB_ACTIONS_PER_MINICLIP == 0):
                            break

                    line.append(video_url)
                    line.append(string_actions[:-1].encode('utf-8'))
                    if (index_key % 4 == 0):
                        writer.writerow(line)

                        line = []

                    if (nb_actions % PARAM_MAX_NB_ACTIONS_PER_MINICLIP == 0 and nb_actions < len(actions)):
                        cont = 1
                        actions = actions[nb_actions:]
                    else:
                        cont = 0

    print "There are " + str(nb_miniclips) + " miniclips and " + str(len(set_videos)) + " videos " + str(
        len(set_playlist)) + " playlists"


def write_in_csv_file(csv_file_name, filtered_clip_actions_time, max_nb_actions_per_video):
    with open(csv_file_name, "wb+") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        header = ["video_url1", "actions1", "video_url2", "actions2", "video_url3", "actions3", "video_url4",
                  "actions4", "video_url5", "actions5"]
        writer.writerow(header)
        index_key = 0
        line = []
        break_all_loops = 0

        for key in filtered_clip_actions_time.keys():

            mini_index = key[0]
            channel_video_index = key[1][1:-1]
            print key, channel_video_index
            channel_index, video_index = channel_video_index.split(", ")

            video_url = channel_index + "_" + video_index + "mini_" + str(mini_index) + '.mp4'
            actions = filtered_clip_actions_time[key][1]

            cont = 1
            nb_actions = 0
            # if (break_all_loops == 1):
            #     break

            while cont == 1:
                index_key += 1
                string_actions = ""
                for a in actions:
                    string_actions += a + ";"
                    nb_actions += 1
                    if (nb_actions % max_nb_actions_per_video == 0):
                        break

                line.append(video_url)
                line.append(string_actions[:-1])
                if (index_key % 5 == 0):
                    writer.writerow(line)

                    line = []

                # if(index_key >= len(filtered_clip_actions_time.keys())):
                #     # if video_url == '1_5mini_4.mp4':
                #     #     print "Haa4"
                #     rest_nb_actions = (10 - len(line)) / 2
                #     #put the first rest_nb_actions
                #     #line = put_first_rest_nb_actions(line, rest_nb_actions)
                #     for elem in range(0,rest_nb_actions * 2):
                #         line.append('nothing')
                #     writer.writerow(line)
                #     break_all_loops = 1
                #     break

                if (nb_actions % 7 == 0 and nb_actions < len(actions)):
                    cont = 1
                    actions = actions[nb_actions:]
                else:
                    cont = 0

    return


def add_ground_truth_in_csv_AMT(IN_csv_file_name, ground_truth_csv_file_name, OUT_csv_file_name):
    # with open(OUT_csv_file_name, "wb+") as csv_file2:
    #     fieldnames = ['video_url1', 'actions1','video_url2', 'actions2','video_url3', 'actions3','video_url4', 'actions4','video_url5', 'actions5']
    #     writer = csv.DictWriter(csv_file2, fieldnames=fieldnames)
    #     writer.writeheader()

    #     nb_lines_to_write = 80 # 80 hits 
    #     with open(IN_csv_file_name, "r") as csv_file1:
    #         reader1 = csv.DictReader(csv_file1)

    #         video_url1 = ""
    #         video_url2 = ""
    #         video_url3 = ""
    #         video_url4 = ""
    #         actions_1 = ""
    #         actions_2 = ""
    #         actions_3 = ""
    #         actions_4 = ""
    #         for row1 in reader1:
    #             nb_lines_to_write -= 1 
    #             for (k,v) in row1.items():
    #                 if k == "video_url1":
    #                     video_url1 = v
    #                 if k == "actions1":
    #                     actions_1 = v
    #                 if k == "video_url2":
    #                     video_url2 = v
    #                 if k == "actions2":
    #                     actions_2 = v
    #                 if k == "video_url3":
    #                     video_url3 = v
    #                 if k == "actions3":
    #                     actions_3 = v
    #                 if k == "video_url4":
    #                     video_url4 = v
    #                 if k == "actions4":
    #                     actions_4 = v

    #             writer.writerow({'video_url1':video_url1, 'actions1':actions_1,'video_url2':video_url2, 'actions2':actions_2,'video_url3':video_url3, 'actions3':actions_3,'video_url4':video_url4, 'actions4':actions_4})

    #             if nb_lines_to_write == 0:
    #                 break

    # nb_lines_wrote = 80 - nb_lines_to_write
    # print nb_lines_wrote

    # nb_lines_to_write = 80
    # nb_cells_to_write = nb_lines_to_write * 2 # 80 hits 
    nb_cells_to_write = 0
    ok = 1
    ok_continue = 1
    what_to_write = []
    with open(ground_truth_csv_file_name, "r") as csv_file3:

        reader2 = csv.DictReader(csv_file3)
        for rowdict in reader2:
            video_url5 = ""
            actions_5 = ""
            for f in reader2.fieldnames:
                if ok_continue == 0:
                    ok_continue = 1
                    continue
                if rowdict[f] in ['3mini_22.mp4', '3mini_13.mp4']:
                    ok_continue = 0
                    continue
                if 'video_url' in f:
                    video_url5 = rowdict[f]
                    nb_cells_to_write -= 1
                if 'actions' in f:
                    actions_5 = rowdict[f]
                    nb_cells_to_write -= 1

                if nb_cells_to_write % 2 == 0 and video_url5 != "" and actions_5 != "":
                    # writer.writerow({'video_url5':video_url5, 'actions5':actions_5})
                    what_to_write.append((video_url5, actions_5))
                # if nb_cells_to_write <= 0:
                #     ok = 0
                #     break
            # if ok == 0:
            #     break

    in_file = open(IN_csv_file_name, "rb")
    reader = csv.reader(in_file)
    out_file = open(OUT_csv_file_name, "wb")
    writer = csv.writer(out_file)

    index_write = 0
    for row in reader:
        if index_write > 0:
            row.append(what_to_write[index_write][0])
            row.append(what_to_write[index_write][1])
        index_write += 1
        if index_write >= len(what_to_write):
            index_write = 1
        writer.writerow(row)

    in_file.close()
    out_file.close()

    return


def add_miniclips_for_test(ground_truth_folder, input_folder, output_folder, IN_csv_file_name):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    with open(IN_csv_file_name, "r") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            for (k, v) in row.items():
                if "video_url" in k and 'p' in v.split('_')[0] and not os.path.exists(output_folder + v):
                    shutil.copyfile(input_folder + v, output_folder + v)
                if "video_url" in k and 'p' not in v.split('_')[0] and not os.path.exists(output_folder + v):
                    shutil.copyfile(ground_truth_folder + v, output_folder + v)

    return


def create_input_GT(filtered_clip_actions_time, output_AMT_GT):
    # put 1st video for each playlist (10 channels -> 20 playlists; 1st video from each playlist -> 20 videos)
    list_playlists = []
    for i in range(9, 11):
        list_playlists.append(str(i) + "p0")
        list_playlists.append(str(i) + "p1")
    nb_miniclips = 0
    set_videos = set()
    set_playlist = set()

    index_key = 0
    line = []
    max_nb_actions_per_video = 7

    with open(output_AMT_GT, "wb+") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        header = ["video_url1", "actions1", "video_url2", "actions2", "video_url3", "actions3", "video_url4",
                  "actions4", "video_url5", "actions5"]
        writer.writerow(header)
        index_key = 0
        line = []
        break_all_loops = 0

        for key in filtered_clip_actions_time.keys():
            playlist_id = key.split(", '(")[1].split(",")[0]
            video_id = str(key.split(", '(")[1].split(", ")[1][:-3])
            if playlist_id in list_playlists and video_id == '1':
                nb_miniclips += 1
                set_videos.add(playlist_id + "_" + video_id)
                set_playlist.add(playlist_id)

                actions = filtered_clip_actions_time[key][1]
                mini_index = key.split(", '(")[0][1:]
                video_url = playlist_id + "_" + video_id + "mini_" + str(mini_index) + '.mp4'

                cont = 1
                nb_actions = 0

                while cont == 1:
                    index_key += 1
                    string_actions = ""
                    for a in actions:
                        string_actions += a + ";"
                        nb_actions += 1
                        if (nb_actions % max_nb_actions_per_video == 0):
                            break

                    line.append(video_url)
                    line.append(string_actions[:-1])
                    if (index_key % 5 == 0):
                        writer.writerow(line)

                        line = []

                    if (nb_actions % 7 == 0 and nb_actions < len(actions)):
                        cont = 1
                        actions = actions[nb_actions:]
                    else:
                        cont = 0

    print "There are " + str(nb_miniclips) + " miniclips and " + str(len(set_videos)) + " videos " + str(
        len(set_playlist)) + " playlists"

    return


def write_video_times(clip_actions_time):
    with open("time_file.csv", 'w+') as csvfile:
        fieldnames = ['miniclip_name', 'time(seconds)']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for (miniclip_index, channel_video_index) in clip_actions_time.keys():
            channel_index, video_index = channel_video_index.split(",")
            channel_index = channel_index[1:]
            video_index = video_index[1:-1]
            video_name = channel_index + "_" + video_index + "mini_" + str(miniclip_index)
            [start_time, end_time] = clip_actions_time[(miniclip_index, channel_video_index)][0]
            FMT = '%M:%S'
            tdelta = datetime.strptime(end_time, FMT) - datetime.strptime(start_time, FMT)

            tdelta_h, tdelta_min, tdelta_seconds = str(tdelta).split(":")
            # print clip_actions_time[(miniclip_index,channel_video_index)][1], tdelta,tdelta_h,tdelta_min,tdelta_seconds
            tdelta_total = int(tdelta_seconds) + int(tdelta_min) * 60 + int(tdelta_h) * 360
            # write in file
            writer.writerow({'miniclip_name': video_name, 'time(seconds)': tdelta_total})

    csvfile.close()


def create_miniclips(PATH_actions_file, PATH_videos, PATH_miniclips):
    video_dict = get_actions_time_from_csv(PATH_actions_file)

    print("Now resizing the time (3 seconds)")
    resized_video_dict = resize_time_per_action(video_dict, 3)
    clip_actions_time = process_time_per_action(resized_video_dict, video_dict)

    # with open('actions_time_FINAL.json', 'w+') as outfile:  
    #     json.dump(clip_actions_time, outfile)

    print("Write time per miniclip")
    # write_video_times(clip_actions_time)

    print("Now making the miniclips")
    # make_mini_clips(clip_actions_time, PATH_videos, PATH_miniclips)

    return clip_actions_time

### separate function
# print "Now writing in csv"
# out_csv_file_name = 'videos/test_AMT2/file_list2.csv'
# max_nb_actions_per_video = 7
# write_in_csv_file( out_csv_file_name, clip_actions_time, max_nb_actions_per_video)
