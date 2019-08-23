#!/usr/bin/env python2

import os
import sys
import pandas as pd
import time
import glob
import filecmp


def read_video_url_file(csv_file_name, MAX_NB_URL_PER_CHANNEL):
    df = pd.read_csv(csv_file_name)
    df = df.dropna()
    list_urls = df["Video_URL"][:MAX_NB_URL_PER_CHANNEL].values.tolist()
    list_video_names = df["Video_Name"][:MAX_NB_URL_PER_CHANNEL].values.tolist()

    return list_urls, list_video_names


def save_video_names(output_folder_path, list_video_titles, list_video_names):
    if not os.path.exists(output_folder_path + "/captions/"):
        os.makedirs(output_folder_path + "/captions/")

    list_all = [list_video_names, list_video_titles]
    print(list_video_names)
    print(list_video_titles)
    df = pd.DataFrame(list_all)
    df = df.transpose()
    df.columns = ["Video Name", "Video Title"]
    df.to_csv('data/Video/video_titles.csv', index=False)


def save_videos_captions(output_folder_path, list_urls, index_channel, index_playlist):
    index_video = 1
    print("Save videos and their vtt captions")
    ok_remove_captions = 0

    if not os.path.exists(output_folder_path + "/videos/"):
        os.makedirs(output_folder_path + "/videos/")

    if not os.path.exists(output_folder_path + "/captions/"):
        os.makedirs(output_folder_path + "/captions/")

    for url in list_urls:
        command_save_video = 'youtube-dl -f bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4 -v -o ' + output_folder_path + "/videos/" + str(
            index_channel) + "p" + str(index_playlist) + "video_" + str(index_video) + " " + url
        os.system(command_save_video)

        command_save_caption = 'youtube-dl --skip-download --write-sub --write-auto-sub --sub-lang "en" -o ' + output_folder_path + "/captions/" + str(
            index_channel) + "p" + str(index_playlist) + "caption_" + str(index_video) + " " + url
        os.system(command_save_caption)

        index_video += 1
    if (index_video == len(list_urls) + 1):
        ok_remove_captions = 1

    # #time.sleep(60)
    index_video = 1

    if (ok_remove_captions == 1):
        print("Convert captions to srt and remove the vtt captions")
        for url in list_urls:
            path_caption = output_folder_path + "/captions/" + str(index_channel) + "p" + str(
                index_playlist) + "caption_" + str(index_video) + '.en'
            command_convert_caption = 'ffmpeg -i ' + path_caption + '.vtt ' + path_caption + '.srt'
            os.system(command_convert_caption)

            command_remove_vtt_caption = 'rm ' + path_caption + '.vtt'
            os.system(command_remove_vtt_caption)

            index_video += 1

        print("Convert webm videos to mp4")
        # time.sleep(60)
        index_video = 1
        ok_remove_videos = 0
        for url in list_urls:
            video_path = output_folder_path + "/videos/" + str(index_channel) + "p" + str(
                index_playlist) + "video_" + str(index_video)
            command_convert_video = 'ffmpeg -i ' + video_path + '.webm -codec copy ' + video_path + '.mp4'
            os.system(command_convert_video)

            index_video += 1
        if (index_video == len(list_urls) + 1):
            ok_remove_videos = 1

        #  time.sleep(60)
        index_video = 1
        if (ok_remove_videos == 1):
            print("Remove webm videos")
            for url in list_urls:
                video_path = output_folder_path + "/videos/" + str(index_channel) + "p" + str(
                    index_playlist) + "video_" + str(index_video)
                command_remove_webm_video = 'rm ' + video_path + '.webm'
                os.system(command_remove_webm_video)

                index_video += 1

    list_video_names = [x.split("/")[-1][:-4] for x in glob.glob(output_folder_path + "/videos/" + "*.mp4") if x.split("/")[-1][:-4].split("video")[0].split("p") == [index_channel, index_playlist]]

    list_1_p0 = [x for x in list_video_names if int(x.split("_")[-1]) < 10 and "p0" in x.split("_")[0]]
    list_2_p0 = [x for x in list_video_names if int(x.split("_")[-1]) > 9 and "p0" in x.split("_")[0]]

    list_1_p1 = [x for x in list_video_names if int(x.split("_")[-1]) < 10 and "p1" in x.split("_")[0]]
    list_2_p1 = [x for x in list_video_names if int(x.split("_")[-1]) > 9 and "p1" in x.split("_")[0]]

    list_video_names = sorted(list_1_p0) + sorted(list_2_p0) + sorted(list_1_p1) + sorted(list_2_p1)
    return list_video_names


def convert_all_part_files(PATH_video_transcripts, list_urls):
    folder_path_videos = PATH_video_transcripts + '/videos/'
    list_files = glob.glob(folder_path_videos + "*.part")

    for part_file in list_files:
        video_name = part_file.split('/')[-1:][0]
        video_path = part_file[0:len(part_file) - len(video_name)]
        new_video_name, _, _ = video_name.split('.')
        _, index_video = new_video_name.split('_')
        # print video_path, video_name, new_video_name,index_video
        command_save_video = 'youtube-dl -f bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4 -v -o ' + str(video_path) + str(
            new_video_name) + ' ' + str(list_urls[int(index_video) - 1])
        os.system(command_save_video)


def get_duplicates(PATH_video_transcripts):
    path_captions = PATH_video_transcripts + '/captions/'
    list_files = glob.glob(path_captions + "*.srt")
    duplicates_list = []
    for i in range(0, len(list_files) - 1):
        for j in range(i + 1, len(list_files)):
            value = filecmp.cmp(list_files[i], list_files[j])
            if (value == True):
                video_name_1 = list_files[i].split('/')[-1:][0]
                new_video_name_1, _, _ = video_name_1.split('.')
                _, index_video1 = new_video_name_1.split('_')
                index_channel1, _ = new_video_name_1.split('caption')

                video_name_2 = list_files[j].split('/')[-1:][0]
                new_video_name_2, _, _ = video_name_2.split('.')
                _, index_video_2 = new_video_name_2.split('_')
                index_channel2, _ = new_video_name_2.split('caption')

                # print index_video1, index_video_2
                duplicates_list.append(index_channel2 + "_" + index_video_2)

    print("There are " + str(len(duplicates_list)) + " duplicates!")
    return duplicates_list


def remove_duplicates(output_folder_path, duplicates_list):
    for index_video_channel in duplicates_list:
        index_channel, index_video = index_video_channel.split('_')

        command_remove_video = 'rm ' + output_folder_path + "/videos/" + str(index_channel) + "video_" + str(
            index_video) + ".mp4"
        os.system(command_remove_video)

        command_remove_caption = 'rm ' + output_folder_path + "/captions/" + str(index_channel) + "caption_" + str(
            index_video) + ".en.srt"
        os.system(command_remove_caption)

    print("Removed " + str(len(duplicates_list)) + " duplicates ")


def compare_video_subtitle(output_folder_path):
    list_captions = glob.glob(output_folder_path + "/captions/" + "*.*")
    list_videos = glob.glob(output_folder_path + "/videos/" + "*.*")

    list_index_captions = []
    for caption_file in list_captions:
        video_name_1 = caption_file.split('/')[-1:][0]
        new_video_name_1, _, _ = video_name_1.split('.')
        _, index_video1 = new_video_name_1.split('_')
        index_channel, _ = new_video_name_1.split('caption')
        list_index_captions.append((index_channel + "_" + index_video1))

    list_index_videos = []
    for video_file in list_videos:
        video_name_1 = video_file.split('/')[-1:][0]
        new_video_name_1, _ = video_name_1.split('.')
        index_channel, _ = new_video_name_1.split('video')
        _, index_video1 = new_video_name_1.split('_')
        list_index_videos.append((index_channel + "_" + index_video1))

    list_videos_no_captions = list(set(list_index_videos) - set(list_index_captions))

    print("There are " + str(len(list_videos_no_captions)) + " no caption videos!")
    return list_videos_no_captions


def remove_no_caption_videos(output_folder_path, list_videos_no_captions):
    for index_video_channel in list_videos_no_captions:
        index_channel, index_video = index_video_channel.split('_')
        command_remove_video = 'rm ' + output_folder_path + "/videos/" + str(index_channel) + "video_" + str(
            index_video) + ".mp4"
        os.system(command_remove_video)

    print("Removed " + str(len(list_videos_no_captions)) + " no caption videos ")


def convert_to_mp4(PATH_video_transcripts):
    list_videos = glob.glob(PATH_video_transcripts + "/videos2/" + "*.*")

    for video_file in list_videos:
        video_path = video_file.split(".")[0]
        command_convert_video = 'ffmpeg -i ' + video_path + '.* -codec copy ' + video_path + '.mp4'
        command_remove_video = 'rm ' + video_path + ".mkv"

        os.system(command_convert_video)


def create_initial_video_transcripts(PATH_csv_url_file, PATH_video_transcripts, MAX_NB_URL_PER_CHANNEL):
    # sort to preserve 1 2 3 order (channel order)
    # list_csv_files = sorted(glob.glob(PATH_csv_url_file + "*.csv"), key=os.path.getmtime)
    list_csv_files = sorted(glob.glob(PATH_csv_url_file + "*.csv"))
    list_video_titles_all = []
    list_video_names_all = []
    for csv_file_name in list_csv_files:
        index_channel, index_playlist = csv_file_name.split("/")[-1][:-4].split("_")
        list_urls, list_video_titles = read_video_url_file(csv_file_name, MAX_NB_URL_PER_CHANNEL)
        list_video_names = save_videos_captions(PATH_video_transcripts, list_urls, index_channel, index_playlist)

        convert_all_part_files(PATH_video_transcripts, list_urls)

        list_video_titles_all += list_video_titles
        list_video_names_all += list_video_names

    duplicates_list = get_duplicates(PATH_video_transcripts)

    #TODO: There is a bug here, it removes both videos (if there are duplicated)
    remove_duplicates(PATH_video_transcripts, duplicates_list)

    list_videos_no_captions = compare_video_subtitle(PATH_video_transcripts)
    remove_no_caption_videos(PATH_video_transcripts, list_videos_no_captions)

    save_video_names(PATH_video_transcripts, list_video_titles_all, list_video_names_all)


