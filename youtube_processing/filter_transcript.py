#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import os
import sys
import glob
import json
import re
from collections import OrderedDict
import shutil

def srt_time_to_seconds(time):
    split_time=time.split(',')
    major, minor = (split_time[0].split(':'), split_time[1])
    [hour, minute, sec] = major
    return [hour, minute, sec]
    #return int(major[0])*1440 + int(major[1])*60 + int(major[2]) + float(minor)/1000

def srt_to_dict(srtText):
    subs=[]
    for s in re.sub('\r\n', '\n', srtText).split('\n\n'):
        st = s.split('\n')
        if len(st)>=3:
            if st[0] == '':
                st = st[1:]
            split = st[1].split(' --> ')
            start_time = srt_time_to_seconds(split[0].strip())
            end_time = srt_time_to_seconds(split[1].strip())
            if start_time != end_time:
                subs.append({'start': start_time,
                            'end': end_time,
                            'text': '<br />'.join(j for j in st[2:len(st)])
                            })
        
    return subs

def filter_transcripts(PATH_captions, PATH_problematic_transcripts, PATH_problematic_videos, PARAM_LEAST_NB_WORDS_PER_SEC):
    
    list_captions = sorted(glob.glob(PATH_captions+"*.en.srt"), key=os.path.getmtime)
    data = OrderedDict()
    for k in range(0,len(list_captions)):
        with open(list_captions[k], "r") as f:
            srtText = f.read()
            subs = srt_to_dict(srtText)
        
            for i in range(0,len(subs) - 1):
                for j in range(i+1,len(subs)):
                    if '<br />' in subs[i]['text']:
                        if subs[i]['text'].split('<br />')[1] in subs[j]['text']:
                            subs[i]['text'] = subs[i]['text'].split('<br />')[0]
            
            for i in range(0,len(subs)):
                if '<br />' in subs[i]['text']:
                   # if subs[i]['text'].split('<br />')[1][0].islower():
                    subs[i]['text'] = subs[i]['text'].replace('<br />',' ')
            
            list_name_file = []
            list_name_file = list_captions[k].split("/")
            name_file = list_name_file[-1:][0]
            channel_id = name_file.split("caption")[0]
            video_id = name_file.split("_")[1].split(".en.srt")[0]

            
    

            # how much music the transcripts have
            # count_music = 0
            # for i in range(0,len(subs)):
            #     if '[Music]' in subs[i]['text']:
            #         count_music += 1
            
            # print k, count_music
            #how many words per second
            list_total_words = []
            if(subs != []):
                for i in range(0,len(subs)):
                    list_total_words.append(subs[i]['text'].split(" "))

                flat_list = [item for sublist in list_total_words for item in sublist]
                total_words = len(flat_list)
            
                [hours,minutes,sec] = subs[-1:][0]['end']
                total_seconds = int(minutes) * 60 + int(sec)

                nb_words_per_sec = 1.0 * total_words/ total_seconds
                name_video = channel_id + "video_" + video_id + ".mp4"

                #print name_video, " ", nb_words_per_sec

                #problematic videos (not enough talking, more music ...)
                if(nb_words_per_sec < PARAM_LEAST_NB_WORDS_PER_SEC):
                   # print channel_id + ", " + video_id
                    name_transcript = channel_id + "caption_" + video_id + ".en.srt"
                    name_video = channel_id + "video_" + video_id + ".mp4"
                    path_video = '/'.join(list_captions[k].split("/")[0:-2]) + "/videos/" + name_video
                    
                    #move video & transcript to another folder
                    shutil.move(list_captions[k], PATH_problematic_transcripts + name_transcript)
                    shutil.move(path_video, PATH_problematic_videos + name_video)

                else:
                    #write in json file
                    data["(" + channel_id + ", " + video_id + ")"]= subs
    
    with open('new_data.json', 'w+') as outfile:  
        json.dump(data, outfile)



        