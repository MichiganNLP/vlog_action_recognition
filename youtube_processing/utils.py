from __future__ import print_function

import json

import nltk
import pandas as pd


def map_action_to_context(dict_video_actions):

    list_miniclips = ["4mini_19.mp4", "1mini_13.mp4", "7mini_10.mp4", "1mini_40.mp4", "6mini_2.mp4", "6mini_17.mp4",
                      "4mini_15.mp4", "7mini_31.mp4", "1mini_17.mp4", "2mini_21.mp4", "4mini_18.mp4", "4mini_4.mp4",
                      "4mini_9.mp4", "7mini_26.mp4", "1mini_22.mp4", "7mini_19.mp4", "1mini_15.mp4", "1mini_8.mp4",
                      "3mini_5.mp4", "7mini_36.mp4", "4mini_10.mp4", "7mini_29.mp4"]
    path_context_file = 'data/Miniclips_DATA/GT_actions_punct.csv'



    pd.options.display.max_colwidth = -1
    df_context = pd.read_csv(path_context_file)
    dict_context = {}

    for miniclip in list_miniclips:
        video_id, _ = miniclip.split("mini")

        df_match_transcript = df_context.loc[df_context['transcript_id'] == int(video_id)]
        for index_visible_not_visible in [0, 1]:
            list_actions_per_miniclip = dict_video_actions[miniclip][index_visible_not_visible]
            for action in list_actions_per_miniclip:
                # pre-process: lower case
                df_lower = df_match_transcript['actions'].str.lower().str.encode('utf8')
                df_match_action = df_match_transcript.loc[df_lower == action]
                sentence = df_match_action['sentences'].to_string(index=False).lower()
                # sentence = sentence.to_string(index = False)

                no_time_sentence = nltk.re.sub('\[*[0-9]*:[0-9]*\]*', '', sentence)

                # end_time = df_match_action['end_time']
                # start_time = df_match_action['start_time']

                if miniclip not in dict_context.keys():
                    dict_context[miniclip] = [[], []]

                dict_context[miniclip][index_visible_not_visible].append(
                    [action, no_time_sentence])


    with open("data/dict_context_GT.json", 'w') as f:
        json.dump(dict_context, f)
