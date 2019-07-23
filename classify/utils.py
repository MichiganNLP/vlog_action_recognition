from __future__ import print_function, absolute_import, unicode_literals, division

import glob
import itertools
import json
import os
from collections import OrderedDict

import pandas as pd
import numpy as np


# from amt.settings import PATH_visible_not_visible_actions_csv

def robust_decode(bs):
    '''Takes a byte string as param and convert it into a unicode one.
First tries UTF8, and fallback to Latin1 if it fails'''
    cr = None
    try:
        cr = bs.decode('utf8')
    except UnicodeDecodeError:
        cr = bs.decode('latin1')
    return cr


def clean_action(action):
    # prepare translation table for removing punctuation
    if not action:
        return action
    action = robust_decode(action)
    if "my vegetables in water instead of oil" in action:
        action = "sauteing my vegetables in water instead of oil"
    list_words = action.split(' ')
    # remove tokens with numbers in them
    action = [word for word in list_words if word.isalpha()]
    action = ' '.join(action)
    list_words = action.split(' ')
    # remove I, you, she, he
    list_words = [word.lower() for word in list_words]
    action = [word for word in list_words if word not in ['you', 'just', 'i', 'I', 'she', 'he']]
    action = ' '.join(action)
    return action


def clean_list_actions(list_actions):
    new_list_actions = []
    for action in list_actions:
        new_list_actions.append(clean_action(action))
    return new_list_actions


def clean_actions_file(clean_visible, clean_not_visible, path_visible_not_visible_actions_csv):
    # path_visible_not_visible_actions_csv = "/local/oignat/action_recognition_clean/data/AMT/Output/All/visible_not_visible_actions_video_after_spam.csv"
    df = pd.read_csv(path_visible_not_visible_actions_csv)

    df.loc[df["Visible Actions"].isnull(), "Visible Actions"] = ""
    list_visible = df["Visible Actions"].values.tolist()
    if clean_visible:
        cleaned_list_visibile = clean_list_actions(list_visible)
    else:
        cleaned_list_visibile = list_visible

    df.loc[df["Not Visible Actions"].isnull(), "Not Visible Actions"] = ""
    list_not_visible = df["Not Visible Actions"].values.tolist()
    if clean_not_visible:
        cleaned_list_not_visibile = clean_list_actions(list_not_visible)
    else:
        cleaned_list_not_visibile = list_not_visible


    list_videos = df["Video_name"].values.tolist()
    # create new df
    dict = OrderedDict()
    dict['Video_name'] = list_videos
    dict['Visible Actions'] = cleaned_list_visibile
    dict['Not Visible Actions'] = cleaned_list_not_visibile

    df_cleaned = pd.DataFrame(dict)
    df_cleaned = df_cleaned.replace(np.nan, '', regex=True)
    path_new_file = "/".join(path_visible_not_visible_actions_csv.split("/")[:-1]) + "/new_clean_" + \
                    path_visible_not_visible_actions_csv.split("/")[-1]

    df_cleaned.to_csv(path_new_file, index=False)


def clean_context_file():
    path_context_csv = "/local/oignat/action_recognition_clean/data/Embeddings/context_embeddings.csv"
    df = pd.read_csv(path_context_csv)
    df.loc[df["action"].isnull(), "action"] = ""
    list_visible = df["action"].values.tolist()
    cleaned_list_visibile = clean_list_actions(list_visible)

    list_videos = df["video"].values.tolist()
    left_context = df["left_context"].values.tolist()
    right_context = df["right_context"].values.tolist()
    # create new df
    dict = OrderedDict()
    dict['video'] = list_videos
    dict['action'] = cleaned_list_visibile
    dict['left_context'] = left_context
    dict['right_context'] = right_context

    df_cleaned = pd.DataFrame(dict)
    df_cleaned = df_cleaned.replace(np.nan, '', regex=True)

    path_new_file = "/".join(path_context_csv.split("/")[:-1]) + "/clean_" + \
                    path_context_csv.split("/")[-1]

    df_cleaned.to_csv(path_new_file, index=False)


def reshape_3d_to_2d(input):
    # img = img.reshape((img.shape[1] *img.shape[2]),img.shape[0])
    # img = img.transpose()
    # return img

    nsamples, nx, ny = input.shape
    input = input.reshape((nsamples, nx * ny))
    return input


def reshape_2d_to_1d(input):
    # img = img.reshape((img.shape[1] *img.shape[2]),img.shape[0])
    # img = img.transpose()
    # return img

    nsamples, nx = input.shape
    input = input.reshape(nx * nsamples)
    return input


def reshape_2d_to_3d(input):
    nsamples, nx = input.shape
    input = input.reshape(nsamples, nx, 1)
    return input


def get_all_combinations(input):
    list_subsets = []
    for L in range(0, len(input) + 1):
        for subset in itertools.combinations(input, L):
            list_subsets.append(subset)
    return list_subsets

def get_all_combinations(input):
    list_subsets = []
    for L in range(0, len(input) + 1):
        for subset in itertools.combinations(input, L):
            list_subsets.append(subset)
    return list_subsets

def merge_csv_files():
    a = pd.read_csv('data/Embeddings/pos_embeddings_test.csv')
    b = pd.read_csv('data/Embeddings/pos_embeddings_train.csv')
    c = pd.read_csv('data/Embeddings/pos_embeddings_val.csv')

    e = a.append(b)

    f = e.append(c)

    f.to_csv("data/Embeddings/pos_embeddings.csv", index=False)



    print(a.shape, b.shape, c.shape, f.shape, e.shape)


def read_labels_inception_files():
    path_input_inception = "/local/oignat/action_recognition_clean/data/YOLO/Features/inception_labels/"
    os.chdir(path_input_inception)
    list_files = glob.glob('*.npy')
    with open('/local/oignat/action_recognition_clean/data/YOLO/Features/imagenet_class_index.json', 'r') as f:
        dict_labels = json.loads(f.read())
    for file in list_files:
        print("For file " + file)
        inception_features = np.load(str(file))
        for i in range(inception_features.shape[0]):
            frame = inception_features[i,:]
            max_probability = np.max(frame)
            list_index_max_probability = np.where(frame == np.amax(frame))
            label = dict_labels[str(list_index_max_probability[0][0])][1]
            print("For frame " + str(i) + "; Max prob " + str(max_probability) + "; Label " + str(label))
        print("----------------------------------------------")



def read_npy_files():
    path_input_inception = "/local/oignat/action_recognition_clean/data/YOLO/Features/corrected_inception/"
    os.chdir(path_input_inception)
    list_files = glob.glob('*.npy')
    for file in list_files:
        print("For file " + file)
        inception_features = np.load(str(file))
        break
    path_input_inception_old = "/local/oignat/action_recognition_clean/data/YOLO/Features/visual/"
    os.chdir(path_input_inception_old)
    list_files = glob.glob('*.npy')
    for file in list_files:
        print("For file " + file)
        inception_features2 = np.load(str(file))
        break


def main():
    # clean_actions_file()
    # clean_context_file()    # clean_actions_file()
    # clean_context_file()
    merge_csv_files()


if __name__ == '__main__':
    read_npy_files()
