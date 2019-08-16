from __future__ import print_function
from __future__ import print_function
import os
from collections import OrderedDict

import numpy as np
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet

from sklearn.metrics.pairwise import cosine_similarity
from itertools import product

from classify import preprocess
from classify.preprocess import process_data
from classify.visualization import get_list_actions_for_label
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score


def process_output_yolo(path_YOLO_output):
    objects_dict = OrderedDict()

    folders = os.listdir(path_YOLO_output)
    for video_name in folders:
        objects_dict[video_name + '.mp4'] = set()
        results_file = path_YOLO_output + video_name + "/result.txt"
        with open(results_file) as f:
            contents = f.read()
        f.close()

        lines = contents.split("Enter Image Path:")
        for line in lines[1:-1]:
            frame_nb = int(line.split(".jpg")[0].split("/")[-1])
            objects_more_list = line.split("\n")[1:]

            for object_more in objects_more_list:
                object = object_more.split(":")[0]
                if object != "":
                    objects_dict[video_name + '.mp4'].add(object)
    return objects_dict


def cosine_similarity_lists(list_objects, list_word_action, embedding_index):
    wbd_obj = []
    wbd_nouns = []
    for word in list_objects:
        wbd = preprocess.get_word_embedding(embedding_index, word)
        if wbd is not None:
            wbd_obj.append(wbd)
    for word in list_word_action:
        wbd = preprocess.get_word_embedding(embedding_index, word)
        if wbd is not None:
            wbd_nouns.append(wbd)

    embedding_matrix_objects = np.array(wbd_obj)  # embedding matrices
    embedding_matrix_nouns = np.array(wbd_nouns)
    if wbd_obj == [] or wbd_nouns == []:
        best = 0
    else:
        similarity_matrix = cosine_similarity(embedding_matrix_objects, embedding_matrix_nouns)
        best = similarity_matrix.max()

    return best


def measure_by_method(action, list_objects, method, embedding_index):
    nouns = [token for token, pos in pos_tag(word_tokenize(action)) if pos.startswith('N')]
    list_word_action_all = action.split(" ")
    list_word_action_nouns = nouns

    if list_objects == [] or list_word_action_all == [] or (list_word_action_nouns == [] and (
            method == 'wup_sim' or method == 'cos_sim nouns')):
        best = 0

    else:
        if method == 'wup_sim':
            # speed it up by collecting the synsets for all words in list_objects and list_word_action once, and taking the product of the synsets.
            allsyns1 = set(ss for word in list_objects for ss in wordnet.synsets(word))
            allsyns2 = set(ss for word in list_word_action_nouns for ss in wordnet.synsets(word))

            if allsyns1 == set([]) or allsyns2 == set([]):
                best = 0
            else:
                best, s1, s2 = max((wordnet.wup_similarity(s1, s2) or 0, s1, s2) for s1, s2 in product(allsyns1, allsyns2))

        elif method == 'cos_sim all':
            best = cosine_similarity_lists(list_objects, list_word_action_all, embedding_index)
        elif method == 'cos_sim nouns':
            best = cosine_similarity_lists(list_objects, list_word_action_nouns, embedding_index)
        else:
            raise ValueError("wrong similarity method name")
    return best


def measure_similarity(method, threshold, objects_dict, embedding_index, train_data, test_data, val_data):

    [train_actions, test_actions, val_actions], [train_labels, test_labels, val_labels], [train_miniclips, test_miniclips, val_miniclips] = process_data(train_data, test_data,
                                                                                                      val_data)
    index = 0
    yolo_labels = []
    gt_labels = test_labels
    for video in test_miniclips:
        if video in objects_dict.keys():
            action = test_actions[index]
            index += 1
            list_objects = objects_dict[video]
            score_yolo = measure_by_method(action, list_objects, method, embedding_index)
            if score_yolo >= threshold:
                yolo_labels.append(0)
            else:
                yolo_labels.append(1)
        else:
            yolo_labels.append(1)

    accuracy = accuracy_score(gt_labels, yolo_labels)
    f1 = f1_score(gt_labels, yolo_labels)
    recall = recall_score(gt_labels, yolo_labels)
    precision = precision_score(gt_labels, yolo_labels)
    return [0, 0, accuracy, recall, precision, f1], yolo_labels