import pandas as pd
import os
import csv
from collections import OrderedDict
import glob
import numpy as np
from shutil import copytree
import string
import json
from tqdm import tqdm

from nltk.tag import StanfordPOSTagger
from nltk import PorterStemmer
stemmer = PorterStemmer()

os.environ["CLASSPATH"] = "stanford-postagger-full-2018-10-16/"
os.environ["STANFORD_MODELS"] = "stanford-postagger-full-2018-10-16/models/"

st = StanfordPOSTagger('english-bidirectional-distsim.tagger')


path_visible_not_visible_actions_csv = 'data/AMT/Output/All/new_clean_visible_not_visible_actions_video_after_spam.csv'

glove = pd.read_table("data/glove.6B.50d.txt", sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
table = str.maketrans({key: None for key in string.punctuation})

glove_pos = pd.read_table("data/glove_vectors.txt", sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)

# Retrieve embedding for a word
def vec(w, glove_emb):
    return glove_emb.loc[w].as_matrix()


def getStartEnd(action, action_context):
    action = action.split()
    action = [i.translate(table) for i in action if i.isalpha()]
    action_context = [i.translate(table) for i in action_context.split()]
    possible_beginnings = [i for i in range(len(action_context)) if action_context[i] == action[0]]
    tenable_beginnings = []
    tenable_endings = []
    for beginning in possible_beginnings:
        current_spot = beginning
        tenable = True
        for word in action[1:]:
            ok = 0
            if word in action_context[current_spot + 1:]:
                current_spot = action_context[current_spot + 1:].index(word) + current_spot + 1
                ok = 1
            else:
                for l in action_context[current_spot + 1:]:
                    if word in l:
                        current_spot = action_context[current_spot + 1:].index(l) + current_spot + 1
                        ok = 1
                        break

                if ok == 0:
                    tenable = False
                    break
        if tenable:
            tenable_beginnings.append(beginning)
            tenable_endings.append(current_spot)

    beginning = tenable_beginnings[-1]
    ending = tenable_endings[-1]
    return (beginning, ending)


def getPOSEmbeddings(action, action_context):
    action = action.replace("y' all", "y'all")
    tagged_sentences = st.tag(action_context.split())

    (beginning, ending) = getStartEnd(action, action_context)
    action_pos = tagged_sentences[beginning:ending + 1]

    pos_representation = [0] * 50
    count = 0
    for (word, pos) in action_pos:
        if pos in glove_pos.index:
            count += 1
            pos_representation += vec(pos, glove_pos)
    if count > 0:
        pos_representation /= count

    return pos_representation



def getContextEmbeddings(action, action_context, context_size=5):
    action = action.replace("y' all", "y'all")
    (beginning, ending) = getStartEnd(action, action_context)
    if beginning - context_size < 0:
        left_context = action_context.split()[0:beginning]
    else:
        left_context = action_context.split()[beginning-context_size:beginning]
    if ending + 1 + context_size > len(action_context.split()):
        right_context = action_context.split()[ending + 1:]
    else:
        right_context = action_context.split()[ending + 1:ending + 1 + context_size]
    left_representation = [0] * 50
    count = 0
    for word in left_context:
        if word in glove.index:
            count += 1
            left_representation += vec(word, glove)
    if count > 0:
        left_representation /= count
    right_representation = [0] * 50
    count = 0
    for word in right_context:
        if word in glove.index:
            count += 1
            right_representation += vec(word, glove)
    if count > 0:
        right_representation /= count

    return (right_representation, left_representation)

def get_pos_emb_all():
    video_list = []
    action_list = []
    pos_embedding_list = []

    with open('data/dict_context.json', 'r') as fp:
        context = json.load(fp)

    for video in tqdm(context.keys()):
        for action in context[video].keys():
            action_context = context[video][action]
            if action_context != []:
                pos_embedding = getPOSEmbeddings(action, action_context)
            else:
                pos_embedding = [0] * 50
            video_list.append(video)
            action_list.append(action)
            pos_embedding_list.append(pos_embedding)

    results_train = pd.DataFrame({'video': video_list, 'action': action_list, 'pos_embedding': pos_embedding_list})

    results_train.to_csv("data/Embeddings/new_pos_embeddings.csv")


def get_context_emb_all():
    video_list = []
    action_list = []
    left_context_list = []
    right_context_list = []
    with open('data/dict_context.json', 'r') as fp:
        context = json.load(fp)

    for video in context.keys():
        for action in context[video].keys():
            action_context = context[video][action]
            if action_context != []:
                (left_context, right_context) = getContextEmbeddings(action, action_context)
            else:
                (left_context, right_context) = ([0] * 50, [0] * 50)

            video_list.append(video)
            action_list.append(action)
            left_context_list.append(left_context)
            right_context_list.append(right_context)

    results_train = pd.DataFrame({'video': video_list, 'action': action_list, 'left_context': left_context_list, \
                                  'right_context': right_context_list})

    results_train.to_csv("data/Embeddings/context_embeddings.csv")


# def get_context_emb_per_action():
#     with open('data/dict_context.json', 'r') as fp:
#         context = json.load(fp)
#
#     dict_contex_embeddings = {}
#
#     for video in context.keys():
#         for action in context[video].keys():
#             action_context = context[video][action]
#             if action_context != []:
#                 (left_context, right_context) = getContextEmbeddings(action, action_context)
#             else:
#                 (left_context, right_context) = (np.zeros(50), np.zeros(50))
#
#             left_right_context = np.concatenate((left_context, right_context), axis=0)
#             dict_contex_embeddings[str((video, action))] = left_right_context.tolist()
#
#     with open('data/dict_context_embeddings.json', 'w+') as fp:
#         json.dump(dict_contex_embeddings, fp)




def get_list_visibile_actions(list_all_actions):
    list_visibile_actions = []

    for l in list_all_actions:
        if l[1] == 0:
            list_visibile_actions.append(l[0])
    return list_visibile_actions


def modify_context():
    new_context = {}
    with open('data/new_context.json', 'r') as fp:
        context = json.load(fp)

    with open('data/miniclip_actions.json', 'r') as fp:
        miniclip_actions = json.load(fp)

    intersection_keys = set(context).intersection(miniclip_actions)

    for key in intersection_keys:
        list_all_action_context = context[key][0] + context[key][1]

        dict_action_context = {}
        for [action, context_1] in list_all_action_context:
            dict_action_context[action] = context_1

        list_action_no_context = dict_action_context.keys()
        list_all_actions = [l[0] for l in miniclip_actions[key]]

        new_dict_action_context = {}

        all_context = dict_action_context.values()

        for action in list_all_actions:
            new_dict_action_context[action] = []

        for action2 in list_action_no_context:
            for action in list_all_actions:
                if action in action2:
                    new_dict_action_context[action] = dict_action_context[action2]
                    break

        for action in new_dict_action_context.keys():
            if new_dict_action_context[action] == []:
                for context_2 in all_context:
                    if action in context_2:
                        new_dict_action_context[action] = context_2
                        break

        new_context[key] = new_dict_action_context

    difference = list(set(miniclip_actions) - set(intersection_keys))

    for key in difference:
        list_all_actions = [l[0] for l in miniclip_actions[key]]
        new_dict_action_context = {}
        for action in list_all_actions:
            new_dict_action_context[action] = []

        new_context[key] = new_dict_action_context

    with open('data/new_new_context.json', 'w+') as fp:
        json.dump(new_context, fp)



def modify_name_yolo():
    path_yolo = "data/YOLO/miniclips_results/"
    my_dirs = [d for d in os.listdir(path_yolo)]
    new_path_yolo = "data/YOLO/new_miniclips_results/"

    for file in my_dirs:
        if len(file.split("_")) == 3:
            channel_playlist, video, miniclip = file.split("_")
            channel, playlist = channel_playlist.split("p")
            video = video[:-4]

            if channel == '2' and playlist == '1':
                channel = '3'
                playlist = '1'
                new_miniclip = channel + 'p' + playlist + "_" + video + "mini" + "_" + miniclip + ".npy"
                copytree(path_yolo + file, new_path_yolo + new_miniclip)


            elif channel == '1' or channel == '9' or channel == '10':
                copytree(path_yolo + file, new_path_yolo + file)

            elif channel == '2' and playlist == '0':
                copytree(path_yolo + file, new_path_yolo + file)

            elif channel == '3' and playlist == '0':
                copytree(path_yolo + file, new_path_yolo + file)

            elif channel == '4' and playlist == '1':
                copytree(path_yolo + file, new_path_yolo + file)

            elif channel == '5' and playlist == '1':
                copytree(path_yolo + file, new_path_yolo + file)

            elif channel == '6' and playlist == '1':
                copytree(path_yolo + file, new_path_yolo + file)

            elif channel == '7' and playlist == '1':
                copytree(path_yolo + file, new_path_yolo + file)

            elif channel == '5' and playlist == '0':
                channel = '2'
                playlist = '1'
                new_miniclip = channel + 'p' + playlist + "_" + video + "mini" + "_" + miniclip + ".npy"
                copytree(path_yolo + file, new_path_yolo + new_miniclip)

            elif channel == '3' and playlist == '1':
                channel = '4'
                playlist = '0'
                new_miniclip = channel + 'p' + playlist + "_" + video + "mini" + "_" + miniclip + ".npy"
                copytree(path_yolo + file, new_path_yolo + new_miniclip)

            elif channel == '8' and playlist == '1':
                channel = '5'
                playlist = '0'
                new_miniclip = channel + 'p' + playlist + "_" + video + "mini" + "_" + miniclip + ".npy"
                copytree(path_yolo + file, new_path_yolo + new_miniclip)

            elif channel == '7' and playlist == '0':
                channel = '6'
                playlist = '0'
                new_miniclip = channel + 'p' + playlist + "_" + video + "mini" + "_" + miniclip + ".npy"
                copytree(path_yolo + file, new_path_yolo + new_miniclip)

            elif channel == '8' and playlist == '0':
                channel = '7'
                playlist = '0'
                new_miniclip = channel + 'p' + playlist + "_" + video + "mini" + "_" + miniclip + ".npy"
                copytree(path_yolo + file, new_path_yolo + new_miniclip)

            elif channel == '4' and playlist == '0':
                channel = '8'
                playlist = '0'
                new_miniclip = channel + 'p' + playlist + "_" + video + "mini" + "_" + miniclip + ".npy"
                copytree(path_yolo + file, new_path_yolo + new_miniclip)

            elif channel == '6' and playlist == '0':
                channel = '8'
                playlist = '1'
                new_miniclip = channel + 'p' + playlist + "_" + video + "mini" + "_" + miniclip + ".npy"
                copytree(path_yolo + file, new_path_yolo + new_miniclip)


def extract_c3d_from_inc3cd():
    list_all_files = glob.glob("data/YOLO/Features/inception_c3d/" + "*.npy")
    for file in list_all_files:
        file = file.split("/")[-1]
        inc_c3d = np.load("data/YOLO/Features/inception_c3d/" + file)
        c3d = inc_c3d[:, 2048:]
        np.save("data/YOLO/Features/c3d/" + file, c3d)


def modify_name_visual_feat():
    path_inception = "data/YOLO/Features/corrected_inception/"
    path_c3d = "data/YOLO/Features/visual_c3d/"
    path_inception_c3d = "data/YOLO/Features/corrected_inception_c3d/"
    new_path_inception_c3d = "data/YOLO/Features/new_corrected_inception_c3d/"
    new_path_c3d = "data/YOLO/Features/new_corrected_c3d/"
    new_path_inception = "data/YOLO/Features/new_corrected_inception/"

    list_all_files = glob.glob(path_c3d + "*.npy")
    for file in list_all_files:
        file = file.split("/")[-1]
        if len(file[:-4].split("_")) == 3:
            channel_playlist, video, miniclip = file[:-4].split("_")
            channel, playlist = channel_playlist.split("p")
            video = video[:-4]

            if channel == '2' and playlist == '1':
                channel = '3'
                playlist = '1'
                new_miniclip = channel + 'p' + playlist + "_" + video + "mini" + "_" + miniclip + ".npy"
                os.rename(path_c3d + file, new_path_c3d + new_miniclip)


            elif channel == '1' or channel == '9' or channel == '10':
                os.rename(path_c3d + file, new_path_c3d + file)

            elif channel == '2' and playlist == '0':
                os.rename(path_c3d + file, new_path_c3d + file)

            elif channel == '3' and playlist == '0':
                os.rename(path_c3d + file, new_path_c3d + file)

            elif channel == '4' and playlist == '1':
                os.rename(path_c3d + file, new_path_c3d + file)

            elif channel == '5' and playlist == '1':
                os.rename(path_c3d + file, new_path_c3d + file)

            elif channel == '6' and playlist == '1':
                os.rename(path_c3d + file, new_path_c3d + file)

            elif channel == '7' and playlist == '1':
                os.rename(path_c3d + file, new_path_c3d + file)

            elif channel == '5' and playlist == '0':
                channel = '2'
                playlist = '1'
                new_miniclip = channel + 'p' + playlist + "_" + video + "mini" + "_" + miniclip + ".npy"
                os.rename(path_c3d + file, new_path_c3d + new_miniclip)

            elif channel == '3' and playlist == '1':
                channel = '4'
                playlist = '0'
                new_miniclip = channel + 'p' + playlist + "_" + video + "mini" + "_" + miniclip + ".npy"
                os.rename(path_c3d + file, new_path_c3d + new_miniclip)

            elif channel == '8' and playlist == '1':
                channel = '5'
                playlist = '0'
                new_miniclip = channel + 'p' + playlist + "_" + video + "mini" + "_" + miniclip + ".npy"
                os.rename(path_c3d + file, new_path_c3d + new_miniclip)

            elif channel == '7' and playlist == '0':
                channel = '6'
                playlist = '0'
                new_miniclip = channel + 'p' + playlist + "_" + video + "mini" + "_" + miniclip + ".npy"
                os.rename(path_c3d + file, new_path_c3d + new_miniclip)

            elif channel == '8' and playlist == '0':
                channel = '7'
                playlist = '0'
                new_miniclip = channel + 'p' + playlist + "_" + video + "mini" + "_" + miniclip + ".npy"
                os.rename(path_c3d + file, new_path_c3d + new_miniclip)

            elif channel == '4' and playlist == '0':
                channel = '8'
                playlist = '0'
                new_miniclip = channel + 'p' + playlist + "_" + video + "mini" + "_" + miniclip + ".npy"
                os.rename(path_c3d + file, new_path_c3d + new_miniclip)

            elif channel == '6' and playlist == '0':
                channel = '8'
                playlist = '1'
                new_miniclip = channel + 'p' + playlist + "_" + video + "mini" + "_" + miniclip + ".npy"
                os.rename(path_c3d + file, new_path_c3d + new_miniclip)

        # os.rename(path_inception_c3d + file, "path/to/new/destination/for/file.foo")


def create_json_actions():
    with open(path_visible_not_visible_actions_csv) as csv_file:
        reader = csv.DictReader(csv_file)
        dict_video_actions = OrderedDict()
        for row in reader:
            visible_action = ''
            not_visible_action = ''
            video_name = ''
            for (column_name, value) in row.items():
                if column_name == 'Video_name':
                    video_name = value
                    if video_name not in dict_video_actions.keys():
                        dict_video_actions[video_name] = []
                if column_name == 'Visible Actions':
                    visible_action = value
                if column_name == 'Not Visible Actions':
                    not_visible_action = value

            if visible_action:
                dict_video_actions[video_name].append([visible_action.encode('utf8').lower(), 0])
            if not_visible_action:
                dict_video_actions[video_name].append([not_visible_action.encode('utf8').lower(), 1])

    new_dict = {}
    for miniclip_key in dict_video_actions:

        if len(miniclip_key[:-4].split("_")) == 3:
            channel_playlist, video, miniclip = miniclip_key[:-4].split("_")
            channel, playlist = channel_playlist.split("p")
            video = video[:-4]

            if channel == '2' and playlist == '1':
                channel = '3'
                playlist = '1'
                new_miniclip = channel + 'p' + playlist + "_" + video + "mini" + "_" + miniclip + ".mp4"
                new_dict[new_miniclip] = dict_video_actions[miniclip_key]
            elif channel == '1' or channel == '9' or channel == '10':
                new_dict[miniclip_key] = dict_video_actions[miniclip_key]

            elif channel == '2' and playlist == '0':
                new_dict[miniclip_key] = dict_video_actions[miniclip_key]
            elif channel == '3' and playlist == '0':
                new_dict[miniclip_key] = dict_video_actions[miniclip_key]
            elif channel == '4' and playlist == '1':
                new_dict[miniclip_key] = dict_video_actions[miniclip_key]
            elif channel == '5' and playlist == '1':
                new_dict[miniclip_key] = dict_video_actions[miniclip_key]
            elif channel == '6' and playlist == '1':
                new_dict[miniclip_key] = dict_video_actions[miniclip_key]
            elif channel == '7' and playlist == '1':
                new_dict[miniclip_key] = dict_video_actions[miniclip_key]

            elif channel == '5' and playlist == '0':
                channel = '2'
                playlist = '1'
                new_miniclip = channel + 'p' + playlist + "_" + video + "mini" + "_" + miniclip + ".mp4"
                new_dict[new_miniclip] = dict_video_actions[miniclip_key]
            elif channel == '3' and playlist == '1':
                channel = '4'
                playlist = '0'
                new_miniclip = channel + 'p' + playlist + "_" + video + "mini" + "_" + miniclip + ".mp4"
                new_dict[new_miniclip] = dict_video_actions[miniclip_key]
            elif channel == '8' and playlist == '1':
                channel = '5'
                playlist = '0'
                new_miniclip = channel + 'p' + playlist + "_" + video + "mini" + "_" + miniclip + ".mp4"
                new_dict[new_miniclip] = dict_video_actions[miniclip_key]
            elif channel == '7' and playlist == '0':
                channel = '6'
                playlist = '0'
                new_miniclip = channel + 'p' + playlist + "_" + video + "mini" + "_" + miniclip + ".mp4"
                new_dict[new_miniclip] = dict_video_actions[miniclip_key]
            elif channel == '8' and playlist == '0':
                channel = '7'
                playlist = '0'
                new_miniclip = channel + 'p' + playlist + "_" + video + "mini" + "_" + miniclip + ".mp4"
                new_dict[new_miniclip] = dict_video_actions[miniclip_key]
            elif channel == '4' and playlist == '0':
                channel = '8'
                playlist = '0'
                new_miniclip = channel + 'p' + playlist + "_" + video + "mini" + "_" + miniclip + ".mp4"
                new_dict[new_miniclip] = dict_video_actions[miniclip_key]
            elif channel == '6' and playlist == '0':
                channel = '8'
                playlist = '1'
                new_miniclip = channel + 'p' + playlist + "_" + video + "mini" + "_" + miniclip + ".mp4"
                new_dict[new_miniclip] = dict_video_actions[miniclip_key]

    with open('miniclip_actions.json', 'w') as fp:
        json.dump(new_dict, fp)


def main():
    df = pd.read_csv(path_visible_not_visible_actions_csv)
    # df = df.fillna(0)
    # visibile = df['Visible Actions']
    # visibile = [value for value in visibile if value != 0]
    # not_visibile = df['Not Visible Actions']
    # not_visibile = [value for value in not_visibile if value != 0]
    # miniclips = df['Video_name']
    # miniclips = set([value for value in miniclips if value != 0])

    # for file in miniclips:
    #     if len(file[:-4].split("_")) == 3:
    #         channel_playlist, video, miniclip = file[:-4].split("_")
    #         channel, playlist = channel_playlist.split("p")
    #         video = video[:-4]
    #         # print(channel,playlist, video, miniclip)
    #         # if int(channel) == 2 and int(playlist) == 1:
    #         if channel == '6' and playlist == '0':
    #             channel = '8'
    #             playlist = '1'
    #             new_file = channel + 'p' + playlist + "_" + video + "mini" + "_" + miniclip + ".mp4"
    #             print(new_file)
    #             os.rename("/local/oignat/miniclips_dataset/" + file, "/local/oignat/miniclips_dataset_new/" + new_file)
    #
    #            # os.rename("/local/oignat/miniclips_dataset/" + file, "/local/oignat/miniclips_dataset_new/" + file)

    # random_index_vis = random.sample(range(len(visibile)), 100)
    # random_index_not_vis = random.sample(range(len(not_visibile)), 100)
    #
    # for i in random_index_not_vis:
    #     print(not_visibile[i])




if __name__ == "__main__":
    # main()
    # clean_actions_file(True, False, path_visible_not_visible_actions_csv)
    # create_json_actions()
    # modify_name_visual_feat()
    # extract_c3d_from_inc3cd()
    # modify_name_yolo()
    # modify_context()
    # get_context_emb_all()
    get_pos_emb_all()
