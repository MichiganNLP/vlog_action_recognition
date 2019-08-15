from __future__ import print_function, absolute_import, unicode_literals, division

import csv
import random
from collections import OrderedDict

import pandas as pd
import nltk
import numpy as np
from keras_preprocessing.sequence import pad_sequences
from nltk import word_tokenize

import json

from sklearn import preprocessing
from tabulate import tabulate
from keras.preprocessing.text import Tokenizer

from amt.settings import PATH_visible_not_visible_actions_csv
from classify.elmo_embeddings import load_elmo_embedding
from classify.utils import reshape_3d_to_2d
from classify.visualization import print_action_balancing_stats, get_list_actions_for_label, get_nb_visible_not_visible, \
    print_nb_actions_miniclips_train_test_eval, measure_nb_unique_actions

import os
import glob
from shutil import copytree
import string
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


def load_embeddings():
    embeddings_index = dict()
    with open("data/glove.6B.50d.txt") as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    print('Loaded %s word vectors.' % len(embeddings_index))

    return embeddings_index


def chunks(l, n):
    n = max(1, n)
    chunk_list = []
    chunk_list += (l[i:i + n] for i in xrange(0, len(l), n))
    return chunk_list


def merge_chunks(list_chunks):
    merged = []
    for l in list_chunks:
        merged += l
    return merged


def get_word_embedding(embeddings_index, word):
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is None:
        return None
    else:
        word_embedding = np.asarray(embedding_vector)
        return word_embedding


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





def create_context_dict(dict_video_actions, path_context_data):
    reader = csv.reader(open(path_context_data, 'r'))
    headers = next(reader)
    dict_action_sentence = dict()
    dict_video_action_sentence = dict()
    for row in reader:
        [action, end_time, video_name, sentence, start_time] = row[1:]
        if video_name not in dict_action_sentence:
            dict_action_sentence[video_name] = []

        dict_action_sentence[video_name].append([action, sentence])

    dict_video_sentence = dict()
    for video_name in dict_video_actions.keys():
        video_name_in_file = "(" + video_name.split("_")[0] + ", " + video_name.split("_")[1].split("mini")[0] + ")"

        for index_visible_not_visible in [0, 1]:
            list_actions = dict_video_actions[video_name][index_visible_not_visible]
            if video_name_in_file in dict_action_sentence:
                # the sentence has also time info
                # strip  the time stamps, compare with both
                for [action, sentence] in dict_action_sentence[video_name_in_file]:

                    if list_actions.count(action) != 0:

                        if (video_name, action) not in dict_video_sentence:
                            dict_video_sentence[(video_name_in_file, action)] = [[], []]
                        if video_name not in dict_video_action_sentence:
                            dict_video_action_sentence[video_name] = [[], []]

                        no_time_sentence = nltk.re.sub('[0-9][0-9]:*', '', sentence)
                        dict_video_sentence[(video_name_in_file, action)][index_visible_not_visible].append(
                            no_time_sentence)
                        dict_video_action_sentence[(video_name)][index_visible_not_visible].append(
                            [action, no_time_sentence])

    return dict_video_sentence, dict_video_action_sentence


def get_data_sentence(dict_train_data):
    dict_context_data, _ = create_context_dict(dict_train_data)
    train_data_sentence = []

    for key in dict_train_data.keys():
        video_name_in_file = "(" + key.split("_")[0] + ", " + key.split("_")[1].split("mini")[0] + ")"

        for visible_action in dict_train_data[key][0]:
            if (video_name_in_file, visible_action) in dict_context_data.keys():
                sentence = dict_context_data[(video_name_in_file, visible_action)][0]
                if sentence == []:
                    train_data_sentence.append([""])
                else:
                    train_data_sentence.append(sentence)

            else:
                train_data_sentence.append([""])

        for non_visible_action in dict_train_data[key][1]:
            if (video_name_in_file, non_visible_action) in dict_context_data.keys():
                sentence = dict_context_data[(video_name_in_file, non_visible_action)][1]
                if sentence == []:
                    train_data_sentence.append([""])
                else:
                    train_data_sentence.append(sentence)
            else:
                train_data_sentence.append([""])
    return train_data_sentence


def create_action_embedding(embeddings_index, action, dimension_embedding):
    # no prev or next action: ned to distinguish between cases when action is not recognized
    if action == "":
        average_word_embedding = np.ones((1, dimension_embedding), dtype='float32') * 10
    else:
        list_words = word_tokenize(action)
        set_words_not_in_glove = set()
        nb_words = 0
        average_word_embedding = np.zeros((1, dimension_embedding), dtype='float32')
        for word in list_words:
            if word in set_words_not_in_glove:
                continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is None:
                set_words_not_in_glove.add(word)
                continue
            word_embedding = np.asarray(embedding_vector)
            average_word_embedding += word_embedding
            nb_words += 1
        if nb_words != 0:
            average_word_embedding = average_word_embedding / nb_words

        if (average_word_embedding == np.zeros((1,), dtype=np.float32)).all():
            # couldn't find any word of the action in the vocabulary -> initialize random
            average_word_embedding = np.random.rand(1, dimension_embedding).astype('float32')

    return average_word_embedding


def create_average_action_embedding(embeddings_index, list_actions):
    dimension_embedding = len(embeddings_index.get("example"))

    embedding_matrix_actions = np.zeros((len(list_actions), dimension_embedding))
    index = 0
    for action in list_actions:
        average_word_embedding = create_action_embedding(embeddings_index, action, dimension_embedding)
        embedding_matrix_actions[index] = average_word_embedding
        index += 1
    return embedding_matrix_actions


def BOW(train_data, list_word_in_vocab):
    vocab_size = len(list_word_in_vocab)
    embedding_matrix_actions_train = np.zeros((len(train_data), vocab_size))
    i = 0
    for action in train_data:
        action_embedding = np.zeros(vocab_size)
        list_words_in_action = nltk.word_tokenize(action)
        for index_word_in_vocab in range(0, len(list_word_in_vocab)):
            if list_word_in_vocab[index_word_in_vocab] in list_words_in_action:
                action_embedding[index_word_in_vocab] = 1
        embedding_matrix_actions_train[i] = action_embedding
        i += 1

    return embedding_matrix_actions_train


def process_batch_data(train_data, batch_size):
    list_data_chunks = chunks(train_data, batch_size)
    first_chunk_list = list_data_chunks[0]
    embedding_matrix_actions_train = load_elmo_embedding(first_chunk_list)
    for chhunk in list_data_chunks[1:]:
        embedding_matrix_actions_train_1 = load_elmo_embedding(chhunk)
        embedding_matrix_actions_train = np.concatenate(
            (embedding_matrix_actions_train, embedding_matrix_actions_train_1), axis=0)
    return embedding_matrix_actions_train


def preprocess_pos_embeddings(train_video, path_embedding):
    df_pos = pd.read_csv(path_embedding)
    dict_pos_embeddings = {}

    for index, row in df_pos.iterrows():
        video = row['video']
        if video in train_video:
            action = row['action']
            if 'my vegetables in water instead of oil' in action:
                action = 'sauteing my vegetables in water instead of oil'
            if ',' in row['pos_embedding'][1:-1]:
                pos_embed = np.asarray(row['pos_embedding'][1:-1].split(','))
            else:
                pos_embed = np.asarray(row['pos_embedding'][1:-1].split())
            #  TODO:  add label
            dict_pos_embeddings[(video, action)] = pos_embed

    return dict_pos_embeddings

def preprocess_context_embeddings(train_video, path_embedding):
    df_context = pd.read_csv(path_embedding)
    dict_context_embedding = {}
    for index, row in df_context.iterrows():
        video = row['video']
        if video in train_video:
            action = row['action']
            if ',' in row['right_context'][1:-1]:
                right_context = np.asarray([float(x) for x in row['right_context'][1:-1].split(',')])
            else:
                right_context = np.asarray([float(x) for x in row['right_context'][1:-1].split()])

            if ',' in row['left_context'][1:-1]:
                left_context = np.asarray([float(x) for x in row['left_context'][1:-1].split(',')])
            else:
                left_context = np.asarray([float(x) for x in row['left_context'][1:-1].split()])

            left_right_context = np.concatenate((left_context, right_context), axis=0)
            dict_context_embedding[(video, action)] = left_right_context

    return dict_context_embedding

def get_pos_embedding(train_data, dict_pos_embeddings):
    pos_embedding_size = 50
    nb_train_actions = len(train_data)
    embedding_pos_train = np.zeros((nb_train_actions, pos_embedding_size))
    index_train = 0

    for [video, action, label] in train_data:

        if (video, action) not in dict_pos_embeddings.keys():
            raise ValueError(str((video, action, label)) + ' not in dict_pos_embeddings!!')
        else:
            pos_embedding = dict_pos_embeddings[(video, action)]
            embedding_pos_train[index_train] = pos_embedding
            index_train += 1
    return embedding_pos_train


def get_context_embedding(train_data, dict_context_embeddings):
    context_embedding_size = 100
    nb_train_actions = len(train_data)
    embedding_context_train = np.zeros((nb_train_actions, context_embedding_size))
    index_train = 0

    json_dict_context_embeddings = {}
    for key in dict_context_embeddings:
        json_dict_context_embeddings[str(key)] = 0

    for [video, action, label] in train_data:

        if (video, action) not in dict_context_embeddings.keys():
            raise ValueError(str((video, action, label)) + 'not in dict_context_embeddings!!')
        else:
            context_embedding = dict_context_embeddings[(video, action)]
            embedding_context_train[index_train] = context_embedding
            index_train += 1
    return embedding_context_train


def create_visual_features_matrices(train_miniclips, type_feat, avg_or_concatenate):
    nb_frames = 61
    if type_feat[0] == 'inception' or type_feat == 'inception':
        print("Using inception")
        path_video_features = 'data/Video/Features/inception/'
        dimension_output = 2048
    elif type_feat[0] == 'inception + c3d' or type_feat == 'inception + c3d':
        print("Using inception + c3d")
        path_video_features = 'data/Video/Features/inception_c3d/'
        dimension_output = 6144
    elif type_feat[0] == 'c3d' or type_feat == 'c3d':
        print("Using c3d")
        path_video_features = 'data/Video/Features/c3d/'
        dimension_output = 4096
    else:
        print("Using default: inception + c3d")
        path_video_features = 'data/Video/Features/inception_c3d/'
        dimension_output = 6144

    index = 0

    if avg_or_concatenate == 'avg':
        matrix_visual_features = np.zeros(
            (len(train_miniclips), dimension_output))  # nb actions = nb miniclips
        padded_video_features = np.zeros(dimension_output) # no need to pad if avg

    else:
        matrix_visual_features = np.zeros(
            (len(train_miniclips), nb_frames, dimension_output))  # nb actions = nb miniclips
        padded_video_features = np.zeros((nb_frames, dimension_output))


    for miniclip_id in train_miniclips:
        video_features = np.load(str(path_video_features + miniclip_id.replace('.mp4', '') + '.npy'))
        # video_features = video_features[1:-1, :]

        if avg_or_concatenate == 'avg':
            avg_video_features = np.mean(video_features, axis=0)
            padded_video_features = avg_video_features
            # L2 normalize: the square elems sum to 1
            padded_video_features = preprocessing.normalize(np.asarray(padded_video_features).reshape(1,-1), norm='l2')

        else:
            for i in range(dimension_output):
                padded_video_features[:, i] = np.array(
                    list(video_features[:, i]) + (nb_frames - video_features.shape[0]) * [0])

        matrix_visual_features[index] = padded_video_features
        index += 1

    return matrix_visual_features


def get_visual_features(train_miniclips, test_miniclips, val_miniclips, type_feat, avg_or_concatenate):
    visual_feat_train = create_visual_features_matrices(train_miniclips, type_feat, avg_or_concatenate)
    visual_feat_test = create_visual_features_matrices(test_miniclips, type_feat, avg_or_concatenate)
    visual_feat_val = create_visual_features_matrices(val_miniclips, type_feat, avg_or_concatenate)

    return visual_feat_train, visual_feat_test, visual_feat_val


def get_matrix_word_embedding(embeddings_index, train_data, test_data, val_data):
    [train_actions, test_actions, val_actions], _, _ = process_data(train_data, test_data, val_data)

    all_actions = train_actions + test_actions + val_actions

    t = Tokenizer()
    t.fit_on_texts(all_actions)
    vocab_size = len(t.word_index) + 1  # nb of unique words

    max_length_word = max(all_actions, key=len)
    max_length = len(max_length_word.split(" "))

    # create a weight matrix for words in all docs
    embedding_words_all = np.zeros((vocab_size, 50))

    for word, i in t.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_words_all[i] = embedding_vector

    return embedding_words_all, max_length


def get_action_embedding(embeddings_index, train_actions, test_actions, val_actions):
    embedding_actions_train = create_average_action_embedding(embeddings_index, train_actions)
    embedding_actions_test = create_average_action_embedding(embeddings_index, test_actions)
    embedding_actions_val = create_average_action_embedding(embeddings_index, val_actions)

    return embedding_actions_train, embedding_actions_test, embedding_actions_val


def add_pos_embed(train_data, test_data, val_data, embedding_matrix_actions_train, embedding_matrix_actions_test,
                  embedding_matrix_actions_val):
    _, _, [train_video, test_video, val_video] = process_data(train_data, test_data, val_data)

    dict_pos_train = preprocess_pos_embeddings(train_video, path_embedding='data/Embeddings/pos_embeddings.csv')
    dict_pos_test = preprocess_pos_embeddings(test_video, path_embedding='data/Embeddings/pos_embeddings.csv')
    dict_pos_val = preprocess_pos_embeddings(val_video, path_embedding='data/Embeddings/pos_embeddings.csv')

    embedding_pos_train = get_pos_embedding(train_data, dict_pos_train)
    embedding_pos_test = get_pos_embedding(test_data, dict_pos_test)
    embedding_pos_val = get_pos_embedding(val_data, dict_pos_val)

    if embedding_matrix_actions_train is None:
        return embedding_pos_train, embedding_pos_test, embedding_pos_val

    if len(embedding_matrix_actions_train.shape) == 3:
        embedding_matrix_actions_train = reshape_3d_to_2d(embedding_matrix_actions_train)
        embedding_matrix_actions_test = reshape_3d_to_2d(embedding_matrix_actions_test)
        embedding_matrix_actions_val = reshape_3d_to_2d(embedding_matrix_actions_val)

    embedding_pos_concat_train = np.concatenate((embedding_matrix_actions_train, embedding_pos_train),
                                                axis=1)
    embedding_pos_concat_test = np.concatenate((embedding_matrix_actions_test, embedding_pos_test),
                                               axis=1)
    embedding_pos_concat_val = np.concatenate((embedding_matrix_actions_val, embedding_pos_val),
                                              axis=1)

    return embedding_pos_concat_train, embedding_pos_concat_test, embedding_pos_concat_val


def add_context_embed(train_data, test_data, val_data, embedding_matrix_actions_train, embedding_matrix_actions_test,
                      embedding_matrix_actions_val):
    _, _, [train_video, test_video, val_video] = process_data(train_data, test_data, val_data)
    dict_context_train = preprocess_context_embeddings(train_video,
                                                       path_embedding='data/Embeddings/context_embeddings.csv')
    dict_context_test = preprocess_context_embeddings(test_video,
                                                      path_embedding='data/Embeddings/context_embeddings.csv')
    dict_context_val = preprocess_context_embeddings(val_video, path_embedding='data/Embeddings/context_embeddings.csv')

    embedding_context_train = get_context_embedding(train_data, dict_context_train)
    embedding_context_test = get_context_embedding(test_data, dict_context_test)
    embedding_context_val = get_context_embedding(val_data, dict_context_val)

    if embedding_matrix_actions_train is None:
        return embedding_context_train, embedding_context_test, embedding_context_val

    if len(embedding_matrix_actions_train.shape) == 3:
        embedding_matrix_actions_train = reshape_3d_to_2d(embedding_matrix_actions_train)
        embedding_matrix_actions_test = reshape_3d_to_2d(embedding_matrix_actions_test)
        embedding_matrix_actions_val = reshape_3d_to_2d(embedding_matrix_actions_val)

    embedding_context_concat_train = np.concatenate((embedding_matrix_actions_train, embedding_context_train),
                                                    axis=1)
    embedding_context_concat_test = np.concatenate((embedding_matrix_actions_test, embedding_context_test),
                                                   axis=1)
    embedding_context_concat_val = np.concatenate((embedding_matrix_actions_val, embedding_context_val),
                                                  axis=1)
    return embedding_context_concat_train, embedding_context_concat_test, embedding_context_concat_val


def pad_actions(train_actions, test_actions, val_actions):
    all_actions = train_actions + test_actions + val_actions
    t = Tokenizer()
    t.fit_on_texts(all_actions)
    # t.fit_on_texts(train_actions)

    max_length_word = max(all_actions, key=len)
    max_length = len(max_length_word.split(" "))
    # process data for LSTM
    encoded_docs_train = t.texts_to_sequences(train_actions)
    encoded_docs_test = t.texts_to_sequences(test_actions)
    encoded_docs_val = t.texts_to_sequences(val_actions)
    # pad documents to a max length of the largest string in the list of actions
    x_train = pad_sequences(encoded_docs_train, maxlen=max_length, padding='post')
    x_test = pad_sequences(encoded_docs_test, maxlen=max_length, padding='post')
    x_val = pad_sequences(encoded_docs_val, maxlen=max_length, padding='post')

    return x_train, x_test, x_val

def get_concreteness_score(list_actions, type):
    with open('data/dict_action_pos_concreteness.json', 'r') as fp:
        dict_concreteness = json.load(fp)

    list_scores = []
    # 98 % coverage
    for action in list_actions:
        if action in dict_concreteness:
            if type == 'all':
                scores = [l[2] for l in dict_concreteness[action]]
            elif type == 'noun + vb':
                scores = [l[2] for l in dict_concreteness[action] if ('VB' in l[1] or 'NN' in l[1])]
            elif type == 'vb':
                scores = [l[2] for l in dict_concreteness[action] if 'VB' in l[1]]
            elif type == 'noun':
                scores = [l[2] for l in dict_concreteness[action] if 'NN' in l[1]]
            else:
                raise ValueError("Wrong type in concreteness dict_type")
        else:
            scores = []

        if scores:
            action_concreteness_score = max(scores)
        else:
            action_concreteness_score = 0

        list_scores.append(action_concreteness_score)

    scores = np.array(list_scores).reshape(-1, 1)
    return scores


def add_concreteness_score(train_actions, test_actions, val_actions, embedding_matrix_actions_train,
                           embedding_matrix_actions_test, embedding_matrix_actions_val, type):

    scores_train = get_concreteness_score(train_actions, type)
    scores_test = get_concreteness_score(test_actions, type)
    scores_val = get_concreteness_score(val_actions, type)

    if embedding_matrix_actions_train is None:
        return scores_train, scores_test, scores_val

    if len(embedding_matrix_actions_train.shape) == 3:
        embedding_matrix_actions_train = reshape_3d_to_2d(embedding_matrix_actions_train)
        embedding_matrix_actions_test = reshape_3d_to_2d(embedding_matrix_actions_test)
        embedding_matrix_actions_val = reshape_3d_to_2d(embedding_matrix_actions_val)

    embedding_concreteness_concat_train = np.concatenate((embedding_matrix_actions_train, scores_train),
                                                         axis=1)
    embedding_concreteness_concat_test = np.concatenate((embedding_matrix_actions_test, scores_test),
                                                        axis=1)
    embedding_concreteness_concat_val = np.concatenate((embedding_matrix_actions_val, scores_val),
                                                       axis=1)

    return embedding_concreteness_concat_train, embedding_concreteness_concat_test, embedding_concreteness_concat_val


def get_embedding_next_action(embeddings_index, train_data):
    dict_prev_next_action = get_dict_prev_next_actions()
    list_next_actions = []
    for (video, action, label) in train_data:
        [_, next_action_label] = dict_prev_next_action[(video, action, label)]
        next_action, next_label = next_action_label
        list_next_actions.append(next_action)
    embedding_next_actions = create_average_action_embedding(embeddings_index, list_next_actions)
    return embedding_next_actions


def add_next_action(embeddings_index, train_data, test_data, val_data, embedding_matrix_actions_train,
                    embedding_matrix_actions_test, embedding_matrix_actions_val):
    embedding_next_actions_train = get_embedding_next_action(embeddings_index, train_data)
    embedding_next_actions_test = get_embedding_next_action(embeddings_index, test_data)
    embedding_next_actions_val = get_embedding_next_action(embeddings_index, val_data)

    if embedding_matrix_actions_train is None:
        return embedding_next_actions_train, embedding_next_actions_test, embedding_next_actions_val

    if len(embedding_matrix_actions_train.shape) == 3:
        embedding_matrix_actions_train = reshape_3d_to_2d(embedding_matrix_actions_train)
        embedding_matrix_actions_test = reshape_3d_to_2d(embedding_matrix_actions_test)
        embedding_matrix_actions_val = reshape_3d_to_2d(embedding_matrix_actions_val)

    embedding_next_concat_train = np.concatenate((embedding_matrix_actions_train, embedding_next_actions_train),
                                                 axis=1)
    embedding_next_concat_test = np.concatenate((embedding_matrix_actions_test, embedding_next_actions_test),
                                                axis=1)
    embedding_next_concat_val = np.concatenate((embedding_matrix_actions_val, embedding_next_actions_val),
                                               axis=1)

    return embedding_next_concat_train, embedding_next_concat_test, embedding_next_concat_val


def get_embedding_prev_action(embeddings_index, train_data):
    dict_prev_next_action = get_dict_prev_next_actions()
    list_prev_actions = []
    for (video, action, label) in train_data:
        [prev_action_label, _] = dict_prev_next_action[(video, action, label)]
        prev_action, prev_label = prev_action_label
        list_prev_actions.append(prev_action)
    embedding_prev_actions = create_average_action_embedding(embeddings_index, list_prev_actions)
    return embedding_prev_actions


def add_prev_action(embeddings_index, train_data, test_data, val_data, embedding_matrix_actions_train,
                    embedding_matrix_actions_test, embedding_matrix_actions_val):
    embedding_prev_actions_train = get_embedding_prev_action(embeddings_index, train_data)
    embedding_prev_actions_test = get_embedding_prev_action(embeddings_index, test_data)
    embedding_prev_actions_val = get_embedding_prev_action(embeddings_index, val_data)

    if embedding_matrix_actions_train is None:
        return embedding_prev_actions_train, embedding_prev_actions_test, embedding_prev_actions_val

    if len(embedding_matrix_actions_train.shape) == 3:
        embedding_matrix_actions_train = reshape_3d_to_2d(embedding_matrix_actions_train)
        embedding_matrix_actions_test = reshape_3d_to_2d(embedding_matrix_actions_test)
        embedding_matrix_actions_val = reshape_3d_to_2d(embedding_matrix_actions_val)

    embedding_prev_concat_train = np.concatenate((embedding_prev_actions_train, embedding_matrix_actions_train),
                                                 axis=1)
    embedding_prev_concat_test = np.concatenate((embedding_prev_actions_test, embedding_matrix_actions_test),
                                                axis=1)
    embedding_prev_concat_val = np.concatenate((embedding_prev_actions_val, embedding_matrix_actions_val),
                                               axis=1)

    return embedding_prev_concat_train, embedding_prev_concat_test, embedding_prev_concat_val


def add_visual_features(train_data, test_data, val_data, x_train, x_test,
                        x_val, type_feat):
    [train_actions, test_actions, val_actions], _, [train_miniclips, test_miniclips, val_miniclips] = process_data(train_data, test_data, val_data)

    video_data_train, video_data_test, video_data_val = get_visual_features(train_miniclips, test_miniclips,
                                                                            val_miniclips, type_feat,
                                                                            avg_or_concatenate='avg')
    if x_train is not None:
        visual_concat_train = np.concatenate((x_train, video_data_train), axis=1)
        visual_concat_test = np.concatenate((x_test, video_data_test), axis=1)
        visual_concat_val = np.concatenate((x_val, video_data_val), axis=1)
    else:
        visual_concat_train = video_data_train
        visual_concat_test = video_data_test
        visual_concat_val = video_data_val

    print("Visual feature: ")
    for i in range(len(video_data_val)):
        print(val_actions[i], video_data_val[i])

    return visual_concat_train, visual_concat_test, visual_concat_val


def get_embeddings_by_type(type_embedding, add_extra,
                           embeddings_index, train_data,
                           test_data, val_data, type_concreteness):
    [train_actions, test_actions, val_actions], _, _ = process_data(train_data, test_data, val_data)

    if type_embedding == "action":
        x_train, x_test, x_val = get_action_embedding(embeddings_index, train_actions, test_actions, val_actions)
    elif type_embedding == "padding":
        x_train, x_test, x_val = pad_actions(train_actions, test_actions, val_actions)
    else:
        print("No embedding to concatenate to. Will store only extra embeddings")
        x_train, x_test, x_val = [None, None, None]

    if "pos" in add_extra:
        print("Add pos")
        x_train, x_test, x_val = add_pos_embed(train_data, test_data, val_data, x_train, x_test, x_val)
    if "context" in add_extra:
        print("Add context")
        x_train, x_test, x_val = add_context_embed(train_data, test_data, val_data, x_train, x_test, x_val)
    if "concreteness" in add_extra:
        print("Add concreteness: " + type_concreteness + " max score")
        x_train, x_test, x_val = add_concreteness_score(train_actions, test_actions, val_actions, x_train, x_test,
                                                        x_val, type_concreteness)
    if "prev-next-action" in add_extra:
        print("Add prev-next action")
        x_train, x_test, x_val = add_prev_action(embeddings_index, train_data, test_data, val_data, x_train, x_test,
                                                 x_val)
        x_train, x_test, x_val = add_next_action(embeddings_index, train_data, test_data, val_data, x_train, x_test,
                                                 x_val)

    if "visual-c3d-inception" in add_extra:
        print("Add visual-c3d-inception")
        x_train, x_test, x_val = add_visual_features(train_data, test_data, val_data, x_train, x_test,
                                                     x_val, type_feat='inception')


    return x_train, x_test, x_val


def get_dict_prev_next_actions(path_visible_not_visible_actions_csv=PATH_visible_not_visible_actions_csv):
    df_data = pd.read_csv(path_visible_not_visible_actions_csv)
    dict_miniclip_action = OrderedDict()
    for index, row in df_data.iterrows():
        miniclip = row['Video_name']
        if pd.isnull(row['Visible Actions']) and pd.isnull(row['Not Visible Actions']):
            continue
        elif pd.isnull(row['Visible Actions']):
            action = row['Not Visible Actions']
            if type(action) is str:
                action = action.encode('utf8').lower()
            label = 1
        else:
            action = row['Visible Actions']
            if type(action) is str:
                action = action.encode('utf8').lower()
            label = 0

        if miniclip not in dict_miniclip_action.keys():
            dict_miniclip_action[miniclip] = []
        dict_miniclip_action[miniclip].append([action, label])

    dict_prev_next_action = OrderedDict()
    for video in dict_miniclip_action.keys():
        list_action_labels = dict_miniclip_action[video]

        # if only action in miniclip:
        if len(list_action_labels) == 1:
            [action, label] = list_action_labels[0]
            prev_action_label = ["", -1]
            next_action_label = ["", -1]
            dict_prev_next_action[(video, action, label)] = [prev_action_label, next_action_label]
        else:
            # first action in the miniclip
            [action, label] = list_action_labels[0]
            prev_action_label = ["", -1]
            next_action_label = list_action_labels[1]
            dict_prev_next_action[(video, action, label)] = [prev_action_label, next_action_label]

            for index in range(1, len(list_action_labels) - 1):
                action, label = list_action_labels[index]
                prev_action_label = list_action_labels[index - 1]
                next_action_label = list_action_labels[index + 1]
                dict_prev_next_action[(video, action, label)] = [prev_action_label, next_action_label]

            # last action in the miniclip
            [action, label] = list_action_labels[-1]
            prev_action_label = list_action_labels[-2]
            next_action_label = ["", -1]
            dict_prev_next_action[(video, action, label)] = [prev_action_label, next_action_label]

    return dict_prev_next_action


def split_data_after_video_from_csv(path_visible_not_visible_actions_csv=PATH_visible_not_visible_actions_csv):

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

    return dict_video_actions


def balance_data(balance, dict_video_actions, dict_train_data):
    nb_visible_actions, nb_not_visible_actions = get_nb_visible_not_visible(dict_train_data)

    if nb_not_visible_actions >= nb_visible_actions:
        ratio_visible_not_visible = int(nb_not_visible_actions / nb_visible_actions)
    else:
        ratio_visible_not_visible = int(nb_visible_actions / nb_not_visible_actions)

    if balance == "upsample":
        # Upsample data
        for video_name in dict_train_data.keys():
            list_visible_actions = get_list_actions_for_label(dict_train_data, video_name, 0)
            for elem in list_visible_actions:
                dict_video_actions[video_name].append([elem, 0])

        nb_visible_actions, nb_not_visible_actions = get_nb_visible_not_visible(dict_train_data)
        diff_nb_actions = abs(nb_not_visible_actions - nb_visible_actions)
        while diff_nb_actions:
            # this makes the # actions to vary in Train, Test Eval after each run
            # run it once and save the list
            random_video_name = random.choice(list(dict_train_data))
            list_visible_actions = get_list_actions_for_label(dict_train_data, random_video_name, 0)
            if list_visible_actions:
                dict_video_actions[random_video_name].append([list_visible_actions[0], 0])
                diff_nb_actions -= 1


    elif balance == "downsample":
        # Downsample data --> delete the non-visible actions
        for video_name in dict_train_data.keys():
            list_not_visible_actions = get_list_actions_for_label(dict_video_actions, video_name, 1)
            index = 0
            list_all_actions = dict_video_actions[video_name]
            for elem in list_not_visible_actions:
                if index % ratio_visible_not_visible == 0:
                    list_all_actions.remove([elem, 1])
                index += 1
            dict_video_actions[video_name] = list_all_actions

        nb_visible_actions, nb_not_visible_actions = get_nb_visible_not_visible(dict_train_data)
        diff_nb_actions = abs(nb_not_visible_actions - nb_visible_actions)
        while (diff_nb_actions):

            # this makes the # actions to vary in Train, Test Eval after each run
            # run it once and save the list
            random_video_name = random.choice(list(dict_train_data))
            list_not_visible_actions = get_list_actions_for_label(dict_video_actions, random_video_name, 1)
            if list_not_visible_actions:
                list_all_actions = dict_video_actions[random_video_name]
                list_all_actions.remove([list_not_visible_actions[0], 1])
                diff_nb_actions -= 1

    return dict_video_actions, dict_train_data


def split_train_test_val_data(dict_video_actions, channel_test, channel_val):
    dict_train_data = OrderedDict()
    dict_test_data = OrderedDict()
    dict_val_data = OrderedDict()

    for channel in range(1, 11):
        if channel == channel_test or channel == channel_val:
            continue
        for key in dict_video_actions.keys():
            # if str(channel) + "p" in key or 'p' not in key[:-3]:
            if str(channel) + "p" in key:
                dict_train_data[key] = dict_video_actions[key]

    for channel in range(channel_val, channel_val + 1):
        for key in dict_video_actions.keys():
            if str(channel) + "p" in key:
                dict_val_data[key] = dict_video_actions[key]

    for channel in range(channel_test, channel_test + 1):
        for key in dict_video_actions.keys():
            if str(channel) + "p" in key:
                dict_test_data[key] = dict_video_actions[key]

    return dict_train_data, dict_test_data, dict_val_data


# lists triples of (miniclip, action, label)
def create_data(dict_train_data, dict_test_data, dict_val_data):
    train_data = []
    test_data = []
    val_data = []
    for miniclip in dict_train_data.keys():
        for [action, label] in dict_train_data[miniclip]:
            train_data.append((miniclip, action, label))

    for miniclip in dict_test_data.keys():
        for [action, label] in dict_test_data[miniclip]:
            test_data.append((miniclip, action, label))

    for miniclip in dict_val_data.keys():
        for [action, label] in dict_val_data[miniclip]:
            val_data.append((miniclip, action, label))

    return train_data, test_data, val_data


def process_data(train_data, test_data, val_data):
    train_labels = [label for (video, action, label) in train_data]
    test_labels = [label for (video, action, label) in test_data]
    val_labels = [label for (video, action, label) in val_data]

    train_actions = [action for (video, action, label) in train_data]
    test_actions = [action for (video, action, label) in test_data]
    val_actions = [action for (video, action, label) in val_data]

    train_video = [video for (video, action, label) in train_data]
    test_video = [video for (video, action, label) in test_data]
    val_video = [video for (video, action, label) in val_data]

    return [train_actions, test_actions, val_actions], [train_labels, test_labels, val_labels], [train_video,
                                                                                                 test_video, val_video]


def get_data(balance, channel_test, channel_val):

    # dict_video_actions = split_data_after_video_from_csv()
    with open("data/miniclip_actions.json") as f:
        dict_video_actions = json.loads(f.read())

    dict_train_data, dict_test_data, dict_val_data = split_train_test_val_data(dict_video_actions, channel_test,
                                                                               channel_val)

    # balance only train data
    if balance:
        dict_video_actions, dict_train_data = balance_data(balance, dict_video_actions, dict_train_data)



    # print_nb_actions_miniclips_train_test_eval(dict_train_data, dict_test_data, dict_val_data)
    # measure_nb_unique_actions(dict_video_actions)

    train_data, test_data, val_data = create_data(dict_train_data, dict_test_data, dict_val_data)

    print_action_balancing_stats(balance, 0, dict_video_actions, dict_train_data, dict_test_data,
                                 dict_val_data, test_data)

    print_nb_actions_miniclips_train_test_eval(dict_train_data, dict_test_data, dict_val_data)

    return dict_video_actions, dict_train_data, dict_test_data, dict_val_data, train_data, test_data, val_data
