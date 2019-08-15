from __future__ import print_function, absolute_import, unicode_literals, division

import json
import os
from tqdm import tqdm

from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score

from classify.preprocess import process_data
from nltk.tag import StanfordPOSTagger
from nltk import PorterStemmer

stemmer = PorterStemmer()

os.environ["CLASSPATH"] = "stanford-postagger-full-2018-10-16/"
os.environ["STANFORD_MODELS"] = "stanford-postagger-full-2018-10-16/models/"
st = StanfordPOSTagger('english-bidirectional-distsim.tagger')


def get_all_words_concreteness_scores(concreteness_words_txt):
    lines = [line.rstrip('\n') for line in open(concreteness_words_txt, 'r')]
    dict_concreteness = dict()
    for line in lines[1:]:

        word, bigram, concreteness_score, concreteness_score_sd, Unknown, Total, Percent_known, SUBTLEX, Dom_pos = line.split(
            "\t")
        # TODO: when bigram = 1
        if str(bigram) == '0':
            dict_concreteness[word] = float(concreteness_score)

    with open('data/dict_all_concreteness.json', 'w+') as fp:
        json.dump(dict_concreteness, fp)


def save_all_action_pos_concreteness_scores(all_actions):
    with open('data/dict_all_concreteness.json', 'r') as fp:
        dict_concreteness = json.load(fp)

    dict_action_pos_concreteness = {}
    all_actions = set(all_actions)

    for action in tqdm(all_actions):
        action_words = action.split()

        pos_words = st.tag(action_words)
        dict_pos = {}
        for word in action_words:
            stemmed_word = stemmer.stem(word)
            upper_word = word.upper()
            upper_stemmed_word = stemmed_word.upper()
            if word.isdigit():
                score = 3.7
            elif word in dict_concreteness.keys():
                score = dict_concreteness[word]
            elif stemmed_word in dict_concreteness.keys():
                score = dict_concreteness[stemmed_word]
            elif upper_word in dict_concreteness.keys():
                score = dict_concreteness[upper_word]
            elif upper_stemmed_word in dict_concreteness.keys():
                score = dict_concreteness[upper_stemmed_word]

            else:
                for key in dict_concreteness.keys():
                    if word == stemmer.stem(key):
                        score = dict_concreteness[key]
                    elif stemmed_word == stemmer.stem(key):
                        score = dict_concreteness[key]
                    elif upper_word == stemmer.stem(key):
                        score = dict_concreteness[key]
                    elif upper_stemmed_word in stemmer.stem(key):
                        score = dict_concreteness[key]
                    else:
                        score = 0

            pos = [l[1] for l in pos_words if l[0] == word][0]
            if pos in dict_pos.keys():
                dict_pos[pos].append([word, float(score)])
            else:
                dict_pos[pos] = [[word, float(score)]]
        dict_action_pos_concreteness[action] = dict_pos

    with open('data/dict_action_pos_concreteness.json', 'w+') as fp:
        json.dump(dict_action_pos_concreteness, fp)


def save_all_action_concreteness_scores(all_actions):
    with open('data/dict_all_concreteness.json', 'r') as fp:
        dict_concreteness = json.load(fp)

    dict_action_concreteness = {}
    all_actions = set(all_actions)

    for action in tqdm(all_actions):
        action_words = action.split()
        list_scores = []
        for word in action_words:
            stemmed_word = stemmer.stem(word)
            upper_word = word.upper()
            upper_stemmed_word = stemmed_word.upper()
            if word.isdigit():
                score = 3.7
            elif word in dict_concreteness.keys():
                score = dict_concreteness[word]
            elif stemmed_word in dict_concreteness.keys():
                score = dict_concreteness[stemmed_word]
            elif upper_word in dict_concreteness.keys():
                score = dict_concreteness[upper_word]
            elif upper_stemmed_word in dict_concreteness.keys():
                score = dict_concreteness[upper_stemmed_word]

            else:
                for key in dict_concreteness.keys():
                    if word == stemmer.stem(key):
                        score = dict_concreteness[key]
                    elif stemmed_word == stemmer.stem(key):
                        score = dict_concreteness[key]
                    elif upper_word == stemmer.stem(key):
                        score = dict_concreteness[key]
                    elif upper_stemmed_word in stemmer.stem(key):
                        score = dict_concreteness[key]
                    else:
                        score = 0

            list_scores.append([score, word])
        dict_action_concreteness[action] = list_scores

    with open('data/dict_action_concreteness.json', 'w+') as fp:
        json.dump(dict_action_concreteness, fp)

def save_all_action_pos_concreteness_scores_1():
    with open('data/dict_action_concreteness.json', 'r') as fp:
        dict_concreteness = json.load(fp)

    dict_action_pos_concreteness = {}

    for action in tqdm(dict_concreteness.keys()):
        dict_action_pos_concreteness[action] = []
        action_words = action.split()

        pos_words = st.tag(action_words)
        for word in action_words:
            pos = [l[1] for l in pos_words if l[0] == word][0]
            score = [l[0] for l in dict_concreteness[action] if l[1] == word][0]
            dict_action_pos_concreteness[action].append([word, pos, float(score)])

    with open('data/dict_action_pos_concreteness.json', 'w+') as fp:
        json.dump(dict_action_pos_concreteness, fp)



def cluster_after_concreteness(type, train_data, test_data, val_data):
    print("# Running concreteness score:")

    [train_actions, test_actions, val_actions], [train_labels, test_labels, val_labels], _ = process_data(train_data, test_data, val_data)
    all_actions = train_actions + test_actions + val_actions

    # save_all_action_pos_concreteness_scores(test_actions)
    # save_all_action_concreteness_scores(test_actions)

    with open('data/dict_action_pos_concreteness.json', 'r') as fp:
        dict_concreteness = json.load(fp)

    # finetune on val & train data to get the threshold
    #TODO also for average (not only max)
    dict_type_threshold = {'noun + vb': 4.7,
                           'noun': 4.7,
                           'vb': 3.5,
                           'all': 4.7
                           }

    # test on test data
    concreteness_step = dict_type_threshold[type]
    concrete_labels = []
    gt_labels = test_labels

    for action in test_actions:
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

            if scores:
                action_concreteness_score = max(scores)
            else:
                action_concreteness_score = 0

            if action_concreteness_score >= concreteness_step:
                concrete_labels.append(0)
            else:
                concrete_labels.append(1)

    accuracy = accuracy_score(concrete_labels, gt_labels)
    f1 = f1_score(gt_labels, concrete_labels)
    recall = recall_score(gt_labels, concrete_labels)
    precision = precision_score(gt_labels, concrete_labels)

    return [0, 0, accuracy, recall, precision, f1], concrete_labels
