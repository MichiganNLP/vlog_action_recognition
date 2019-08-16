from __future__ import print_function, absolute_import, unicode_literals, division

from collections import OrderedDict, Counter

import pandas as pd
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt

# Adapted from https://gist.github.com/ShinNoNoir/4749548
from amt.detect_spam import get_compromised_hits


def fleiss_kappa(ratings, n):
    '''
    Computes the Fleiss' kappa measure for assessing the reliability of
    agreement between a fixed number n of raters when assigning categorical
    ratings to a number of items.

    Args:
        ratings: a list of (item, category)-ratings
        n: number of raters
        k: number of categories
    Returns:
        the Fleiss' kappa score

    See also:
        http://en.wikipedia.org/wiki/Fleiss'_kappa
    '''
    items = set()
    categories = set()
    n_ij = {}

    for i, c in ratings:
        items.add(i)
        categories.add(c)
        n_ij[(i, c)] = n_ij.get((i, c), 0) + 1

    N = len(items)

    p_j = dict(((c, sum(n_ij.get((i, c), 0) for i in items) / (1.0 * n * N)) for c in categories))
    P_i = dict(((i, (sum(n_ij.get((i, c), 0) ** 2 for c in categories) - n) / (n * (n - 1.0))) for i in items))

    P_bar = sum(P_i.values()) / (1.0 * N)
    P_e_bar = sum(value ** 2 for value in p_j.values())

    kappa = (P_bar - P_e_bar) / (1 - P_e_bar)

    return kappa


def plot_agreement(x, threshold):
    plt.hist(x, normed=True, bins=30)
    plt.xlabel('Agreement score')
    plt.savefig('data/plots/' + str(threshold) + '_agreement_score.png')  # save the figure to file
    plt.close()

def plot_threshold_agreement(threshold_list, agreement_list):
    plt.style.use('seaborn-whitegrid')
    avg_agreement = [i[0] for i in agreement_list]
    overall_agreement = [i[1] for i in agreement_list]

    plt.plot(threshold_list, avg_agreement, 'o',  '-ok', color='black', label='avg agreement')
    plt.plot(threshold_list, overall_agreement, '*',  '-ok', color='pink', label='overall agreement')
    plt.legend(numpoints=1)
    plt.xlabel('spam accuracy threshold')
    plt.ylabel('agreement scores')
    plt.savefig('data/plots/spam_agreement_score.png')  # save the figure to file
    plt.close()




def agreement_per_task(hit_id, video_name):
    return


def agreement_total(dict_hit_agreement):
    # average of per hit agreements
    values = [score for score in dict_hit_agreement.values()]
    print("For avg total agreement ( erase hits with >=2 spammer) -> number of hits is {0}".format(len(dict_hit_agreement.keys())))

    return sum(values) / len(values)


def compute_ratings(df_worker_1, df_worker_2, df_worker_3=None):
    list_ratings = []
    for i in range(0, len(df_worker_1)):
        if df_worker_3 == None:
            list_results = [df_worker_1[i], df_worker_2[i]]
        else:
            list_results = [df_worker_1[i], df_worker_2[i], df_worker_3[i]]
        c = Counter(list_results)
        list_ratings += [(i, '0')] * c[0] + [(i, '1')] * c[1]
    return list_ratings


def agreement_overall(path_after_spam_filter_csv, do_1_eq_2, potential_spammers):

    df = pd.read_csv(path_after_spam_filter_csv)
    if do_1_eq_2:
        df['Worker_1'] = df['Worker_1'].replace(2, 1)
        df['Worker_2'] = df['Worker_2'].replace(2, 1)
        df['Worker_3'] = df['Worker_3'].replace(2, 1)

    df_hit = df['HIT_nb']
    set_hit_ids = set(df_hit)
    list_df_worker_1 = []
    list_df_worker_2 = []
    list_df_worker_3 = []
    nb_hits = 0
    for hit_id in set_hit_ids:

        if potential_spammers:
            if hit_id in potential_spammers.keys() and potential_spammers[hit_id] != set():
                continue

        df_per_hit = (df.loc[df_hit == hit_id])
        df_worker_1 = df_per_hit['Worker_1']
        df_worker_2 = df_per_hit['Worker_2']
        df_worker_3 = df_per_hit['Worker_3']

        # if potential_spammers != None:

        if not df_worker_3[df_worker_3.isin(['-1'])].empty:
            continue
        elif not df_worker_2[df_worker_2.isin(['-1'])].empty:
            continue
        elif not df_worker_1[df_worker_1.isin(['-1'])].empty:
            continue
        else:
            list_df_worker_1 += df_worker_1.values.tolist()
            list_df_worker_2 += df_worker_2.values.tolist()
            list_df_worker_3 += df_worker_3.values.tolist()
            nb_hits += 1

    ratings = compute_ratings(list_df_worker_1, list_df_worker_2,
                              list_df_worker_3)
    print("For overall agreement ( erase hits with >=1 spammer) -> number of hits is {0}".format(nb_hits))

    score = fleiss_kappa(ratings, 3)

    return score


def agreement_per_hit(path_after_spam_filter_csv, do_1_eq_2, do_cohen):
    # return ordered dictionary with (hit_id, agreement)

    dict_hit_agreement = {}
    df = pd.read_csv(path_after_spam_filter_csv)
    if do_1_eq_2:
        df['Worker_1'] = df['Worker_1'].replace(2, 1)
        df['Worker_2'] = df['Worker_2'].replace(2, 1)
        df['Worker_3'] = df['Worker_3'].replace(2, 1)

    df_hit = df['HIT_nb']
    set_hit_ids = set(df_hit)
    if do_cohen:
        print(" ---------Doing Cohen agreement for 2 workers -------")
    else:
        print(" ---------Doing Fleiss agreement for 2 workers -------")

    for hit_id in set_hit_ids:
        df_per_hit = (df.loc[df_hit == hit_id])
        df_worker_1 = df_per_hit['Worker_1']
        df_worker_2 = df_per_hit['Worker_2']
        df_worker_3 = df_per_hit['Worker_3']

        if do_cohen:
            if not df_worker_3[df_worker_3.isin(['-1'])].empty:
                score = cohen_kappa_score(df_worker_1.values.tolist(), df_worker_2.values.tolist())
            elif not df_worker_2[df_worker_2.isin(['-1'])].empty:
                score = cohen_kappa_score(df_worker_1.values.tolist(), df_worker_3.values.tolist())
            elif not df_worker_1[df_worker_1.isin(['-1'])].empty:
                score = cohen_kappa_score(df_worker_3.values.tolist(), df_worker_2.values.tolist())
            else:
                ratings = compute_ratings(df_worker_1.values.tolist(), df_worker_2.values.tolist(),
                                          df_worker_3.values.tolist())

                score = fleiss_kappa(ratings, 3)
                print(score)
        else:
            if not df_worker_3[df_worker_3.isin(['-1'])].empty:
                ratings = compute_ratings(df_worker_1.values.tolist(), df_worker_2.values.tolist())
                score = fleiss_kappa(ratings, 2)
            elif not df_worker_2[df_worker_2.isin(['-1'])].empty:
                ratings = compute_ratings(df_worker_1.values.tolist(), df_worker_3.values.tolist())
                score = fleiss_kappa(ratings, 2)
            elif not df_worker_1[df_worker_1.isin(['-1'])].empty:
                ratings = compute_ratings(df_worker_2.values.tolist(), df_worker_3.values.tolist())
                score = fleiss_kappa(ratings, 2)
            else:
                ratings = compute_ratings(df_worker_1.values.tolist(), df_worker_2.values.tolist(),
                                          df_worker_3.values.tolist())
                score = fleiss_kappa(ratings, 3)


        dict_hit_agreement[hit_id] = score

    # dictionary sorted by value
    return OrderedDict(sorted(dict_hit_agreement.items(), key=lambda t: t[1], reverse=True))
