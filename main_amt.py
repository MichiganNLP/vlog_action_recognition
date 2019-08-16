#!/usr/bin/env python2
import argparse
import sys
import pprint
import json

sys.path.insert(0, '../')

from amt.read_AMT import process_results_amt
from amt.detect_spam import create_after_spam_filtered_results
from amt.settings import PATH_after_spam_filter_csv, PATH_output_results_csv
from amt.compute_agreement import agreement_per_hit, agreement_total, plot_agreement, agreement_overall, \
    plot_threshold_agreement


def main():
    args = parse_args()

    dict_output, dict_actions_output = process_results_amt(args.print_stats)

    agreements_score = []
    threshold_list = [0.2, 0.5, 0.8, 1]

    print("------ For threshold {0}".format(args.threshold))
    if args.do_spam:
        potential_spammers = create_after_spam_filtered_results(dict_output, args.print_stats, args.threshold)

    if args.do_agreement:

        potential_spammers = create_after_spam_filtered_results(dict_output, args.print_stats, args.threshold)

        ordered_dict = agreement_per_hit(PATH_after_spam_filter_csv, args.do_1_eq_2, args.do_cohen)
        # print("Greatest agreement per hit is {0} for hit nb {1}".format(ordered_dict.values()[0], ordered_dict.keys()[0]))
        # print("Smallest agreement per hit is {0} for hit nb {1}".format(ordered_dict.values()[-1], ordered_dict.keys()[-1]))

        with open("data/dict_agreement.json", 'w') as f:
            json.dump(ordered_dict, f)

        if args.do_plots:
            plot_agreement(ordered_dict.values(), args.threshold)

        avg_agreement = agreement_total(ordered_dict)
        print("Average agreement per hit is {0} when do_1_eq_2 is {1}".format(avg_agreement, args.do_1_eq_2))

        overall_fleiss_agreement_all = agreement_overall(PATH_after_spam_filter_csv, args.do_1_eq_2, None)
        print("All Overall Fleiss agreement is {0} when do_1_eq_2 is {1}".format(overall_fleiss_agreement_all,
                                                                                 args.do_1_eq_2))

        overall_fleiss_agreement_gt = agreement_overall(PATH_output_results_csv, args.do_1_eq_2, potential_spammers)
        print("GT Overall Fleiss agreement is {0} when do_1_eq_2 is {1}".format(overall_fleiss_agreement_gt,
                                                                                args.do_1_eq_2))

        agreements_score.append((avg_agreement, overall_fleiss_agreement_all))

    if args.do_plots:
        plot_threshold_agreement(threshold_list, agreements_score)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--do-read', action='store_true')  # process just original AMT result
    parser.add_argument('--print-stats', type=int, default=1)
    parser.add_argument('--do-1-eq-2', type=str, default=True)
    parser.add_argument('--do-cohen', type=str, default=True)
    parser.add_argument('--do-spam', action='store_true')  # process results after spam removal
    parser.add_argument('--do-agreement', action='store_true')
    parser.add_argument('--do-plots', action='store_true')
    parser.add_argument('--threshold', type=float, default=0.5)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    main()
