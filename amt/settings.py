PATH_input_batch = "data/AMT/Input"
PATH_output_batch = "data/AMT/Output"

# initial AMT results
PATH_input_file_name = PATH_input_batch + "/Batch_video_batch_results_all.csv"
# processed AMT results -> GT data is repeated
PATH_output_file_name = PATH_output_batch + "/results_batch_video_all.csv"  # sys.argv[2]
# the miniclips where I agreed with Laura and have at least 4 actions ; used to detect spammers
PATH_output_results_csv = PATH_output_batch + "/results_BOTH_GT_video.csv"  # sys.argv[2]
# AMT output file - the on where I label the spammers & good results
PATH_spammers_files = PATH_output_batch + "/spammers_file.csv"


# compromised hits are filtered out; spammers are labeled with  -1
PATH_after_spam_filter_csv = PATH_output_batch + "/results_after_spam_filter_video.csv"


PATH_visible_not_visible_actions_csv = PATH_output_batch + "/new_clean_visible_not_visible_actions_video_after_spam.csv"
# PATH_visible_not_visible_actions_csv = PATH_output_batch + "/clean_visible_not_visible_actions_video_after_spam.csv"
PATH_GT_AMT = "data/AMT/Input/" + "BOTH_Laura_Oana - Copy.csv"  # sys.argv[2]