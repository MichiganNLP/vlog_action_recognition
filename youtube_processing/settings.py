import os



channel_ids = ['UCM3P_G21gOSVdepXrEFojIg', 'UC-8yLb1K-DEC6dCYlLOJfiQ', 'UCDy89wegrl-5Qv0ZkTtlnPg',
               'UC0YvTCy1I4_a-3pn47_5DBA',
               'UCK2d_KfjVPwh9gqoczQ9sSw', 'UCSeeUM-1TJjWfxFQfbyg6eA', 'UCVKFs0cesQUEGtn8gQmhbHw',
               'UCJA8OyDxRY-wm0ya2gtHOsw',
               'UCMfXv2enRXepxG92VoxfrEg', 'UCcRkaK3Wn9HcYfpDL_Lqf6g', 'UCuEYwG9tWOvnEyWdbZH98Lg',
               'UCZB32syI0FFtThd6xkPTRrg', 'UCq2E1mIwUKMWzCA4liA_XGQ',
               'UCUuMYw2l2UeWyTGYixYfRCA', 'UCUAvzgYg6QhPsN9bxKg_NEQ']

# I changed and removed some channels due to video content (not enough visible actions)
# 2 playlists per channel - 10 channels
# ('UC0YvTCy1I4_a-3pn47_5DBA','PLBa2sQF9PG4RZqlT6ZOpEnFfscde-N4wb'),
# ('UCSeeUM-1TJjWfxFQfbyg6eA','PLRblXG3KMmX5WBN9XqPFUCGgNxxZ3lMEe'),

# channel_playlist_ids = [('UCM3P_G21gOSVdepXrEFojIg', 'PL5FDE7204E0BE6621'),
#                         ('UCM3P_G21gOSVdepXrEFojIg', 'PL2nx-cbEV-8E4wXli47yM0OvriBPM5isR'), \
#                         ('UC-8yLb1K-DEC6dCYlLOJfiQ', 'PLzeqoZCwnosh1c_BLsdOvTPT_rzjzyEyx'),
#                         ('UC-8yLb1K-DEC6dCYlLOJfiQ', 'PLzeqoZCwnoshKFwYf4S_BE8Gcmy8-tq8u'), \
#                         ('UCDy89wegrl-5Qv0ZkTtlnPg', 'PLUIsKy_4f5UJV9TKP_uuDc4NVXag2z8Bz'),
#                         ('UCDy89wegrl-5Qv0ZkTtlnPg', 'PLUIsKy_4f5UL4rf95P-Hk55IHWKmFSVIo'), \
#                         ('UCbQj1aJiioDM8g0tmGmtC_w', 'PLqG1TZi4MLDtwd_rM4h2HfSOfY2f4-JZ2'),
#                         ('UCbQj1aJiioDM8g0tmGmtC_w', 'PLqG1TZi4MLDux8Nli92_mwBFZkhTLtbFB'), \
#                         ('UCK2d_KfjVPwh9gqoczQ9sSw', 'PLmG6XZ6dRgFLoO3sAd9fxQIsSWpzKFquM'),
#                         ('UCK2d_KfjVPwh9gqoczQ9sSw', 'PLmG6XZ6dRgFI_ukgkblj2sf0wOINpmcjO'), \
#                         ('UCJCgaQzY5z5sA-xwkwibygw', 'PLSQOK8tr3v4IuF9sMQO6w1tk9Q4cSGHSF'),
#                         ('UCJCgaQzY5z5sA-xwkwibygw', 'PLSQOK8tr3v4L87QT4O7dgPHhSRuPSrSPk'), \
#                         ('UCVKFs0cesQUEGtn8gQmhbHw', 'PLHGmd2Hq3kj0YRQOvD2AGbv-LQJ6xZcqY'),
#                         ('UCVKFs0cesQUEGtn8gQmhbHw', 'PLHGmd2Hq3kj0eaJbFh0hJlFIrl2E6xEg7'), \
#                         ('UCJA8OyDxRY-wm0ya2gtHOsw', 'PLp7B7s7uE_2MtxBHIwg6ZKkLVsbqYJNt_'),
#                         ('UCJA8OyDxRY-wm0ya2gtHOsw', 'PLp7B7s7uE_2M2ZFBhQuaM4seaz0HjAtCR'), \
#                         ('UCMfXv2enRXepxG92VoxfrEg', 'PL2bG30UsyEQgCINjHbHXa-PDniLE0i3Nf'),
#                         ('UCMfXv2enRXepxG92VoxfrEg', 'PL2bG30UsyEQgGwwdV9gQpvPamgyfnCHLy'), \
#                         ('UCZB32syI0FFtThd6xkPTRrg', 'PLEpwurODsrVPsXNK7LCvcxmw1UiDMnlvQ'),
#                         ('UCZB32syI0FFtThd6xkPTRrg', 'PLEpwurODsrVMb1VtxJg3jXVX5GamnZ6X_')]

channel_playlist_ids = [('UCM3P_G21gOSVdepXrEFojIg', 'PL5FDE7204E0BE6621'),
                        ('UCM3P_G21gOSVdepXrEFojIg', 'PL2nx-cbEV-8E4wXli47yM0OvriBPM5isR'), \
                        ('UCDy89wegrl-5Qv0ZkTtlnPg', 'PLUIsKy_4f5UJV9TKP_uuDc4NVXag2z8Bz'),
                        ('UCDy89wegrl-5Qv0ZkTtlnPg', 'PLUIsKy_4f5UL4rf95P-Hk55IHWKmFSVIo'), \
                        ('UC-8yLb1K-DEC6dCYlLOJfiQ', 'PLzeqoZCwnoshKFwYf4S_BE8Gcmy8-tq8u'), \
                        ('UC-8yLb1K-DEC6dCYlLOJfiQ', 'PLzeqoZCwnosh1c_BLsdOvTPT_rzjzyEyx'),
                        ('UCbQj1aJiioDM8g0tmGmtC_w', 'PLqG1TZi4MLDux8Nli92_mwBFZkhTLtbFB'), \
                        ('UCbQj1aJiioDM8g0tmGmtC_w', 'PLqG1TZi4MLDtwd_rM4h2HfSOfY2f4-JZ2'),
                        ('UCVKFs0cesQUEGtn8gQmhbHw', 'PLHGmd2Hq3kj0eaJbFh0hJlFIrl2E6xEg7'), \
                        ('UCVKFs0cesQUEGtn8gQmhbHw', 'PLHGmd2Hq3kj0YRQOvD2AGbv-LQJ6xZcqY'),
                        ('UCJCgaQzY5z5sA-xwkwibygw', 'PLSQOK8tr3v4IuF9sMQO6w1tk9Q4cSGHSF'),
                        ('UCJCgaQzY5z5sA-xwkwibygw', 'PLSQOK8tr3v4L87QT4O7dgPHhSRuPSrSPk'), \
                        ('UCJA8OyDxRY-wm0ya2gtHOsw', 'PLp7B7s7uE_2MtxBHIwg6ZKkLVsbqYJNt_'),
                        ('UCJA8OyDxRY-wm0ya2gtHOsw', 'PLp7B7s7uE_2M2ZFBhQuaM4seaz0HjAtCR'), \
                        ('UCK2d_KfjVPwh9gqoczQ9sSw', 'PLmG6XZ6dRgFLoO3sAd9fxQIsSWpzKFquM'),
                        ('UCK2d_KfjVPwh9gqoczQ9sSw', 'PLmG6XZ6dRgFI_ukgkblj2sf0wOINpmcjO'), \
                        ('UCMfXv2enRXepxG92VoxfrEg', 'PL2bG30UsyEQgCINjHbHXa-PDniLE0i3Nf'),
                        ('UCMfXv2enRXepxG92VoxfrEg', 'PL2bG30UsyEQgGwwdV9gQpvPamgyfnCHLy'), \
                        ('UCZB32syI0FFtThd6xkPTRrg', 'PLEpwurODsrVPsXNK7LCvcxmw1UiDMnlvQ'),
                        ('UCZB32syI0FFtThd6xkPTRrg', 'PLEpwurODsrVMb1VtxJg3jXVX5GamnZ6X_')]

PATH_root = "data/Video/"
PATH_csv_url_file = PATH_root + "video_urls/"
PATH_video_transcripts = PATH_root + "videos_captions"
PATH_captions = PATH_video_transcripts + "/captions/"
PATH_videos = PATH_video_transcripts + "/videos/"
PATH_miniclips = PATH_video_transcripts + "/miniclips/"
PATH_problematic_transcripts = PATH_video_transcripts + "/problematic_captions/"
PATH_problematic_videos = PATH_video_transcripts + "/problematic_videos/"

PATH_actions_file = PATH_root + "actions.csv"
# PATH_actions_file = "AMT2/Batch3/actions_FINAL.csv"
# # PATH_INPUT_AMT = "/mnt/c/Users/ignat/Desktop/Workspace_Research/CODE/input_AMT.csv"
#

# PARAM_MAX_NB_ACTIONS_PER_MINICLIP = 7
#
# if not os.path.exists(PATH_csv_url_file):
#     os.makedirs(PATH_csv_url_file)
# if not os.path.exists(PATH_captions):
#     os.makedirs(PATH_captions)
# if not os.path.exists(PATH_videos):
#     os.makedirs(PATH_videos)
# if not os.path.exists(PATH_miniclips):
#     os.makedirs(PATH_miniclips)
# if not os.path.exists(PATH_problematic_transcripts):
#     os.makedirs(PATH_problematic_transcripts)
# if not os.path.exists(PATH_problematic_videos):
#     os.makedirs(PATH_problematic_videos)
