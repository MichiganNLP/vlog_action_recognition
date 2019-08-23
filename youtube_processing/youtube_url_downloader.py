from __future__ import print_function, absolute_import, unicode_literals, division

import json
from youtube_processing.secret_settings import api_key
import csv
import os
import urllib.request


# channel_ids = ['UCM3P_G21gOSVdepXrEFojIg', 'UC-8yLb1K-DEC6dCYlLOJfiQ','UCDy89wegrl-5Qv0ZkTtlnPg','UC0YvTCy1I4_a-3pn47_5DBA',
# 'UCK2d_KfjVPwh9gqoczQ9sSw','UCSeeUM-1TJjWfxFQfbyg6eA','UCVKFs0cesQUEGtn8gQmhbHw','UCJA8OyDxRY-wm0ya2gtHOsw',
# 'UCMfXv2enRXepxG92VoxfrEg','UCcRkaK3Wn9HcYfpDL_Lqf6g','UCuEYwG9tWOvnEyWdbZH98Lg','UCZB32syI0FFtThd6xkPTRrg','UCq2E1mIwUKMWzCA4liA_XGQ',
# 'UCUuMYw2l2UeWyTGYixYfRCA','UCUAvzgYg6QhPsN9bxKg_NEQ']


def get_all_video_in_channel(channel_id, playlist_id):
    base_video_url = 'https://www.youtube.com/watch?v='
    base_search_url = 'https://www.googleapis.com/youtube/v3/playlistItems?'
    first_url = base_search_url + 'key={}&channelId={}&part=snippet,id&order=date&maxResults=25&playlistId={}'.format(
        api_key, channel_id, playlist_id)

    video_links = []
    video_names = []
    url_str = first_url
    while True:
        with urllib.request.urlopen(url_str) as url:
            resp = json.loads(url.read())

        for i in resp['items']:
            video_names.append(i['snippet']['title'])
            if i['snippet']['resourceId']['kind'] == "youtube#video":
                video_links.append(base_video_url + i['snippet']['resourceId']['videoId'])

        try:
            next_page_token = resp['nextPageToken']
            url_str = first_url + '&pageToken={}'.format(next_page_token)
        except:
            break
    return video_links, video_names


def write_url_in_csv(index_channel, index_playlist, channel_id, playlist_id, PATH_csv_url_file):
    video_links, video_names = get_all_video_in_channel(channel_id, playlist_id)

    if not os.path.exists(PATH_csv_url_file):
        os.makedirs(PATH_csv_url_file)

    with open(PATH_csv_url_file + str(index_channel) + "_" + str(index_playlist) + ".csv", 'w+') as csvfile:
        fieldnames = ['Video_URL', 'Video_Name']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        index = 0
        for video_url in video_links:
            writer.writerow({'Video_URL': video_url, 'Video_Name': video_names[index]})
            index += 1
    csvfile.close()
