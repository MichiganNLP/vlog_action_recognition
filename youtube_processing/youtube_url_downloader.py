#!/usr/bin/env python2

import urllib
import json
import csv

# channel_ids = ['UCM3P_G21gOSVdepXrEFojIg', 'UC-8yLb1K-DEC6dCYlLOJfiQ','UCDy89wegrl-5Qv0ZkTtlnPg','UC0YvTCy1I4_a-3pn47_5DBA',
# 'UCK2d_KfjVPwh9gqoczQ9sSw','UCSeeUM-1TJjWfxFQfbyg6eA','UCVKFs0cesQUEGtn8gQmhbHw','UCJA8OyDxRY-wm0ya2gtHOsw',
# 'UCMfXv2enRXepxG92VoxfrEg','UCcRkaK3Wn9HcYfpDL_Lqf6g','UCuEYwG9tWOvnEyWdbZH98Lg','UCZB32syI0FFtThd6xkPTRrg','UCq2E1mIwUKMWzCA4liA_XGQ',
# 'UCUuMYw2l2UeWyTGYixYfRCA','UCUAvzgYg6QhPsN9bxKg_NEQ']


def get_all_video_in_channel(channel_id, playlist_id):
    api_key = 'AIzaSyAMvOxtW0vTNKdpczHPs0ZmsTvAWuKLOT4'

    base_video_url = 'https://www.youtube.com/watch?v='
   # base_search_url = 'https://www.googleapis.com/youtube/v3/search?'
    base_search_url = 'https://www.googleapis.com/youtube/v3/playlistItems?'

#https://www.googleapis.com/youtube/v3/playlistItems?part=snippet&maxResults=50&playlistId=PLB03EA9545DD188C3&key=MY_API_KEY

   # first_url = base_search_url+'key={}&channelId={}&part=snippet,id&order=date&maxResults=25'.format(api_key, channel_id)
    first_url = base_search_url+'key={}&channelId={}&part=snippet,id&order=date&maxResults=25&playlistId={}'.format(api_key, channel_id, playlist_id)

    video_links = []
    url = first_url
    while True:
        inp = urllib.urlopen(url)
        resp = json.load(inp)

        for i in resp['items']:
            # if i['id']['kind'] == "youtube#video":
            #     video_links.append(base_video_url + i['id']['videoId'])
            if i['snippet']['resourceId']['kind'] == "youtube#video":
                video_links.append(base_video_url + i['snippet']['resourceId']['videoId'])
            

        try:
            next_page_token = resp['nextPageToken']
            url = first_url + '&pageToken={}'.format(next_page_token)
        except:
            break
    return video_links

def write_url_in_csv(index_channel, index_playlist, channel_id, playlist_id, PATH_csv_url_file):

    video_links = get_all_video_in_channel(channel_id, playlist_id)

    with open(PATH_csv_url_file  + str(index_channel) + "_" + str(index_playlist) + ".csv", 'w+') as csvfile:
        fieldnames = ['Video_URL']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for video_url in video_links:
            #write in file
            writer.writerow({'Video_URL': video_url})
    csvfile.close()

    
