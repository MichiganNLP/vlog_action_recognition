#!/usr/bin/env python2
from __future__ import print_function, absolute_import, unicode_literals, division

import argparse
from youtube_processing.settings import channel_playlist_ids, PATH_csv_url_file, PATH_video_transcripts, PATH_captions, \
    PATH_problematic_transcripts, PATH_problematic_videos, PATH_videos, PATH_miniclips, PATH_actions_file
from youtube_processing.youtube_url_downloader import write_url_in_csv
from youtube_processing.create_video_transcript_dataset import create_initial_video_transcripts, convert_to_mp4
from youtube_processing.filter_transcript import filter_transcripts
from youtube_processing.script_html import create_miniclips, filter_miniclips_max_60_minutes
from youtube_processing.filter_miniclip import filter_miniclips

import json

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--save_urls', type=int, default=0, help="Saving the video urls in csv file...")
    parser.add_argument('-i', '--create_initial_video_transcript', type=int, default=0,
                        help="Creating the initial video& transcripts dataset...")
    parser.add_argument('-ft', '--filter_transcript', type=int, default=0, help="Filter transcripts...")
    parser.add_argument('--create_miniclips', type=int, default=0, help="Create miniclips...")
    parser.add_argument('-fm', '--filter_miniclips', type=int, default=0, help="Filter miniclips...")

    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    if args.save_urls:
        index_channel = 0
        index_playlist = 1
        for (channel_id, playlist_id) in channel_playlist_ids:
            index_playlist += 1
            if index_playlist % 2 == 0:
                index_playlist = 0
                index_channel += 1

        write_url_in_csv(index_channel, index_playlist, channel_id, playlist_id, PATH_csv_url_file)
        print("DONE Saving the video urls in csv file")

    if args.create_initial_video_transcript:
        create_initial_video_transcripts(PATH_csv_url_file, PATH_video_transcripts, PARAM_MAX_NB_URL_PER_CHANNEL=10)
        convert_to_mp4(PATH_video_transcripts)
        print("DONE Creating the initial video& transcripts dataset")

    if args.filter_transcript:
        filter_transcripts(PATH_captions, PATH_problematic_transcripts, PATH_problematic_videos,
                           PARAM_LEAST_NB_WORDS_PER_SEC=0.5)
        print("DONE Filtering transcripts")

    if args.create_miniclips:
        clip_actions_time = create_miniclips(PATH_actions_file, PATH_videos, PATH_miniclips)
        print("DONE Creating miniclips")

    if args.filter_miniclips:
        filter_miniclips(PATH_miniclips, PATH_problematic_videos, PARAM_CORR2D_COEFF=0.8)
        # # filter miniclips to have max 60 seconds
        # filtered_clip_actions_time = filter_miniclips_max_60_minutes(clip_actions_time, PATH_miniclips)
        #
        # #to save it as json
        # filtered_clip_actions_time =  {str(k): v for k, v in filtered_clip_actions_time.items()}
        # with open('AMT2/Batch3/filtered_actions_time_FINAL.json', 'w+') as outfile:
        #     json.dump(filtered_clip_actions_time, outfile)

        print("DONE Filtering miniclips")


if __name__ == '__main__':
    main()
