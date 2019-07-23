#!/bin/bash

input="/local/oignat/extra_miniclips/"
output="/local/oignat/Action_Recog/action_recognition_clean/data/YOLO/Features/visual/"

for video in `ls $input | grep mp4`; do
    video_name="${video%.*}"
    folder_video="$output$video_name/frames/"
    echo "$folder_video"
    if [ ! -d "$folder_video" ]; then
            mkdir -p "$folder_video";
    fi
    ffmpeg -i  "$input$video" -vf fps=1 "$folder_video"%04d.jpg -hide_banner
    find "$folder_video" -type f -maxdepth 1 > "$output$video_name/"filenames.txt
    #./darknet detector test ./cfg/coco.data ./cfg/yolov3.cfg ./yolov3.weights -dont_show -ext_output < "$output$video_name/"filenames.txt > "$output$video_name/"result.txt


done
