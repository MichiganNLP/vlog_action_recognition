# Identifying Visible Actions in Lifestyle Vlogs

This repository contains the dataset and code for our ACL 2019 paper:

[Identifying Visible Actions in Lifestyle Vlogs](https://arxiv.org/abs/1906.04236)

## Task Description

![Example instance](images/task_description.jpg)
<p align="center"> Given a video and its transcript, which human actions are visible in the video? </p>


## Raw Videos

We provide a [Google Drive folder with the raw video miniclips](https://drive.google.com/drive/folders/1JJApVcu5o_zvtL3M0Y9a1sAiJCVTPQ6z?usp=sharing),

## Data Format
The annotations of the miniclips are available at [`data/miniclip_action.json`](data/actions_miniclip.json).
The JSON file contains a dictionary: the keys represent the miniclips (e.g. "4p1_3mini_5.mp4") and the values represent the (action, label) pairs.
The miniclip name is formed by concatenating its YouTube channel, playlist (0 or 1), video and miniclip index:
For "4p1_3mini_5.mp4":
* 4 = channel index
* p1 = playlist index (0 or 1) in the channel
* 3 = video index in the playlist
* mini_5 = miniclip index in the video

Example format in json:

```json
{
  "4p1_3mini_5": [
                    ["smelled it", 1],
                    ["used this in my last pudding video", 1],
                    ["make it smell nice", 0],
                    ["loving them", 1],
                    ["using them about a week", 1],
                    ["using my favorite cleaner which", 0],
                    ["work really really well", 1],
                    ["wipe down my counters", 0],
                    ["wiping down my barstools i", 0],
                    ["using the car shammies that i", 0]
    ]
}

For each miniclip we store actions and their corresponding labels: 0 for visible and 1 for not visible.

```
## Citation

Please cite the following paper if you find this dataset useful in your research:

```tex
@inproceedings{mustard,
    title = "Identifying Visible Actions in Lifestyle Vlogs",
    author = "Ignat, Oana and Burdick, Laura and Deng, Jia and Mihalcea, Rada",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = "7",
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
}
```

## Run the code