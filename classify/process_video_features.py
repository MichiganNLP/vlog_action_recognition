from __future__ import print_function

import argparse
import json
import os
import string
import sys

import cv2
import numpy as np
import scipy
import skvideo
from keras_preprocessing.image import load_img, img_to_array
from tqdm import tqdm
from moviepy.editor import VideoFileClip

from keras.models import Model
# from keras.applications.imagenet_utils import preprocess_input, decode_predictions #For Resnet
from keras.applications.inception_v3 import preprocess_input, decode_predictions

from classify import sports1M_utils
# from classify.c3d import C3D
from keras.applications import InceptionV3


def crop_center(im):
    """
    Crops the center out of an image.
    Args:
        im (numpy.ndarray): Input image to crop.
    Returns:
        numpy.ndarray, the cropped image.
    """

    h, w = im.shape[0], im.shape[1]

    if h < w:
        return im[0:h, int((w - h) / 2):int((w - h) / 2) + h, :]
    else:
        return im[int((h - w) / 2):int((h - w) / 2) + w, 0:w, :]


def get_inception_frame_nb(video_name, path_miniclips):
    path_input_video = os.path.join(path_miniclips, video_name)

    # Open video clip for reading
    try:
        clip = VideoFileClip(path_input_video)
    except Exception as e:
        sys.stderr.write("Unable to read '%s'. Skipping...\n" % video_name)
        sys.stderr.write("Exception: {}\n".format(e))
        return []
    # Sample frames at 1fps
    fps = int(np.round(clip.fps))
    # print("Frames per second: " + str(fps))

    list_frame_nbs = []
    for idx, x in enumerate(clip.iter_frames()):
        if idx % fps == fps // 2:
            list_frame_nbs.append(idx)
    return list_frame_nbs


def test_c3d(path_input_video):
    base_model = C3D(weights='sports1M')
    c3d_model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc8').output)

    with open('/local/oignat/sports-1m-dataset/labels.txt', 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    print('Total labels: {}'.format(len(labels)))

    vid = skvideo.io.vread(str(path_input_video))
    subsample_video = vid[0:16]
    # subsample_video = vid

    input_for_c3d = sports1M_utils.preprocess_input(subsample_video)
    # skvideo.io.vwrite("/local/oignat/action_recognition_clean/data/YOLO/output/" + str(path_input_video.split("/")[-1]),
    #                  input_for_c3d)

    predictions = c3d_model.predict(input_for_c3d)
    print('Position of maximum probability: {}'.format(predictions[0].argmax()))
    # print('Maximum probability: {:.5f}'.format(max(predictions[0][0])))
    print('Maximum probability: {:.5f}'.format(max(predictions[0])))
    print('Corresponding label: {}'.format(labels[predictions[0].argmax()]))

    # sort top five predictions from softmax output
    top_inds = predictions[0].argsort()[::-1][:5]  # reverse sort and take five largest items
    print('\nTop 5 probabilities and labels:')
    _ = [print('{:.5f} {}'.format(predictions[0][i], labels[i])) for i in top_inds]


def load_inception_c3d_feat(path_miniclips, output_dir):
    path_corrected_inceptions = 'data/YOLO/Features/corrected_inception/'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    def is_video(x):
        return x.endswith('.mp4') or x.endswith('.avi') or x.endswith('.mov')

    vis_existing = [x.split('.')[0] for x in os.listdir(output_dir)]
    video_filenames = [x for x in sorted(os.listdir(path_miniclips)) if is_video(x)
                       and os.path.splitext(x)[0] not in vis_existing]

    path_inception_c3d = 'data/YOLO/Features/inception_c3d/'

    for video_name in tqdm(video_filenames):
        print("Video " + video_name)
        print("Video " + video_name)
        path_input_inception = str(path_corrected_inceptions + video_name[:-3] + 'npy')
        path_input_inception_c3d = str(path_inception_c3d + video_name[:-3] + 'npy')

        matrix_inception = np.load(path_input_inception)
        print("matrix_inception.shape: {0}".format(matrix_inception.shape))

        matrix_inception_c3d = np.load(path_input_inception_c3d)
        print("matrix_inception_c3d.shape: {0}".format(matrix_inception_c3d.shape))

        matrix_inception_c3d[:, :2048] = matrix_inception
        print("matrix_inception_c3d.shape: {0}".format(matrix_inception_c3d.shape))

        feat_filepath = os.path.join(output_dir, video_name[:-4] + '.npy')
        with open(feat_filepath, 'w+') as f:
            np.save(f, matrix_inception_c3d)


def get_inception_c3d_feat(path_miniclips, path_inception_feat, output_dir):
    # Get outputs of model from layer just before softmax predictions

    base_model = C3D(weights='sports1M')
    c3d_model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc6').output)

    # Find all videos that need to have features extracted

    def is_video(x):
        return x.endswith('.mp4') or x.endswith('.avi') or x.endswith('.mov')

    vis_existing = [x.split('.')[0] for x in os.listdir(output_dir)]
    video_filenames = [x for x in sorted(os.listdir(path_miniclips)) if is_video(x)
                       and os.path.splitext(x)[0] not in vis_existing]

    # Go through each video and extract features

    for video_name in tqdm(video_filenames):

        path_input_video = str(os.path.join(path_miniclips, video_name))

        try:
            vid = skvideo.io.vread(path_input_video)
        except Exception as e:
            sys.stderr.write("Unable to read '%s'. Skipping...\n" % path_input_video)
            sys.stderr.write("Exception: {}\n".format(e))
            continue

        path_input_inception = str(path_inception_feat + video_name[:-3] + 'npy')

        all_inception_features = np.load(path_input_inception)

        list_inception_frame_nbs = get_inception_frame_nb(video_name, path_miniclips)
        if list_inception_frame_nbs == []:
            continue

        first_inception_frame = list_inception_frame_nbs[0]
        input_for_c3d = sports1M_utils.preprocess_input(vid[first_inception_frame - 8:first_inception_frame + 8])
        c3d_features = c3d_model.predict(input_for_c3d)
        matrix_c3d = c3d_features

        middle_inception_features = all_inception_features[0]

        matrix_inception = middle_inception_features.reshape(1, -1)

        nb_frames_vid = vid.shape[0]
        index = 1
        for inception_frame_nbs in list_inception_frame_nbs[1:]:
            if 8 <= inception_frame_nbs <= nb_frames_vid - 8:
                start = inception_frame_nbs - 8
                end = inception_frame_nbs + 8
            elif inception_frame_nbs < 8:
                start = 0
                end = 16
            else:
                start = nb_frames_vid - 16
                end = nb_frames_vid

            vid_16_frames = vid[start:end]
            input_for_c3d = sports1M_utils.preprocess_input(vid_16_frames)

            c3d_vec_features = c3d_model.predict(input_for_c3d)
            matrix_c3d = np.concatenate((matrix_c3d, c3d_vec_features), axis=0)
            # print("matrix_c3d.shape: {0}".format(matrix_c3d.shape))

            inception_vec_features = all_inception_features[index].reshape(1, -1)
            index += 1
            matrix_inception = np.concatenate((matrix_inception, inception_vec_features), axis=0)

        print("matrix_inception.shape: {0}".format(matrix_inception.shape))
        print("matrix_c3d.shape: {0}".format(matrix_c3d.shape))
        concat_feature_matrix = np.concatenate((matrix_inception, matrix_c3d), axis=1)
        print("matrix_concat.shape: {0}".format(concat_feature_matrix.shape))

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        feat_filepath = os.path.join(output_dir, video_name[:-4] + '.npy')
        with open(feat_filepath, 'w+') as f:
            np.save(f, concat_feature_matrix)

    return

def print_diff_files(dcmp):
    for name in dcmp.left_only:
        print("diff_file %s found in %s and not in %s" % (name, dcmp.left, dcmp.right))
    for sub_dcmp in dcmp.subdirs.values():
        print_diff_files(sub_dcmp)

def read_open_pose_results():
    path_openpose = 'data/YOLO/OpenPose/'
    list_openpose_res = []
    for root, dirs, files in os.walk(path_openpose):
        sorted_files = sorted(files)
        for file in sorted_files:
            if file.endswith(".jpg"):
                file = file.replace('pose', 'points')
                file = file.replace('jpg', 'npy')
                list_openpose_res.append(file)

    path_input_openpose = 'data/YOLO/OpenPose/body_points'
    features = []
    for root, dirs, files in os.walk(path_input_openpose):
        sorted_files = sorted(files)
        old_miniclip = sorted_files[0].split('_points')[0]
        
        per_frame_result_openpose = np.load(file=root + "/" + sorted_files[0])
        # Processed One Hot encoding
        per_frame_one_hot_pose = process_open_pose_results(per_frame_result_openpose)
        features.append(per_frame_one_hot_pose)
        
        for file in sorted_files[1:]:
            miniclip = file.split('_points')[0]
           
            per_frame_result_openpose = np.load(file=root + "/" + file)
            per_frame_one_hot_pose = process_open_pose_results(per_frame_result_openpose)
            
            if miniclip == old_miniclip:
                features.append(per_frame_one_hot_pose)
            else:
                feat_filepath = os.path.join('data/YOLO/Features/coordinates_pose/', old_miniclip + '.npy')
                print("Saved one hot pose for " + old_miniclip)
                old_miniclip = miniclip

                with open(feat_filepath, 'wb') as f:
                    np.save(f, features)
                features = [per_frame_one_hot_pose]

        feat_filepath = os.path.join('data/YOLO/Features/coordinates_pose/', old_miniclip + '.npy')
        print("Saved one hot pose for " + old_miniclip)
        with open(feat_filepath, 'wb') as f:
            np.save(f, features)


def process_open_pose_results(per_frame_result_openpose):

    if not per_frame_result_openpose.shape:
        return np.zeros(25)
    first_pose = per_frame_result_openpose[0]
    one_hot_pose = np.zeros(len(first_pose))

    index = 0
    for v in first_pose:
        if np.any(v):
            one_hot_pose[index] = 1
        index += 1
    return one_hot_pose


def read_inception_results(path_inception):
    with open(path_inception, 'r') as f:
        content = f.read()

    dict_miniclip = {}
    list_result = content.split('data/YOLO/miniclips_results/')
    for results in list_result[1:]:
        miniclip = results.split()[0]
        no_miniclip = ' '.join(results.split()[1:])

        # all_labels = ' '.join(results.split(' '))
        # jpg_results = no_miniclip.split('\n')
        jpg_results = string.split(no_miniclip, ' ')
        image_list = jpg_results[0:len(jpg_results):16]

        index_image = 0
        dict_images = {}
        for i in range(1, len(jpg_results), 16):
            labels = jpg_results[i:i + 15]
            # print(labels[1] + " " + labels[2])
            dict_images[image_list[index_image]] = labels[1] + " " + labels[2] + " " + labels[4] + " " + labels[5]
            index_image += 1
        dict_miniclip[miniclip] = dict_images
    return dict_miniclip


def print_inception_results(path_images_openpose, path_inception, path_to_save):
    dict_miniclip = read_inception_results(path_inception)

    for root, dirs, files in os.walk(path_images_openpose):
        sorted_files = sorted(files)
        for file in sorted_files:
            if file.endswith(".jpg"):
                file = root + "/" + file

                miniclip = file.split('/')[-1].split('pose')[0][:-1]
                image = file.split('/')[-1].split('pose')[1]
                object_labels = dict_miniclip[miniclip][image]

                # print(miniclip, image, object_labels)

                # load the image via OpenCV, draw the top prediction on the image,
                # and display the image to our screen
                orig = cv2.imread(file)
                cv2.putText(orig, "Labels: " + object_labels,
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                # cv2.imwrite(path_to_save + miniclip + "_poseobj_" + image, orig)

                cv2.imwrite(file, orig)
                print("Saved obj on " + file)
                # cv2.imshow("Classification", orig)
                # cv2.waitKey(0)


def run_inception(path):
    model = InceptionV3(include_top=True, weights='imagenet')

    for root, dirs, files in os.walk(path):
        print(root)
        sorted_files = sorted(files)
        for file in sorted_files:
            if file.endswith(".jpg"):
                print(file)
                file = root + "/" + file
                image = load_img(file, target_size=(299, 299))
                image = img_to_array(image)
                # print(image.shape)
                image = np.expand_dims(image, axis=0)
                # print(image.shape)
                image = preprocess_input(image)
                preds = model.predict(image)
                # print(preds.shape)
                P = decode_predictions(preds)
                # loop over the predictions and display the rank-5 predictions +
                # probabilities to our terminal
                for (i, (imagenetID, label, prob)) in enumerate(P[0]):
                    print("{}. {}: {:.2f}%".format(i + 1, label, prob * 100))

                # load the image via OpenCV, draw the top prediction on the image,
                # and display the image to our screen
                # orig = cv2.imread(file)
                # (imagenetID, label, prob) = P[0][0]
                # cv2.putText(orig, "Label: {}, {:.2f}%".format(label, prob * 100),
                #             (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                # cv2.imshow("Classification", orig)
                # cv2.waitKey(0)


def extract_video_features(input_dir, output_dir, model_type='inceptionv3'):
    """
    Extracts features from a CNN trained on ImageNet classification from all
    videos in a directory.
    Args:
        input_dir (str): Input directory of videos to extract from.
        output_dir (str): Directory where features should be stored.
        model_type (str): Model type to use.
        batch_size (int): Batch size to use when processing.
    """
    print(model_type)
    if not os.path.isdir(input_dir):
        sys.stderr.write("Input directory '%s' does not exist!\n" % input_dir)
        sys.exit(1)

    # Load desired ImageNet model

    # Note: import Keras only when needed so we don't waste time revving up
    #       Theano/TensorFlow needlessly in case of an error
    # visual_dir = os.path.join(output_dir, 'corrected_inception')  # RGB features
    visual_dir = os.path.join(output_dir, 'visual')  # RGB features

    # if model_type.lower() == 'inceptionv3':
    #     model = InceptionV3(include_top=True, weights='imagenet')
    # elif model_type.lower() == 'xception':
    #     from keras.applications import Xception
    #     model = Xception(include_top=True, weights='imagenet')
    # elif model_type.lower() == 'resnet50':
    #     from keras.applications import ResNet50
    #     model = ResNet50(include_top=True, weights='imagenet')
    # elif model_type.lower() == 'vgg16':
    #     from keras.applications import VGG16
    #     model = VGG16(include_top=True, weights='imagenet')
    # elif model_type.lower() == 'vgg19':
    #     from keras.applications import VGG19
    #     model = VGG19(include_top=True, weights='imagenet')
    #
    # else:
    #     sys.stderr.write("'%s' is not a valid ImageNet model.\n" % model_type)
    #     sys.exit(1)
    #
    # if model_type.lower() == 'inceptionv3' or model_type.lower() == 'xception':
    #     shape = (299, 299)
    # elif model_type.lower() == 'resnet50':
    #     shape = (224, 224)

    # Create output directories

    # motion_dir = os.path.join(output_dir, 'motion') # Spatiotemporal features
    # opflow_dir = os.path.join(output_dir, 'opflow') # Optical flow features

    for directory in [visual_dir]:  # , motion_dir, opflow_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # Find all videos that need to have features extracted

    def is_video(x):
        return x.endswith('.mp4') or x.endswith('.avi') or x.endswith('.mov')

    vis_existing = [x.split('.')[0] for x in os.listdir(visual_dir)]
    # mot_existing = [os.path.splitext(x)[0] for x in os.listdir(motion_dir)]
    # flo_existing = [os.path.splitext(x)[0] for x in os.listdir(opflow_dir)]

    video_filenames = [x for x in sorted(os.listdir(input_dir))
                       if is_video(x) and os.path.splitext(x)[0] not in vis_existing]

    # # Go through each video and extract features
    # model = Model(model.inputs, output=model.layers[-2].output)

    for video_filename in tqdm(video_filenames):

        # Open video clip for reading
        try:
            clip = VideoFileClip(os.path.join(input_dir, video_filename))
        except Exception as e:
            sys.stderr.write("Unable to read '%s'. Skipping...\n" % video_filename)
            sys.stderr.write("Exception: {}\n".format(e))
            continue

        # Sample frames at 1fps
        fps = int(np.round(clip.fps))
        # frames = [scipy.misc.imresize(crop_center(x.astype(np.float32)), shape)
        frames = [x.astype(np.float32)
                  for idx, x in enumerate(clip.iter_frames()) if idx % fps == fps // 2]

        n_frames = len(frames)

        # frames_arr = np.empty((n_frames,) + shape + (3,), dtype=np.float32)
        # for idx, frame in enumerate(frames):
        #     frames_arr[idx, :, :, :] = frame
        #
        # frames_arr = preprocess_input(frames_arr)
        #
        # features = model.predict(frames_arr, batch_size=32)

        name, _ = os.path.splitext(video_filename)
        # feat_filepath = os.path.join(visual_dir, name + '.npy')

        path_to_save_preprocessed_frames = visual_dir + "/"+name + "/"
        print("Saving" + path_to_save_preprocessed_frames)
        if not os.path.exists(path_to_save_preprocessed_frames):
            os.makedirs(path_to_save_preprocessed_frames)

        # for idx, frame in enumerate(frames_arr):
        for idx, frame in enumerate(frames):
            cv2.imwrite(path_to_save_preprocessed_frames + "frame%d.jpg" % idx, frame)

        # with open(feat_filepath, 'wb') as f:
        #     np.save(f, features)

def split_video_into_frames():
    vidcap = cv2.VideoCapture('output.mp4')
    success, image = vidcap.read()
    count = 0
    success = True
    while success:
        cv2.imwrite("video_frames/frame%d.jpg" % count, image)  # save frame as JPEG file
        success, image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1


def print_action_concreteness(dict_video_actions,
                              path_images_openpose='/local/oignat/Action_Recog/large_data/open_pose_img',
                              path_save='/local/oignat/Action_Recog/large_data/concr/'):
    with open('/local/oignat/Action_Recog/action_recognition_clean/data/dict_concreteness.json') as f:
        dict_concreteness = json.loads(f.read())
    list_miniclips = []
    nb_concrete_notvisible = 0
    nb_abstract_notvisible = 0
    nb_concrete_visible = 0
    nb_abstract_visible = 0
    for root, dirs, files in os.walk(path_images_openpose):
        sorted_files = sorted(files)
        for file in sorted_files:
            if file.endswith(".jpg"):
                file = root + "/" + file

                miniclip = file.split('/')[-1].split('pose')[0][:-1]
                image = file.split('/')[-1].split('pose')[1]
                if miniclip+ '.mp4' not in dict_video_actions.keys() or miniclip+ '.mp4' in list_miniclips:
                    continue
                list_miniclips.append(miniclip+ '.mp4')

                action_label_list = dict_video_actions[miniclip+ '.mp4']

                str_action_label = ""
                for [action, label] in action_label_list:
                    if label != 0:
                        continue
                    if action in dict_concreteness.keys():
                        score = dict_concreteness[action][0]
                        word = dict_concreteness[action][1]
                        if score >= 4.0:
                            nb_concrete_visible += 1
                        else:
                            nb_abstract_visible += 1

                    else:
                        score = 0
                        word = ''
                    str_action_label += action +" | " + word + " " + str(score) + " | " + str(label) + '\n'

                for [action, label] in action_label_list:
                    if label == 0:
                        continue
                    if action in dict_concreteness.keys():
                        score = dict_concreteness[action][0]
                        word = dict_concreteness[action][1]
                        if score >= 4.0:
                            nb_concrete_notvisible += 1
                        else:

                            nb_abstract_notvisible += 1
                    else:
                        score = 0
                        word = ''
                        # print(action, word, score, miniclip)
                    str_action_label += action +" | " + word + " " + str(score) + " | " + str(label) + '\n'
                # print(miniclip, image, object_labels)
                # load the image via OpenCV, draw the top prediction on the image,
                # and display the image to our screen
                # orig = cv2.imread(file)
                # # # cv2.putText(orig, str_action_label,
                # # #             (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                # y0, dy = 60, 30
                # for i, line in enumerate(str_action_label.split('\n')):
                #     y = y0 + i * dy
                #     cv2.putText(orig, line, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                #
                # # #cv2.imwrite(file, orig)
                # cv2.imwrite(path_save + miniclip + "_poseobjconcr_" + image, orig)
                #
                # print("Saved action_concreteness_labels on " + path_save + miniclip + "_poseobjconcr_" + image)

                # cv2.imshow("action_concreteness_labels", orig)
                # cv2.waitKey(0)
    print("nb_abstract_visible:", nb_abstract_visible)
    print("nb_concrete_visible:", nb_concrete_visible)
    print("nb_concrete_notvisible:", nb_concrete_notvisible)
    print("nb_abstract_notvisible:", nb_abstract_notvisible)


if __name__ == '__main__':
    # test_c3d(path_input_video='/local/oignat/miniclips/0mini_0.mp4')
    #  run_inception(path='data/YOLO/miniclips_results/')
    # extract_video_features('', '', 'inceptionv3')
    #print_inception_results(path_images_openpose = '/local/oignat/Action_Recog/large_data/open_pose_img', path_inception = 'data/Test/features/Inception_classif_results.txt', path_to_save = '/local/oignat/Action_Recog/large_data/open_pose_img')

    # print_action_concreteness(dict_video_actions,
    #                           path_images_openpose='/local/oignat/Action_Recog/large_data/open_pose_img')
    # read_inception_results(path_inception='data/Test/features/Inception_classif_results.txt')
    read_open_pose_results()
