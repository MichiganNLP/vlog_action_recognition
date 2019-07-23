from __future__ import print_function, absolute_import, unicode_literals, division

import os
from time import time

import numpy as np
import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Input

from classify.elmo_embeddings import ElmoEmbeddingLayer
from classify.preprocess import get_visual_features, process_data
from sklearn.metrics import f1_score, recall_score, precision_score

from classify.utils import reshape_3d_to_2d

from keras import backend as K, Model
from keras import layers
import tensorflow as tf

# # Initialize session
sess = tf.Session()
K.set_session(sess)

from keras.utils import multi_gpu_model
from keras.utils import plot_model


def build_dense_model(input_size):
    model = Sequential()
    model.add(Dense(128, input_dim=input_size, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    model.summary()

    return model


def train_dense(x_train, x_test, x_val, train_data, test_data, val_data):
    _, [train_labels, test_labels, val_labels], _ = process_data(train_data, test_data, val_data)

    model = build_dense_model(x_train.shape[1])

    model.fit(x_train, train_labels,
              validation_data=(x_val, val_labels),
              epochs=30, batch_size=64, verbose=0)

    # Evaluate
    score, acc_train = model.evaluate(x_train, train_labels)
    score, acc_val = model.evaluate(x_val, val_labels)
    score, acc_test = model.evaluate(x_test, test_labels)

    predicted = model.predict_classes(x_test)
    f1 = f1_score(test_labels, predicted)
    recall = recall_score((test_labels, predicted))
    precision = precision_score((test_labels, predicted))

    return [acc_train, acc_val, acc_test, recall, precision, f1], predicted


def create_model_concat(input_dim_video, input_dim_text):
    # The first input
    video_input = Input(shape=(input_dim_video[1], input_dim_video[2],), dtype='float32', name='video_input')

    # A LSTM will transform the vector sequence into a single vector,
    # containing information about the entire sequence
    lstm_out = LSTM(128)(video_input)
    # Second input
    action_input = Input(shape=(input_dim_text[1],), name='action_input')

    x = keras.layers.concatenate([lstm_out, action_input])

    # We stack a deep densely-connected network on top
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)

    # And finally we add the main logistic regression layer
    main_output = Dense(1, activation='sigmoid', name='main_output')(x)

    model = Model(inputs=[video_input, action_input], outputs=[main_output])

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()

    return model

def create_elmo_extra(input_dim_extra):
    # The first input
    action_input = layers.Input(shape=(1,), dtype=tf.string)
    embedding = ElmoEmbeddingLayer()(action_input)

    # Second input
    extra_action_input = Input(shape=(input_dim_extra[1],), dtype='float32', name='extra_action_input')
    x = keras.layers.concatenate([embedding, extra_action_input])

    # We stack a deep densely-connected network on top
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)

    # And finally we add the main logistic regression layer
    main_output = Dense(1, activation='sigmoid', name='main_output')(x)

    model = Model(inputs=[action_input, extra_action_input], outputs=[main_output])

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()

    return model


def create_elmo_video_model(input_dim_video, input_dim_extra):
    # The first input
    video_input = Input(shape=(input_dim_video[1], input_dim_video[2],), dtype='float32', name='video_input')

    # A LSTM will transform the vector sequence into a single vector,
    # containing information about the entire sequence
    lstm_out = LSTM(128)(video_input)
    # Second input
    action_input = layers.Input(shape=(1,), dtype=tf.string)
    embedding = ElmoEmbeddingLayer()(action_input)

    # Third Input
    if input_dim_extra is None:
        x = keras.layers.concatenate([lstm_out, embedding])
    else:
        extra_action_input = Input(shape=(input_dim_extra[1],), dtype='float32', name='extra_action_input')
        x = keras.layers.concatenate([lstm_out, embedding, extra_action_input])

    # We stack a deep densely-connected network on top
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)

    # And finally we add the main logistic regression layer
    main_output = Dense(1, activation='sigmoid', name='main_output')(x)

    if input_dim_extra is None:
        model = Model(inputs=[video_input, action_input], outputs=[main_output])
    else:
        model = Model(inputs=[video_input, action_input, extra_action_input], outputs=[main_output])

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    plot_model(model, to_file='data/model_elmo_video.png', show_shapes=True)

    model.summary()

    return model


def text_concat_elmo(param_epochs, train_data, test_data, val_data, x_train, x_test, x_val,
                           channel_test, add_extra):
    [train_actions, test_actions, val_actions], [train_labels, test_labels, val_labels], _ = process_data(train_data, test_data, val_data)

    text_data_train = np.array(train_actions, dtype=object)[:, np.newaxis]
    text_data_test = np.array(test_actions, dtype=object)[:, np.newaxis]
    text_data_val = np.array(val_actions, dtype=object)[:, np.newaxis]
    print("Elmo actions, concat text_data_train.shape: {0}".format(text_data_train.shape))

    model = create_elmo_extra(x_train.shape)
    try:
        model = multi_gpu_model(model)
    except:
        print("parallelizing is not enabled")
        pass


    file_path_best_model = 'data/Model_params/bestmodel/' + 'model_elmo_extra' + '_' + str(add_extra) + str(
            channel_test) + '.hdf5'

    checkpointer = ModelCheckpoint(monitor='val_acc', filepath=file_path_best_model, verbose=1,
                                   save_best_only=True, save_weights_only=True)
    earlystopper = EarlyStopping(monitor='val_acc', patience=20, verbose=1)

    tensorboard = TensorBoard(log_dir="ar_elmo_logs/{}".format(time()))


    # TODO: ERROR because tensorboard expects tring input in the model, I need float
    # tensorboard = TensorBoard(log_dir=path_tensorboard, histogram_freq=2000, write_graph=True, write_images=False)

    callback_list = [checkpointer, earlystopper, tensorboard]

    # Evaluate

    print("Extra info: x_train.shape: {0}".format(x_train.shape))
    if not os.path.isfile(file_path_best_model):
        model.fit([text_data_train, x_train], [train_labels],
                  validation_data=([text_data_val, x_val], [val_labels]),
                  epochs=param_epochs, batch_size=256, callbacks=callback_list)

    print("Load best model weights from " + file_path_best_model)
    model.load_weights(file_path_best_model)

    score, acc_train = model.evaluate([text_data_train, x_train], [train_labels])
    score, acc_val = model.evaluate([text_data_val, x_val], [val_labels])
    score, acc_test = model.evaluate([text_data_test, x_test], [test_labels])

    predicted = model.predict([text_data_test, x_test]) > 0.5

    f1 = f1_score(test_labels, predicted)
    recall = recall_score(test_labels, predicted)
    precision = precision_score(test_labels, predicted)
    return [acc_train, acc_val, acc_test, recall, precision, f1], predicted



def video_text_concat_elmo(param_epochs, train_data, test_data, val_data, x_train, x_test, x_val, type_feat,
                           add_extra, avg_or_concatenate):
    [train_actions, test_actions, val_actions], [train_labels, test_labels, val_labels], [train_miniclips,
                                                                                          test_miniclips, val_miniclips] \
        = process_data(train_data, test_data, val_data)

    text_data_train = np.array(train_actions, dtype=object)[:, np.newaxis]
    text_data_test = np.array(test_actions, dtype=object)[:, np.newaxis]
    text_data_val = np.array(val_actions, dtype=object)[:, np.newaxis]
    print("Elmo actions, concat text_data_train.shape: {0}".format(text_data_train.shape))

    video_data_train, video_data_test, video_data_val = get_visual_features(train_miniclips, test_miniclips,
                                                                            val_miniclips, type_feat, avg_or_concatenate)

    print("video_data_train.shape: {0}".format(video_data_train.shape))
    if x_train is None:
        model = create_elmo_video_model(video_data_train.shape, None)
    else:
        model = create_elmo_video_model(video_data_train.shape, x_train.shape)
    try:
        model = multi_gpu_model(model)
    except:
        print("parallelizing is not enabled")
        pass

    patience_param = 20

    if x_train is None:
        file_path_best_model = 'data/Model_params/bestmodel/' + 'model_elmo_concat_corr_' + str(type_feat) + "_pat" + str(patience_param) + '.hdf5'
    else:
        file_path_best_model = 'data/Model_params/bestmodel/' + 'model_elmo_concat_corr_' + str(type_feat) + "_pat"+ str(patience_param) + '_' + str(add_extra) + '.hdf5'

    checkpointer = ModelCheckpoint(monitor='val_acc', filepath=file_path_best_model, verbose=1,
                                   save_best_only=True, save_weights_only=True)
    earlystopper = EarlyStopping(monitor='val_acc', patience=patience_param, verbose=1)

    tensorboard = TensorBoard(log_dir="ar_multimodal_logs/{}".format(time()))


    # TODO: ERROR because tensorboard expects tring input in the model, I need float
    # tensorboard = TensorBoard(log_dir=path_tensorboard, histogram_freq=2000, write_graph=True, write_images=False)

    callback_list = [checkpointer, earlystopper,tensorboard]

    # Evaluate
    if x_train is None:
        if not os.path.isfile(file_path_best_model):
            model.fit([video_data_train, text_data_train], [train_labels],
                      validation_data=([video_data_val, text_data_val], [val_labels]),
                      epochs=100, batch_size=256, callbacks=callback_list, verbose=1)

        print("Load best model weights from " + file_path_best_model)
        model.load_weights(file_path_best_model)

        score, acc_train = model.evaluate([video_data_train, text_data_train], [train_labels])
        score, acc_val = model.evaluate([video_data_val, text_data_val], [val_labels])
        score, acc_test = model.evaluate([video_data_test, text_data_test], [test_labels])
        predicted = model.predict([video_data_test, text_data_test]) > 0.5

    else:
        print("Extra info: x_train.shape: {0}".format(x_train.shape))
        if not os.path.isfile(file_path_best_model):
            model.fit([video_data_train, text_data_train, x_train], [train_labels],
                      validation_data=([video_data_val, text_data_val, x_val], [val_labels]),
                      epochs=param_epochs, batch_size=256, callbacks=callback_list)

        print("Load best model weights from " + file_path_best_model)
        model.load_weights(file_path_best_model)

        score, acc_train = model.evaluate([video_data_train, text_data_train, x_train], [train_labels])
        score, acc_val = model.evaluate([video_data_val, text_data_val, x_val], [val_labels])
        score, acc_test = model.evaluate([video_data_test, text_data_test, x_test], [test_labels])

        predicted = model.predict([video_data_test, text_data_test, x_test]) > 0.5

    f1 = f1_score(test_labels, predicted)
    recall = recall_score(test_labels, predicted)
    precision = precision_score(test_labels, predicted)
    return [acc_train, acc_val, acc_test, recall, precision, f1], predicted


def train_video_text_concat(param_epochs, train_data, test_data, val_data, text_data_train, text_data_test,
                            text_data_val, type_feat):
    _, [train_labels, test_labels, val_labels], [train_miniclips, test_miniclips, val_miniclips] \
        = process_data(train_data, test_data, val_data)

    video_data_train, video_data_test, video_data_val = get_visual_features(train_miniclips, test_miniclips,
                                                                            val_miniclips, type_feat)

    if len(text_data_train.shape) == 3:
        text_data_train = reshape_3d_to_2d(text_data_train)
        text_data_test = reshape_3d_to_2d(text_data_test)
        text_data_val = reshape_3d_to_2d(text_data_val)

    print("concat text_data_train.shape: {0}".format(text_data_train.shape))
    print("concat video_data_train.shape: {0}".format(video_data_train.shape))

    model = create_model_concat(video_data_train.shape, text_data_train.shape)
    try:
        model = multi_gpu_model(model)
    except Exception as e:
        print("parallelizing is not enabled")
        print(e)

    model.fit([video_data_train, text_data_train], [train_labels],
              validation_data=([video_data_val, text_data_val], [val_labels]),
              epochs=param_epochs, batch_size=256)

    # Evaluate
    score, acc_train = model.evaluate([video_data_train, text_data_train], [train_labels])
    score, acc_val = model.evaluate([video_data_val, text_data_val], [val_labels])
    score, acc_test = model.evaluate([video_data_test, text_data_test], [test_labels])

    predicted = model.predict_classes([video_data_test, text_data_test]) > 0.5

    f1 = f1_score(test_labels, predicted)
    recall = recall_score(test_labels, predicted)
    precision = precision_score(test_labels, predicted)

    return [acc_train, acc_val, acc_test, recall, precision, f1], predicted
