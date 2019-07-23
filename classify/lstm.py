from __future__ import print_function, absolute_import, unicode_literals, division

import os

import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Embedding

from classify.elmo_embeddings import build_elmo_model
from classify.preprocess import process_data
from sklearn.metrics import f1_score, recall_score, precision_score
from keras import Sequential, Input
from keras.layers import LSTM, Dropout, Dense


def create_pretrained_embedding_layer(embedding_matrix_for_pretrain, max_length):
    vocab_size, dimension_embed = embedding_matrix_for_pretrain.shape
    print("vocab size: {0}, dimension_embed: {1}, max_length: {2}".format(vocab_size, dimension_embed, max_length))
    embeddinglayer = Embedding(vocab_size, dimension_embed, weights=[embedding_matrix_for_pretrain],
                               input_length=max_length,
                               trainable=True)

    return embeddinglayer


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


def create_lstm_model(embedding_matrix_for_pretrain, max_length):
    model = Sequential()

    embeddinglayer = create_pretrained_embedding_layer(embedding_matrix_for_pretrain, max_length)
    model.add(embeddinglayer)
    model.add(LSTM(256))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model


def train_lstm(embedding_matrix_for_pretrain, x_train, x_test, x_val, train_data, test_data, val_data):
    _, [train_labels, test_labels, val_labels], _ = process_data(train_data, test_data, val_data)

    max_length = x_train.shape[1]
    print("max_length: %d" % max_length)
    model = create_lstm_model(embedding_matrix_for_pretrain, max_length)


    path_filename_best_model = 'data/Model_params/bestmodel/model_lstm.hdf5'

    if not os.path.isfile(path_filename_best_model):

        checkpointer = ModelCheckpoint(monitor='val_acc', filepath=path_filename_best_model, verbose=1,
                                       save_best_only=True, save_weights_only=True)
        earlystopper = EarlyStopping(monitor='val_acc', patience=10, verbose=1)

        #tensorboard = TensorBoard(log_dir=path_tensorboard, histogram_freq=2000, write_graph=True, write_images=False)

        callback_list = [checkpointer, earlystopper]
        model.fit(x_train, train_labels, validation_data=(x_val, val_labels), epochs=100,
                  batch_size=64, callbacks=callback_list, verbose=1)

    model.load_weights(path_filename_best_model)

    # Evaluate
    score, acc_train = model.evaluate(x_train, train_labels)
    score, acc_test = model.evaluate(x_test, test_labels)
    score, acc_val = model.evaluate(x_val, val_labels)

    # Predict Output
    predicted = model.predict_classes(x_test)
    f1 = f1_score(test_labels, predicted)
    recall = recall_score(test_labels, predicted)
    precision = precision_score(test_labels, predicted)

    return [acc_train, acc_val, acc_test, recall, precision, f1], predicted


def train_elmo(train_data, test_data, val_data):
    [train_actions, test_actions, val_actions], [train_labels, test_labels, val_labels], _ = process_data(train_data,
                                                                                                          test_data,
                                                                                                          val_data)
    train_actions = np.array(train_actions, dtype=object)[:, np.newaxis]
    test_actions = np.array(test_actions, dtype=object)[:, np.newaxis]
    val_actions = np.array(val_actions, dtype=object)[:, np.newaxis]

    model = build_elmo_model()

    file_path_best_model = 'data/Model_params/bestmodel/' + 'model_elmo.hdf5'
    checkpointer = ModelCheckpoint(monitor='val_acc', filepath=file_path_best_model, verbose=1,
                                   save_best_only=True, save_weights_only=True)
    earlystopper = EarlyStopping(monitor='val_acc', patience=10, verbose=1)
    callback_list = [checkpointer, earlystopper]

    if not os.path.isfile(file_path_best_model):
        model.fit(train_actions,
                  train_labels,
                  validation_data=(val_actions, val_labels),
                  epochs=20,
                  batch_size=16, callbacks=callback_list, verbose=1)
    print("Load best model weights from " + file_path_best_model)
    model.load_weights(file_path_best_model)
    # Evaluate
    score, acc_train = model.evaluate(train_actions, train_labels)
    score, acc_val = model.evaluate(val_actions, val_labels)
    score, acc_test = model.evaluate(test_actions, test_labels)

    predicted = model.predict(test_actions) > 0.5
    f1 = f1_score(test_labels, predicted)
    recall = recall_score(test_labels, predicted)
    precision = precision_score(test_labels, predicted)

    return [acc_train, acc_val, acc_test, recall, precision, f1], predicted
