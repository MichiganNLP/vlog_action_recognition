from __future__ import print_function, absolute_import, unicode_literals, division
import tensorflow as tf
import tensorflow_hub as hub
from keras import backend as K, Model
from keras.engine import Layer
from keras import layers

# # Initialize session
sess = tf.Session()
K.set_session(sess)


class ElmoEmbeddingLayer(Layer):
    def __init__(self, **kwargs):
        self.dimensions = 1024
        self.trainable = True
        super(ElmoEmbeddingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.elmo = hub.Module('https://tfhub.dev/google/elmo/2', trainable=self.trainable,
                               name="{}_module".format(self.name))

        self.trainable_weights += K.tf.trainable_variables(scope="^{}_module/.*".format(self.name))
        super(ElmoEmbeddingLayer, self).build(input_shape)

    def call(self, x, mask=None):
        result = self.elmo(K.squeeze(K.cast(x, tf.string), axis=1),
                           as_dict=True,
                           signature='default',
                           )['default']

        return result

    def compute_mask(self, inputs, mask=None):
        return K.not_equal(inputs, '--PAD--')

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.dimensions

#
# def build_elmo_model():
#     input_action = layers.Input(shape=(1,), dtype=tf.string)
#     embedding = ElmoEmbeddingLayer()(input_action)
#     extra_input = layers.Input(shape=(1,), dtype=tf.string)
#     extra_emb = layers.Embedding(128, output_dim= 512)(extra_input)
#     print(embedding.shape, extra_emb.shape)
#     concat_embed = keras.layers.concatenate([embedding, extra_emb])
#
#     lstm = layers.LSTM(128)(concat_embed)
#
#     dense = layers.Dense(256, activation='relu')(lstm)
#     dropout = layers.Dropout(0.5)(dense)
#     pred = layers.Dense(1, activation='sigmoid')(dropout)
#     model = Model(inputs=[input_action, extra_input], outputs=[pred])
#
#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#     model.summary()
#
#     return model


def build_elmo_model():
    input_text = layers.Input(shape=(1,), dtype=tf.string)
    embedding = ElmoEmbeddingLayer()(input_text)
    dense = layers.Dense(256, activation='relu')(embedding)
    dropout = layers.Dropout(0.5)(dense)
    pred = layers.Dense(1, activation='sigmoid')(dropout)
    model = Model(inputs=[input_text], outputs=pred)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    return model


# With the default signature, the module takes untokenized sentences as input
# The input tensor is a string tensor with shape [batch_size].
# The module tokenizes each string by splitting on spaces
def test_elmo2():
    elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
    embeddings = elmo(
        ["the cat", "dogs are big"],
        signature="default",
        as_dict=True)["word_emb"]
    # convert from Tensor to numpy array
    array = K.eval(embeddings)
    return array


'''
The output dictionary contains:

word_emb: the character-based word representations with shape [batch_size, max_length, 512].
lstm_outputs1: the first LSTM hidden state with shape [batch_size, max_length, 1024].
lstm_outputs2: the second LSTM hidden state with shape [batch_size, max_length, 1024].
elmo: the weighted sum of the 3 layers, where the weights are trainable. 
This tensor has shape [batch_size, max_length, 1024]
default: a fixed mean-pooling of all contextualized word representations with shape [batch_size, 1024].
'''


def load_elmo_embedding(train_data):
    print("# ---- loading elmo embeddings ---")
    elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
    embeddings = elmo(
        train_data,
        signature="default",
        as_dict=True)["default"] #["word_emb"]
    # K.eval is computationally expensive (might be doing some convs)
    #array = K.eval(embeddings)
    #print("# ---- loaded elmo embeddings ---")
    return K.eval(embeddings)
