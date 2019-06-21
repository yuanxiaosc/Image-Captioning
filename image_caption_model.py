from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf


class BahdanauAttention(tf.keras.Model):
    def __init__(self, attention_units=512):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(attention_units)
        self.W2 = tf.keras.layers.Dense(attention_units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, encoder_image_features, caption):
        # encoder_image_features shape [batch_size, 64, encoder_image_dim]
        # caption shape [batch_size, caption_embedding_dim]

        # hidden_with_time_axis shape == (batch_size, 1, caption_embedding_dim)
        hidden_with_time_axis = tf.expand_dims(caption, 1)

        # score shape == (batch_size, 64, attention_units)
        score = tf.nn.tanh(self.W1(encoder_image_features) + self.W2(hidden_with_time_axis))

        # attention_weights shape == [batch_size, 64, 1]
        # you get 1 at the last axis because you are applying score to self.V
        attention_weights = tf.nn.softmax(self.V(score), axis=1)

        # context_vector shape after sum == [batch_size, encoder_image_dim]
        context_vector = attention_weights * encoder_image_features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class CNN_Encoder(tf.keras.Model):
    # Since you have already extracted the features and dumped it using pickle
    # This encoder passes those features through a Fully connected layer
    def __init__(self, encoder_image_dim=256):
        super(CNN_Encoder, self).__init__()
        # shape after fc == (batch_size, 64, encoder_image_dim)
        self.fc = tf.keras.layers.Dense(encoder_image_dim)

    def call(self, image_features):
        image_features = self.fc(image_features)
        image_features = tf.nn.relu(image_features)
        return image_features


class RNN_Decoder(tf.keras.Model):
    def __init__(self, vocab_size, caption_embedding_dim=128, decoder_units=512):
        super(RNN_Decoder, self).__init__()
        self.decoder_units = decoder_units

        self.embedding = tf.keras.layers.Embedding(vocab_size, caption_embedding_dim)
        self.gru = tf.keras.layers.GRU(self.decoder_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc1 = tf.keras.layers.Dense(self.decoder_units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)

        self.attention = BahdanauAttention(self.decoder_units)

    def call(self, x, features, hidden):
        # defining attention as a separate model
        # x shape [batch_size, 1]
        # features shape [batch_size, 64, encoder_image_dim]
        # hidden shape [batch_size, caption_embedding_dim]

        # context_vector shape after sum == [batch_size, encoder_image_dim]
        # attention_weights shape == [batch_size, 64, 1]
        context_vector, attention_weights = self.attention(features, hidden)

        # x shape after passing through embedding == (batch_size, 1, caption_embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, encoder_image_dim + caption_embedding_dim)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # shape == (batch_size, max_length, decoder_units)
        x = self.fc1(output)

        # x shape == (batch_size * max_length, decoder_units)
        x = tf.reshape(x, (-1, x.shape[2]))

        # output shape == (batch_size * max_length, vocab_size)
        x = self.fc2(x)

        return x, state, attention_weights

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.decoder_units))



loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)

if __name__=='__main__':
    encoder_image_dim = 256
    caption_embedding_dim = 128
    decoder_units = 512
    vocab_size = 5031
    cnn_encoder = CNN_Encoder(encoder_image_dim)
    rnn_decoder = RNN_Decoder(vocab_size, caption_embedding_dim, decoder_units)

    import numpy as np
    #Test code
    batch_size = 16
    image_features = tf.constant(np.random.random(size=(batch_size, 64, 2048)), dtype=tf.float32)
    print(f"Input: image_features.shape {image_features.shape}")
    image_features_encoder = cnn_encoder(image_features)
    print("---------------Pass by cnn_encoder---------------")
    print(f"Output: image_features_encoder.shape {image_features_encoder.shape}\n")

    batch_words = tf.constant(np.random.random(size=(batch_size, 1)), dtype=tf.float32)
    state = tf.constant(np.random.random(size=(batch_size, decoder_units)), dtype=tf.float32)
    print(f"Input: batch_words.shape {batch_words.shape}")
    print(f"Input: rnn state shape {state.shape}")
    out_batch_words, out_state, attention_weights = rnn_decoder(batch_words, image_features_encoder, state)
    print("---------------Pass by rnn_decoder---------------")
    print(f"Output: out_batch_words.shape {out_batch_words.shape}")
    print(f"Output: out_state.shape {out_state.shape}")
    print(f"Output: attention_weights.shape {attention_weights.shape}")
