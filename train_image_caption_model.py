import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import time

from image_caption_model import CNN_Encoder, RNN_Decoder, loss_function
from data_utils import get_caption_tokenizer, load_raw_image_path_and_caption_content


def produce_train_dataset(img_name_train, cap_train, cache_image_dir, BATCH_SIZE=64, BUFFER_SIZE=1000):
    # Load the numpy files
    # Note: image features store in cache_image_dir
    def map_func(img_name, cap):
        img_path = img_name.decode('utf-8') + '.npy'
        path_of_feature_file_name = os.path.basename(img_path)
        cache_path = os.path.join(cache_image_dir, path_of_feature_file_name)
        img_tensor = np.load(cache_path)
        return img_tensor, cap

    dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))

    # Use map to load the numpy files in parallel
    dataset = dataset.map(lambda item1, item2: tf.numpy_function(
        map_func, [item1, item2], [tf.float32, tf.int32]),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Shuffle and batch
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE,  drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


def plot_loss_picture(loss_plot, plt_show=False):
    plt.plot(loss_plot)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Plot')
    plt.savefig("loss.png")
    if plt_show:
        plt.show()


def main(EPOCHS, BATCH_SIZE, BUFFER_SIZE, checkpoint_path, cache_image_dir, raw_image_path_and_caption_content_dir):
    @tf.function
    def train_step(img_tensor, target):
        loss = 0
        # initializing the hidden state for each batch
        # because the captions are not related from image to image
        hidden = decoder.reset_state(batch_size=target.shape[0])
        dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * BATCH_SIZE, 1)

        with tf.GradientTape() as tape:
            features = encoder(img_tensor)
            for i in range(1, target.shape[1]):
                # passing the features through the decoder
                predictions, hidden, _ = decoder(dec_input, features, hidden)
                loss += loss_function(target[:, i], predictions)
                # using teacher forcing
                dec_input = tf.expand_dims(target[:, i], 1)

        total_loss = (loss / int(target.shape[1]))
        trainable_variables = encoder.trainable_variables + decoder.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)
        optimizer.apply_gradients(zip(gradients, trainable_variables))
        return loss, total_loss

    # load data
    img_name_train, _, cap_train, _ = load_raw_image_path_and_caption_content(
        save_dir_name=raw_image_path_and_caption_content_dir)

    # prepare dataset
    dataset = produce_train_dataset(img_name_train, cap_train, cache_image_dir, BATCH_SIZE, BUFFER_SIZE)

    # restore tokenizer
    tokenizer = get_caption_tokenizer(caption_tokenizer_path="caption_tokenizer")
    vocab_size = len(tokenizer.word_index) + 1

    num_steps = len(img_name_train) // BATCH_SIZE

    # create model
    encoder = CNN_Encoder()
    decoder = RNN_Decoder(vocab_size)
    optimizer = tf.keras.optimizers.Adam()

    # create checkpoint
    ckpt = tf.train.Checkpoint(encoder=encoder, decoder=decoder, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    start_epoch = 0
    if ckpt_manager.latest_checkpoint:
        start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])

    # train
    loss_plot = []
    for epoch in range(start_epoch, EPOCHS):
        start = time.time()
        total_loss = 0

        for (batch, (img_tensor, target)) in enumerate(dataset):
            batch_loss, t_loss = train_step(img_tensor, target)
            total_loss += t_loss

            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(
                    epoch + 1, batch, batch_loss.numpy() / int(target.shape[1])))
        # storing the epoch end loss value to plot later
        loss_plot.append(total_loss / num_steps)
        #print(f"loss_plot {loss_plot}")
        if epoch % 5 == 0: ckpt_manager.save()

        print('Epoch {} Loss {:.6f}'.format(epoch + 1, total_loss / num_steps))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

    #print(f"loss plot {loss_plot}")
    plot_loss_picture(loss_plot, plt_show=False)


if __name__ == "__main__":
    # Feel free to change these parameters according to your system's configuration
    EPOCHS = 20
    BATCH_SIZE = 64
    BUFFER_SIZE = 1000
    checkpoint_path = "./checkpoints/train"
    cache_image_dir = "cache_image"
    raw_image_path_and_caption_content_dir = "raw_image_path_and_caption_content"
    main(EPOCHS, BATCH_SIZE, BUFFER_SIZE, checkpoint_path, cache_image_dir, raw_image_path_and_caption_content_dir)
