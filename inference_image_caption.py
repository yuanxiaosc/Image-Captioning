import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from data_utils import load_image, initialize_InceptionV3_image_features_extract_model, \
    get_caption_tokenizer, load_raw_image_path_and_caption_content
from image_caption_model import CNN_Encoder, RNN_Decoder


def plot_attention(idx, image, result, attention_plot, plt_show=False):
    temp_image = np.array(Image.open(image))

    fig = plt.figure(figsize=(10, 10))

    len_result = len(result)
    if not (len_result % 2) == 0:
        len_result = len_result - 1

    for l in range(len_result):
        temp_att = np.resize(attention_plot[l], (8, 8))
        ax = fig.add_subplot(len_result // 2, len_result // 2, l + 1)
        ax.set_title(result[l])
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())

    plt.tight_layout()
    plt.savefig(f"{idx}.png")
    if plt_show:
        plt.show()


def restore_model(checkpoint_path, vocab_size):
    image_features_extract_model = initialize_InceptionV3_image_features_extract_model()
    encoder = CNN_Encoder()
    decoder = RNN_Decoder(vocab_size)
    optimizer = tf.keras.optimizers.Adam()

    ckpt = tf.train.Checkpoint(encoder=encoder,
                               decoder=decoder,
                               optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!')

    return image_features_extract_model, encoder, decoder,


def main(model_predicte_number, checkpoint_path, raw_image_path_and_caption_content_dir,
         caption_max_length, attention_features_shape, plot_image_attention):
    def evaluate():
        attention_plot = np.zeros((caption_max_length, attention_features_shape))

        hidden = decoder.reset_state(batch_size=1)

        temp_input = tf.expand_dims(load_image(image)[0], 0)
        img_tensor_val = image_features_extract_model(temp_input)
        img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

        image_features_encoder = encoder(img_tensor_val)

        dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
        result = []
        for i in range(caption_max_length):
            predictions, hidden, attention_weights = decoder(dec_input, image_features_encoder, hidden)

            attention_plot[i] = tf.reshape(attention_weights, (-1,)).numpy()

            predicted_id = tf.argmax(predictions[0]).numpy()
            result.append(tokenizer.index_word[predicted_id])

            if tokenizer.index_word[predicted_id] == '<end>':
                return result, attention_plot

            dec_input = tf.expand_dims([predicted_id], 0)

        attention_plot = attention_plot[:len(result), :]
        return result, attention_plot

    # restore tokenizer
    tokenizer = get_caption_tokenizer(caption_tokenizer_path="caption_tokenizer")
    vocab_size = len(tokenizer.word_index) + 1

    # Preparing validation set data
    _, img_name_val, _, cap_val = load_raw_image_path_and_caption_content(
        save_dir_name=raw_image_path_and_caption_content_dir)

    # restore image caption model
    image_features_extract_model, encoder, decoder = restore_model(checkpoint_path, vocab_size)

    # model prediction
    for idx in range(model_predicte_number):
        # captions on the validation set
        rid = np.random.randint(0, len(img_name_val))
        image = img_name_val[rid]
        real_caption = ' '.join([tokenizer.index_word[i] for i in cap_val[rid] if i not in [0]])
        result, attention_plot = evaluate()

        print('Real Caption:', real_caption)
        print('Prediction Caption:', ' '.join(result))
        if plot_image_attention:
            Image.open(image)
        plot_attention(idx, image, result, attention_plot, plt_show=False) #Todo:fix Bug
        print("")


if __name__ == "__main__":
    model_predicte_number = 5
    caption_max_length = 30
    checkpoint_path = "checkpoints/train"
    raw_image_path_and_caption_content_dir = "raw_image_path_and_caption_content"
    attention_features_shape = 64
    plot_image_attention = False

    main(model_predicte_number, checkpoint_path, raw_image_path_and_caption_content_dir,
         caption_max_length, attention_features_shape, plot_image_attention)
