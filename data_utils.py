import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
import os
import json
import pickle
from tqdm import tqdm


def dowload_MS_COCO_raw_data(download_folder=None):
    """
    data from http://cocodataset.org/#home
    :return: annotation_file (json file path) ...captions_train2014.json
            img_file_dir (raw image file folder) ... train2014
    """
    if download_folder is not None and os.path.exists(download_folder):
        annotation_file = download_folder + '/annotations/captions_train2014.json'
        img_file_dir = download_folder + '/train2014'
        return annotation_file, img_file_dir

    annotation_zip = tf.keras.utils.get_file('captions.zip',
                                             cache_subdir=os.path.abspath('.'),
                                             origin='http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
                                             extract=True)
    annotation_file = os.path.dirname(annotation_zip) + '/annotations/captions_train2014.json'

    name_of_zip = 'train2014.zip'
    if not os.path.exists(os.path.abspath('.') + '/' + name_of_zip):
        image_zip = tf.keras.utils.get_file(name_of_zip,
                                            cache_subdir=os.path.abspath('.'),
                                            origin='http://images.cocodataset.org/zips/train2014.zip',
                                            extract=True)
        img_file_dir = os.path.dirname(image_zip) + '/train2014/'
    else:
        img_file_dir = os.path.abspath('.') + '/train2014/'

    return annotation_file, img_file_dir


def read_raw_image_and_caption_file(annotation_file, img_file_dir, num_examples=None):
    # Read the json file
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    # Store captions and image names in vectors
    all_captions = []
    all_img_name_vector = []

    for annot in annotations['annotations']:
        caption = '<start> ' + annot['caption'] + ' <end>'
        image_id = annot['image_id']
        full_coco_image_path = os.path.join(img_file_dir, 'COCO_train2014_' + '%012d.jpg' % (image_id))
        all_img_name_vector.append(full_coco_image_path)
        all_captions.append(caption)

    # Shuffle captions and image_names together
    # Set a random state
    train_captions, img_name_vector = shuffle(all_captions, all_img_name_vector, random_state=1)

    # Select the first num_examples captions from the shuffled set, None for all data
    train_captions = train_captions[:num_examples]
    img_name_vector = img_name_vector[:num_examples]

    print(f"train_captions numbers {len(train_captions)}\t img_name_vector numbers {len(img_name_vector)}")
    return train_captions, img_name_vector


def initialize_InceptionV3_image_features_extract_model():
    """
    :return: InceptionV3 model instance
    """
    image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                    weights='imagenet')
    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output  # shape [batch_size, 8, 8, 2048]

    image_features_extract_model = tf.keras.Model(new_input, hidden_layer)
    return image_features_extract_model


def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path


def caching_image_features_extracted_from_InceptionV3(img_name_vector, cache_image_dir="cache_image"):
    """
    :param img_name_vector: image file path list
    :param cache_image_dir: folder of store image features extracted from Inception V3
    :return:
    """
    if not os.path.exists(cache_image_dir):
        os.mkdir(cache_image_dir)

    # Get unique images
    encode_train = sorted(set(img_name_vector))

    # Feel free to change batch_size according to your system configuration
    image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
    image_dataset = image_dataset.map(
        load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(16)

    image_features_extract_model = initialize_InceptionV3_image_features_extract_model()

    for img, path in tqdm(image_dataset):
        # shape [batch_size, 8, 8 ,2048]
        batch_features = image_features_extract_model(img)
        # shape [batch_size, 64, 2048]
        batch_features = tf.reshape(batch_features, (batch_features.shape[0], -1, batch_features.shape[3]))

        for batch_features, img_path in zip(batch_features, path):
            path_of_feature = img_path.numpy().decode("utf-8")
            path_of_feature_file_name = os.path.basename(path_of_feature)
            save_file_path_of_feature = os.path.join(cache_image_dir, path_of_feature_file_name)

            np.save(save_file_path_of_feature, batch_features.numpy())


def preprocess_and_tokenize_captions(train_captions, top_k):
    """
    :param train_captions: captions list
    :param top_k: limit the vocabulary size to the top k words (to save memory)
    :return: caption_vector (token and pad)
    """

    # Find the maximum length of any caption in our dataset
    def calc_max_length(tensor):
        return max(len(t) for t in tensor)

    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k, oov_token="<unk>",
                                                      filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
    tokenizer.fit_on_texts(train_captions)

    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'

    with open("caption_tokenizer", "wb") as f:
        pickle.dump(tokenizer, f)

    # Create the tokenized vectors
    train_seqs = tokenizer.texts_to_sequences(train_captions)

    # Pad each vector to the max_length of the captions
    # If you do not provide a max_length value, pad_sequences calculates it automatically
    caption_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')

    # Calculates the max_length, which is used to store the attention weights
    max_length = calc_max_length(train_seqs)
    print(f"max_length:\t{max_length}")

    return caption_vector


def split_and_save_raw_image_path_and_caption_content(img_name_vector, caption_vector, test_size,
                                                      raw_image_path_and_caption_content):
    save_dir_name = raw_image_path_and_caption_content
    if not os.path.exists(save_dir_name):
        os.mkdir(save_dir_name)
    # Create training and validation sets using an 80-20 split
    img_name_train, img_name_val, cap_train, cap_val = train_test_split(img_name_vector, caption_vector,
                                                                        test_size=test_size, random_state=0)

    print(f"img_name_train numbers:\t{len(img_name_train)}\tcap_train numbers:\t{len(cap_train)}\n"
          f"img_name_val numbers:\t {len(img_name_val)}\tcap_val numbers:\t{len(cap_val)}")

    np.save(save_dir_name + '/img_name_train.npy', img_name_train)
    np.save(save_dir_name + '/img_name_val.npy', img_name_val)
    np.save(save_dir_name + '/cap_train.npy', cap_train)
    np.save(save_dir_name + '/cap_val.npy', cap_val)
    print(f"Prepared file save to {save_dir_name}")


def get_caption_tokenizer(caption_tokenizer_path="caption_tokenizer"):
    with open(caption_tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
        return tokenizer


def load_raw_image_path_and_caption_content(save_dir_name="raw_image_path_and_caption_content"):
    if not os.path.exists(save_dir_name):
        raise ValueError(f"Not found {save_dir_name}, please first run prepare_MS-COCO_data.py")
    img_name_train = np.load(save_dir_name + '/img_name_train.npy')
    img_name_val = np.load(save_dir_name + '/img_name_val.npy')
    cap_train = np.load(save_dir_name + '/cap_train.npy')
    cap_val = np.load(save_dir_name + '/cap_val.npy')
    return img_name_train, img_name_val, cap_train, cap_val


def main(download_folder=None, num_examples=None, top_k=5000, test_size=0.2,
         raw_image_path_and_caption_content="raw_image_path_and_caption_content", cache_image_dir="cache_image"):
    """
    :param download_folder: str, None->Automatic download of files
    :param num_examples: int, Select the first 30000 captions from the shuffled set, None for all data
    :param top_k:  int, Choose the top 5000 words from the vocabulary
    :param test_size: float, test data number : train data number = test_size : 1
    :param cache_image_path_and_tokened_caption_file_dir: str, Store processed file paths
    :param cache_image_dir: str, Store processed image file dir
    :return:
    """
    # Download files
    annotation_file, img_file_dir = dowload_MS_COCO_raw_data(download_folder=download_folder)

    # read_raw_image_and_caption_file
    train_captions, img_name_vector = read_raw_image_and_caption_file(annotation_file, img_file_dir,
                                                                      num_examples=num_examples)

    # Preprocess the images and captions
    caption_vector = preprocess_and_tokenize_captions(train_captions, top_k)
    caching_image_features_extracted_from_InceptionV3(img_name_vector, cache_image_dir=cache_image_dir)

    # split_and_save file
    split_and_save_raw_image_path_and_caption_content(img_name_vector, caption_vector, test_size=test_size,
                                                      raw_image_path_and_caption_content=raw_image_path_and_caption_content)


if __name__ == "__main__":
    download_folder = "/home/b418a/disk1/pycharm_room/yuanxiao/my_lenovo_P50s/Image_captioning"
    raw_image_path_and_caption_content_dir = "raw_image_path_and_caption_content"
    cache_image_dir = "cache_image"
    main(download_folder=download_folder, num_examples=10000, top_k=3000, test_size=0.2,
         raw_image_path_and_caption_content=raw_image_path_and_caption_content_dir,
         cache_image_dir=cache_image_dir)
