# Image-Captioning
Template code for image caption or image to text (TensorFlow version 2). 图片描述或者图片生成文本的模板代码

## Task Description 
Given an image like the example below, our goal is to generate a caption such as "a surfer riding on a wave".

![Man Surfing](https://tensorflow.org/images/surf.jpg)

To accomplish this, you'll use an attention-based model, which enables us to see what parts of the image the model focuses on as it generates a caption.

![Prediction](https://tensorflow.org/images/imcap_prediction.png)

The model architecture is similar to [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/abs/1502.03044).

## Code test pass
+ Pyhon 3.6
+ TensorFlow version 2

## Usage

### Preparing data

```
python data_utils.py
```

**Manual download of data**
If the code can't download the data automatically because of network reasons, you can download the data manually.

1. Downloading captions data from http://images.cocodataset.org/annotations/annotations_trainval2014.zip
2. unzip annotations_trainval2014.zip and move annotations to project
3. Downloading images data from http://images.cocodataset.org/zips/train2014.zip
4. unzip train2014.zip and move train2014 to project

### Train model

```
python train_image_caption_model.py
```

### Model predicte

```
python predicte_image_caption.py
```


## Reference Code
> [image_captioning.ipynb](https://github.com/tensorflow/docs/blob/master/site/en/r2/tutorials/text/image_captioning.ipynb)

This notebook is an end-to-end example. When you run the notebook, it downloads the [MS-COCO](http://cocodataset.org/#home) dataset, preprocesses and caches a subset of images using Inception V3, trains an encoder-decoder model, and generates captions on new images using the trained model.

In this example, you will train a model on a relatively small amount of data—the first 30,000 captions  for about 20,000 images (because there are multiple captions per image in the dataset).


## Learn more

|Title|Content|
|-|-|
|[awesome-image-captioning](https://github.com/zhjohnchan/awesome-image-captioning)|A curated list of image captioning and related area resources.|
