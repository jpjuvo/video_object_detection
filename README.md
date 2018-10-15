# TensorFlow object detection on a video

This is a python implementation for doing object detection inference on a video. All required **TensorFlow/models/research/object_detection** utilities are repackaged here so you can get started quickly.

![Example video](/cars.gif)

*Note!* Currently, only *.avi video files are supported.

## Dependencies

- Python 3.5 or 3.6 (TensorFlow does not support 3.7 - Aug 2018)
- TensorFlow, preferably 1.9+
- Pillow python package
- OpenCV python package

You can install the required packages with pip.
```pip install tensorflow pillow opencv```

## Usage

```python object_detection.py --video="myVideo.avi" --fps=24 --out_video="outputVideo.avi"```

or

```python object_detection.py --video="myVideo.avi" --fps=24 --out_video="outputVideo.avi" --model="ssdlite_mobilenet_v2_coco_2018_05_09"```

## Modifications to object_detection repository

I made some minor modifications to original distributed TensorFlow files to get this repository to work independently without the full copy of tensorflow models repository.

## Model

This repository is distributed with the **ssdlite_mobilenet_v2_coco_2018_05_09** model that is suited for fast and lightweight inference.
If you want to try something else, you can get another coco-trained model from the [model-zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md).

*Note!* Only cooc-trained models are currently supported.

Unzip the model folder to a /model_name folder here and make sure the **frozen_inference_graph.pb** is found on the **/model_name/frozen_inference_graph.pb**.
Use the ```--model="model_name"``` flag to select the inference model.

## License

[Apache License 2.0](LICENSE)
