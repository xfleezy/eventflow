# Optical flow for event camera
A lightweight network to learn event-based optical flow. (ICPR 2020)

## Introduction
In this work, we present a lightweight pyramid network to learn event-based flow in a self-supervised way. We combine the
pyramidal processing and channel attention mechanism into the proposed network, which significantly simplifies network
but reaches the comparable results over state-of-the-arts.


## Prerequistes
* Pytorch 1.0.1
* Numpy
* Opencv
* Matplotlib

## Training
Before the training process, you need to download datasets from https://daniilidis-group.github.io/mvsec/. The data takes up to about 30 GB. For efficient data reading, you need to convert data into event images and grayscale images then save them in the ./data.

To train a new model, run 
```python main.py```
![pic1](/media/xusper/KESU/learning/Learn Optical Flow from Event Camera/177_flow.jpg)

## Video sample
Video sample contains continuous estimation of event-based optical flow for outdoor_day, indoor_day and outdoor_night scenes.


## Authors
Zhuoyan Li, Jiawei Shen
