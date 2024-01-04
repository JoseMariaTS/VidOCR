# VidOCR

**Introduction:**

To conclude my studies in the Master's Degree in Industrial Engineering at the University of Seville, the following Master's Thesis has been developed: [Reconocimiento y clasificación de imágenes utilizando técnicas de aprendizaje profundo (Deep Learning)](https://hdl.handle.net/11441/151661). This work was originally conducted in Spanish, as seen in the previous "link". Nevertheless, the codes required to replicate the obtained results have been translated, and they are presented in this GitHub repository.

The objective of the Master's Thesis is to develop an application that, through the webcam, allows for text detection in images and recognition to display it on the screen. The idea is for the final application to function as a video OCR where the text is of the "scene text" or "text in the wild" type, inspired by what can be achieved with [keras-ocr](https://keras-ocr.readthedocs.io/en/latest/) or [Google Lens](https://lens.google/intl/es/).

![Detection, recognition, and correction of the title of the book "Mundo Anillo"](/FinalApp/Result.PNG)

o simplify the task, the final application will be divided into two algorithms: one capable of detecting text in the image and another capable of recognizing the detected text. Both algorithms will be based on neural networks (Deep Learning) and will be designed/trained using [TensorFlow](https://www.tensorflow.org/?hl=es-419). Although the codes are intended to run on a portable computer, the training requires significant computational power; therefore, they have been carried out in [Google Colab Pro](https://colab.google/) to leverage a powerful GPU. These codes are programmed in [Python 3.9.7](https://www.python.org/psf-landing/) using [JupyterLab 3.2.1](https://jupyter.org/). The Python libraries used are as follows:

```python
import os #base
import glob #base
import csv #base
import time #base
import numpy as np #v1.20.3
import pandas as pd #v1.3.4
import tensorflow as tf #v2.7.0
import matplotlib.pyplot as plt #v3.4.3
import cv2 #v4.5.4.60 (opencv-python)
import shapely as shap #v2.0.1
import enchant #v3.2.2 (pyenchant)
```

**[Text detection](/TextDetection):**

Text detection in the image will be performed by an algorithm composed of a fully convolutional neural network and a post-processing system. This algorithm is known as [EASTlite](https://github.com/bittorala/east_lite), a lighter version of [EAST:An Efficient and Accurate Scene Text Detector](https://github.com/zxytim/EAST). This algorithm will receive the image and return the coordinates of oriented bounding boxes, highlighting the areas of the image that contain text.

The neural network training for the detection algorithm was carried out using the images from ICDAR2015 (complete) and ICDAR2013 (only images from the training set), which can be found at the following [link](https://rrc.cvc.uab.es/). To adapt the images and annotations from ICDAR2013 to the format of ICDAR2015, the file [Sp0_DatasetsAdministration.ipynb](/TextDetection/Sp0_DatasetsAdministration.ipynb) was used.

The codes in the file [Sp1_Visulization_and_Preprocessing.ipynb](/TextDetection/Sp1_Visulization_and_Preprocessing.ipynb) allow visualizing the operation of the EASTlite algorithm and the process of training its neural network. The file [Sp2_ModelCreation_and_Training.ipynb](/TextDetection/Sp2_ModelCreation_and_Training.ipynb) enables the training and evaluation of the neural network, using the function file [TextDetection_Functions.py](/TextDetection/TextDetection_Functions.py).

In the mentioned function file, reference is made to the [lamns](https://github.com/argman/EAST/tree/master/lanms) package, which is a "Non-maximum Suppression (NMS)" algorithm compiled in C++. This package makes the post-processing of EASTlite and EAST very fast but apparently shows problems on Windows. My solution has been to bypass it and create my own NMS algorithm programmed in Python, which unfortunately takes longer to apply the post-processing but does not have compatibility issues.

**[Text recognition](/TextRecognition):**

PFor text recognition, the simplest version has been chosen, consisting of a convolutional neural network that classifies the characters of the detected words and determines which character it refers to. As can be seen, two versions have been designed: one trained with the [balanced EMNIST dataset](https://www.kaggle.com/datasets/crawford/emnist) and the other trained with the [Chars74k dataset](http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k) (only the partition of synthetic English characters found at this [link](https://www.kaggle.com/datasets/supreethrao/chars74kdigitalenglishfont)).

Using one dataset or the other involves changing the structure of the neural network by changing the number of classes. However, regardless of the chosen version, the file **Sp0_DataAdministration_and_Visualization.ipynb** allows visualizing the dataset, and the file **Sp1_ModelCreation_and_Training.ipynb** allows designing/training/evaluating the chosen neural network.

**[Final app](/FinalApp):**

The final step is to merge the text detection algorithm with the text recognition algorithm, adapting the output of one to the input of the other. For this union, it will be necessary to organize the words returned by the detector and segment the characters using thresholding techniques. Finally, once each character is recognized, the words must be formed and corrected orthographically to resolve those misclassified characters. All this is done in the file [TextDetection_and_Recognition.ipynb](/FinalApp/TextDetection_and_Recognition.ipynb) and in the function file [TextDetection_and_Recognition_Functions.py](/FinalApp/TextDetection_and_Recognition_Functions.py).


**Final Invitation and Possible Enhancements:**

I invite you to explore and utilize this repository for your projects and applications. Your feedback and contributions are highly valued. If you find any issues or have ideas for improvements, please feel free to open an issue or submit a pull request. Thank you for your interest and collaboration!!!
