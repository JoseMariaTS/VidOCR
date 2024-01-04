# VidOCR
**Introduction:**

Para concluir mis estudios del Máster Universitario en Ingeniería Industrial en la Universidad de Sevilla se ha realizado el siguiente Trabajo Fin de Máster: [Reconocimiento y clasificación de imágenes utilizando técnicas de aprendizaje profundo (Deep Learning)](https://hdl.handle.net/11441/151661). Este Trabajo originalmente se realizó en español, como se puede ver en en el link anterior. Aun así, los códigos necesarios para replicar los resultados obtendios se han traducido y son los expuestos en este repositorio de GitHub.

El objetivo del Trabajo Fin de Máster es desarrollar una aplicación que, a través de la webcam, permita
detectar texto en la imagen y reconocerlo para escribirlo por pantalla. La idea es que la aplicación final funcione como un OCR para video donde el texto sea de tipo "scene text" o "text in the wild", inspirando en lo que se puede conseguir con [keras-ocr](https://keras-ocr.readthedocs.io/en/latest/) o [Google Lens](https://lens.google/intl/es/).

![Detección, reconocmiento y corrección del título del libro Mundo Anillo](/FinalApp/Result.PNG)



es diseñar y entrenar una aplicación capaz de detectar y reconocer texto en vídeo de forma automática. Para sinplificar esta tarea se dibidido la aplicación final en dos algoritmos: uno capzas de detectar el texto en la imagen y otro capaz de reconocer el texto detectado.



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

**Text detection:**


**Text recogntion:**


**Final app**


