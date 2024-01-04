# VidOCR

**Introduction:**

Para concluir mis estudios del Máster Universitario en Ingeniería Industrial en la Universidad de Sevilla se ha realizado el siguiente Trabajo Fin de Máster: [Reconocimiento y clasificación de imágenes utilizando técnicas de aprendizaje profundo (Deep Learning)](https://hdl.handle.net/11441/151661). Este Trabajo originalmente se realizó en español, como se puede ver en en el link anterior. Aun así, los códigos necesarios para replicar los resultados obtendios se han traducido y son los expuestos en este repositorio de GitHub.

El objetivo del Trabajo Fin de Máster es desarrollar una aplicación que, a través de la webcam, permita detectar texto en la imagen y reconocerlo para escribirlo por pantalla. La idea es que la aplicación final funcione como un OCR para video donde el texto sea de tipo "scene text" o "text in the wild", inspirando en lo que se puede conseguir con [keras-ocr](https://keras-ocr.readthedocs.io/en/latest/) o [Google Lens](https://lens.google/intl/es/).

![Detección, reconocmiento y corrección del título del libro Mundo Anillo](/FinalApp/Result.PNG)

Para simplificar la tarea se dibidirá la aplicación final en dos algoritmos: uno capaz de detectar el texto en la imagen y otro capaz de reconocer el texto detectado. Ambos algoritmos estarán basados en redes neuronales (Deep Learning) y se diseñaran/entrenarán haciendo uso de [TensorFlow](https://www.tensorflow.org/?hl=es-419). Aunque los códigos están pesanados para correr en un ordenado portatil, los entrenamientos requieren de mucha capacidad computación y, por esos, se han realizado en [Google Colab Pro](https://colab.google/) para disponer de una GPU potente. Estos códigos están programados en [Python 3.9.7](https://www.python.org/psf-landing/) usando [JupyterLab 3.2.1](https://jupyter.org/). La liberías de Python usadas son las siguientes:

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

La detección de texto en la imagen será realizada por una algoritmo compuesto por una red neuronal completamente convolucional y un sistema de posprocesado. Este algoritmo se conoce como [EASTlite](https://github.com/bittorala/east_lite), una versión más ligera de [EAST:An Efficient and Accurate Scene Text Detector](https://github.com/zxytim/EAST). Este algoritmo recibirá la imagen y devolverá las coordenadas de unas cajas delimitadoras o “bounding boxes” orientadas, que destacará las zonas de la imagen que tienen texto.

El entrenamiento de la red neuronal del algoritmo de detección se realizó con las imágenes del ICDAR2015 (completo) e ICDAR2013 (sólo las imágenes del set de entrenamiento), las cuales se puede encontrar en la dirección en siguiente [link](https://rrc.cvc.uab.es/). Para adaptar las imagenes ya notaciones de ICDAR2013 al formato de ICDAR2015, se hizo uso del archivo: [Sp0_DatasetsAdministration.ipynb](/TextDetection/Sp0_DatasetsAdministration.ipynb).

Los códigos del archivo [Sp1_Visulization_and_Preprocessing.ipynb](/TextDetection/Sp1_Visulization_and_Preprocessing.ipynb) permite visualizar el funcionamiento del algortmo de EASTlite y la forma de entrenar su red neuronal. El archivo [Sp2_ModelCreation_and_Training.ipynb](/TextDetection/Sp2_ModelCreation_and_Training.ipynb) permite entrenar y evaluar la red neuronal, haciendo uso del archivo de funciones [TextDetection_Functions.py](/TextDetection/TextDetection_Functions.py).

En el mecnionado archivo de función se hace referencia al paquete [lamns](https://github.com/argman/EAST/tree/master/lanms), el cual es un algoritmo de "Non-maximum Suppression (NMS)" compilado en C++. Este paquete hace que el posprocesado de EASTlite y EAST sea muy veloz, pero aparentemente muestra porblemas en Windows. Mi solución ha sido obviarlo y crear mi propio algoritmo de NMS programado en Python, que desgraciadamente tarda más en aplicar el prospocesado, pero sin dar problemas de copatibilidad.

**[Text recogntion](/TextRecognition):**

Para el reconocimiento de texto se ha optado por la versión más sencilla que consite en una red neurona convolucional que clasifique los caracteres de las palabras detectadas y dertermine a que caracter se refiere.Como se puede ver, se han diseñado dos versiones: una que se ha entrenado con el dataset de [EMNIST balanceado](https://www.kaggle.com/datasets/crawford/emnist) y el otro que se ha entrenado con el dataset [Chars74k](http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k) (sólo la partición de los caracteres siteticos en ingles que se enconetra en este [link](https://www.kaggle.com/datasets/supreethrao/chars74kdigitalenglishfont)).

El usar uno u otro dataset supone cambiar la estructra de la red neuronal al cambiar el número de clases. Aun así, indeoendientemente de la versión es cogida, el archivo **Sp0_DataAdministration_and_Visualization.ipynb** permite visualizar el datset y el archivo **Sp1_ModelCreation_and_Training.ipynb** permite diseñar/entrenar/evaluar la red neuronal escogida.

**[Final app](/FinalApp):**

El paso final es juntar el agoritmo de detección con el algoritmo de reconocimiento, adaptando la salida de uno a la entrada del otro. Para est unión será necesario ordenar las palabras que devuelve el detector y segmentar los caractetres con técnicas de umbralización. Finalmente, una vez que se reconozca cada caracter, se debe formar las palabras y corregirlas ortograficamente para solucionar aquellos caracteres mal clasificados. Todo esto se realiza en el archivo [TextDetection_and_Recognition.ipynb](/FinalApp/TextDetection_and_Recognition.ipynb) y en el archivo de función [TextDetection_and_Recognition_Functions.py](/FinalApp/TextDetection_and_Recognition_Functions.py).

