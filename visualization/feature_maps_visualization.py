# Inspiration: https://machinelearningmastery.com/how-to-visualize-filters-and-feature-maps-in-convolutional-neural-networks/

from PIL import Image
import numpy as np
import os

from keras.models import load_model, Model

import matplotlib.pyplot as plt

from datetime import datetime


def load_image(path, img_size):
    """ Nacita obrazok o ktorom predpokladame, ze ma 16-bitovu farebnu hlbku, upravi ho na zadany rozmer img_size a normalizuje ho.

    :param path: cesta k nacitavanemu obrazku
    :type path: str
    :param img_size: rozmer aky ma mat obrazok po nacitani, upravi sa na stvorcovy tvar podla zadaneho rozmeru
    :type img_size: int
    :return image nacitany a upraveny obrazok
    :rtype list
    """
    image = []

    img = np.array(Image.open(path).resize((img_size, img_size), Image.ANTIALIAS))
    img_norm = img / 65535
    image.append([np.array(img_norm)])

    return image


def timestamp():
    dateTimeObj = datetime.now()
    return dateTimeObj.strftime("%m%d%H%M")


def feature_maps_visualization(path_model, path_image):
    """ Vykresli feature maps pre kazdu konvolucnu vrstvu pre zadanu siet, ktora dostane na vstup zadany obrazok.
    O zadanej sieti predpokladame, ze ma 6 konvolucnych vrstiev s poctamy konvolucnych jadier na jednotlivych vrstvach,
    v poradi od prvej konvolucnej vrstvy, 8, 16, 16, 32, 32, 64.

    :param path_model: cesta k naucenemu modelu siete ulozenej v subore formatu HDF5, ktorej feature maps vizualizujeme
    :type path_model: str
    :param path_image: cesta k obrazku, pre ktory vykreslime feature maps, predpokladame, ze ma 16-bitovu farebnu hlbku
    :type path_image: str
    """
    img_size = 256

    base = os.path.basename(path_image)
    name = os.path.splitext(base)[0]

    image = load_image(path_image, img_size)

    testImage = np.array([i[0] for i in image]).reshape(-1, img_size, img_size, 1)

    # nacitanie modelu
    model = load_model(path_model, compile=False)  # compile=False nemusi byt
    model.summary()

    outputs = []
    for i in range(len(model.layers)):
        layer = model.layers[i]
        # vyberieme len konvolucne vrstvy
        if 'conv' not in layer.name:
            continue

        outputs.append(model.layers[i].output)

    model = Model(inputs=model.inputs, outputs=outputs)

    feature_maps = model.predict(testImage)

    i = 1
    for fmap in feature_maps:

        if i == 1:
            fig = plt.figure(figsize=(8, 4))
            nrows, ncols = 2, 4
        elif i == 2 or i == 3:
            fig = plt.figure(figsize=(8, 8))
            nrows, ncols = 4, 4
        elif i == 4 or i == 5:
            fig = plt.figure(figsize=(16, 8))
            nrows, ncols = 4, 8
        else:
            fig = plt.figure(figsize=(16, 16))
            nrows, ncols = 8, 8

        ix = 1
        for _ in range(ncols):
            for _ in range(nrows):
                # specify subplot and turn of axis
                ax = plt.subplot(nrows, ncols, ix)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_aspect('equal')
                plt.subplots_adjust(wspace=0.05, hspace=0.05)
                # plot filter channel in grayscale
                plt.imshow(fmap[0, :, :, ix - 1], cmap='gray')
                ix += 1

        plt.savefig('aktivacie' + str(i) + '_' + name + '_' + timestamp() + '.png')
        # show the figure
        plt.show()
        i += 1
