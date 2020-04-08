from PIL import Image
import numpy as np
import os

from keras.models import load_model

import matplotlib.pyplot as plt

from datetime import datetime


def label_img(labels, PDB):
    """ Priradi, ake poradove cislo ma trieda so zadanym PDB na zaklade pola labels.

    :param labels: oznacenia tried, ktore su mozne pre nacitany obrazok a su v poradi v akom do nich klasifikuje zadana siet
    :type labels: list
    :param PDB: oznacenie obrazka podla, ktoreho mu vieme dat, ze do triedy s akym poradovym cislom patri.
    :type PDB: str
    :return: poradove cislo triedy, do ktorej by mal obrazok patrit pre danu siet
    :rtype: int
    """
    return labels.index(PDB)


def load_image_label(path, img_size, labels=('4QEI', '1MFQ', '1E7K', '1A9N', '1E8O')):
    """ Nacita obrazok na zadanej ceste, o ktorom predpokladame, ze ma 16-bitovu farebnu hlbku,
    normalizuje ho na hodnoty od 0 po 1 a vrati aj do akej triedy ma patrit na zaklade jeho nazvu
    a zadanych moznych oznaceni obrazkov (labels), ktore su v poradi v akom do nich klasifikuje zadana siet

    :param path: cesta k obrazku, ktory chceme nacitat, predpokladame, ze ma 16-bitovu farebnu hlbku
    :type path: str
    :param img_size: rozmer aky ma mat obrazok po nacitani, upravi sa na stvorcovy tvar podla zadaneho rozmeru
    :type img_size: int
    :param labels: oznacenia tried, ktore su mozne pre nacitany obrazok a su v poradi v akom do nich klasifikuje zadana siet
    :type labels: list
    :return: image_label pole kde je nacitany obrazok spolu s jeho zaradenim do triedy vo forme one-hot
        image_class poradove cislo triedy, do ktorej obrazok patri
        full_image_name nazov suboru v ktorom bol obrazok
    :rtype: list
        int
        str
    """
    # zistenie nazvu obrazka
    base = os.path.basename(path)
    full_image_name = os.path.splitext(base)[0]

    # zistenie triedy obrazka na zaklade zaciatocneho PDB
    PDB = full_image_name.split('_')[0]
    image_class = label_img(labels, PDB)

    # pocet tried je pocet moznych oznaceni
    count_classes = len(labels)

    # na poziciu triedy obrazka dame 1
    label = np.zeros(count_classes, dtype=int)
    label[image_class] = 1

    # premenna na obrazok a jeho label
    image_label = []

    img = np.array(Image.open(path).resize((img_size, img_size), Image.ANTIALIAS))
    img_norm = img / 65535
    image_label.append([np.array(img_norm), label])

    return image_label, image_class, full_image_name


def iter_occlusion(image, size=2):
    """ Postupne vracia upravene obrazky vytvorene zo zadaneho obrazku pomocou vynulovania stvorcovej oblasti o zadanom rozmere.
    Vynulovana oblast je pri kazdom dalsom vratenom obrazku systematicky posuvana o 1 pixel.

    :param image: obrazok, na zaklade ktoreho sa vytvaraju vystupne obrazky
    :type image: np.array
    :param size: rozmer strany stvorca nulovanej oblasti
    :type size: int
    :return:i 0-tá súradnica nulovanej oblasti
        j 1-tá súradnica nulovanej oblasti
        occlusion_image obrazok vytvoreny na zaklade vstupneho obrazku, pomocou vynulovania jeho jednej
            stvorcovej oblasti o rozmere stvorca so stranou size
    :rtype: int
        int
        np.array
    """
    occlusion = np.full((size, size, 1), 0, np.float64)

    for i in range(image.shape[0] - size):
        for j in range(image.shape[1] - size):
            occlusion_image = image.copy()

            occlusion_image[i:i + size, j:j + size] = occlusion
            yield i, j, occlusion_image


def predict(model, test_image, image_class):
    """ Vrati pre zadanu naucenu siet (model) a zadany obrazok, ze z akou pravdepodobnostou by dany model zaradil obrazok
    do jeho spravnej triedy a to, ze aku triedu by obrazku siet predpovedala.

    :param model: nauceny model siete
    :type model: keras.models.Sequential
    :param test_image: obrazok ktory chceme dat ako vstup naucenej sieti
    :type test_image: np.array
    :param image_class: poradove cislo triedy, do ktorej obrazok patri pre zadanu naucenu siet
    :type image_class: int
    :return: probability pravdepodobnost z akou siet klasifikuje zadany obrazok do jeho spravnej triedy image_class
        class_prediction trieda, ktoru zadana siet predpovedala, ze do nej zadany obrazok patri
    :rtype: float
        int
    """
    y_predict_probability = model.predict_proba(test_image)

    probability = y_predict_probability[0][image_class]
    class_prediction = np.argmax(y_predict_probability)

    return probability, class_prediction


def timestamp():
    dateTimeObj = datetime.now()
    return dateTimeObj.strftime("%m%d%H%M")


def occlusion_sensitivity(path_model, path_image, occlusion_size, labels=('4QEI', '1MFQ', '1E7K', '1A9N', '1E8O')):
    """ Zeiler, M. D., and Fergus, R. Visualizing and understanding convolutional networks, 2013.
    https://arxiv.org/abs/1311.2901.

    Vizualizácia toho, že ktorá časť vstupneho obrázku je pre siet najdolezitejsia.
    Pouzity postup je inspirovany pracou Zeiler a Fergus, v ktorej ho pomenovali Occlusion sensitivity.
    Pri tomto pristupe su postupne systematicky s posunom 1 nulované štvorcové oblasti vstupneho obrazka
    o rozmere, ktory je zadany parametrom occlusion_sizea kazdy takto upraveny
    obrazok sme vzdy dali na vstup naucenej sieti. Siet nam pre kazdy obrazok s takto vynulovanou
    oblastou dala pravdepodobnost pre kazdu naucenu triedu. My sme z tychto
    pravdepodobnosti sledovali len pravdepodobnost pre triedu, do ktorej tento difraktogram
    realne patri. Tieto predpovedane hodnoty pravdepodobnosti sme si vykreslili
    vo forme obrazku (heatmap).

    :param path_model: cesta k naucenemu modelu siete ulozenej v subore formatu HDF5, ktorej feature maps vizualizujeme
    :type path_model: str
    :param path_image: cesta k obrazku, pre ktory vykreslime feature maps, predpokladame, ze ma 16-bitovu farebnu hlbku
    :type path_image: str
    :param occlusion_size: rozmer strany stvorca nulovanej oblasti
    :type occlusion_size: int
    :param labels: oznacenia tried, ktore su mozne pre nacitany obrazok a su v poradi v akom do nich klasifikuje zadana siet
    :type labels: list
    """
    img_size = 256

    # nacitanie modelu
    model = load_model(path_model, compile=False)  # compile=False nemusi byt

    image_label, image_class, full_image_name = load_image_label(path_image, img_size, labels)

    test_image = np.array([i[0] for i in image_label]).reshape(-1, img_size, img_size, 1)
    # testLabels = np.array([i[1] for i in image_label])

    heatmap = np.zeros((img_size - occlusion_size, img_size - occlusion_size), np.float64)

    image = test_image[0].copy()

    probability_original_image, class_prediction_original_image = predict(model, test_image, image_class)
    print('class:', image_class, 'prob:', probability_original_image, 'class predicted:',
          class_prediction_original_image)
    print('softmax:', model.predict_proba(test_image))

    for i, j, occlusion_image in iter_occlusion(image, size=occlusion_size):
        X = occlusion_image.reshape(1, img_size, img_size, 1)
        probability, class_prediction = predict(model, X, image_class)

        heatmap[i, j] = probability

    fig = plt.figure()
    plt.imshow(heatmap, cmap='jet')
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    plt.savefig('heatmap_C' + str(image_class) + '_oc' + str(
        occlusion_size) + '_' + full_image_name + '_' + timestamp() + '.png')
    plt.show()

    print('----------------------------------------')
    for i in range(heatmap.shape[0]):
        print([heatmap[i][j] for j in range(heatmap.shape[1])])
    print('----------------------------------------')
