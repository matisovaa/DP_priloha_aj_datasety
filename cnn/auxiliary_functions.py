import os
import numpy as np

from PIL import Image
from random import shuffle
from datetime import datetime


def load_train_data(dataset_directory, img_size):
    """ Vrati z datasetu pole s nacitanymi trenovacimi obrazkami a ich priradeniami do prislusnych tried
    vo forme one-hot vectora, ktore je vo vhodnom formte aby mohlo sluzit ako vstup pre neuronovu siet.

    :param dataset_directory: cesta k datasetu, z ktoreho maju byt nacitane obrazky, predpokladame, ze maju 16-bitovu farebnu hlbku
    :type dataset_directory: str
    :param img_size: rozmer aky maju mat obrazky po nacitani, upravia sa na stvorcovy tvar podla zadaneho rozmeru
    :type img_size: int
    :return train_data pole kde je na kazdom indexe zaznam o jednom obrazku pre trening, konkretne prislusny
            obrazok normovany na interval [0, 1] a jeho zaradenie do triedy vo forme one-hot vectora
        count_classes pocet tried na trenovanie, ktory sa zisti podla poctu adresarov s obrazkami
    :rtype list
        int
    """
    train_data = []

    # pocet tried sa rovna poctu podpriecinkov    
    path_dir = os.path.join(dataset_directory, 'train')

    count_classes = len(os.listdir(path_dir))
    # pocitadlo ze ktora je aktualne trieda (pre vyrabanie one-hot vectora pre triedu)
    image_class = 0
    # prechadzame jednotlive adresare tried
    for class_directory in os.listdir(path_dir):
        # vyrobenie one-hot vectora pre triedu
        label = np.zeros(count_classes, dtype=int)
        label[image_class] = 1

        path_img_dict = os.path.join(path_dir, class_directory)
        for img in os.listdir(path_img_dict):
            path = os.path.join(path_img_dict, img)
            img = np.array(Image.open(path).resize((img_size, img_size), Image.ANTIALIAS))
            img_norm = img / 65535
            train_data.append([np.array(img_norm), label])

        image_class += 1

    shuffle(train_data)
    return train_data, count_classes


def load_test_data(dataset_directory, img_size):
    """ Vrati z datasetu pole s nacitanymi testvacimi obrazkami a ich priradeniami do prislusnych tried
    vo forme one-hot vectora, ktore je vo vhodnom formte aby mohlo sluzit ako vstup pre neuronovu siet.

    :param dataset_directory: cesta k datasetu, z ktoreho maju byt nacitane obrazky, predpokladame, ze maju 16-bitovu farebnu hlbku
    :type dataset_directory: str
    :param img_size: rozmer aky maju mat obrazky po nacitani, upravia sa na stvorcovy tvar podla zadaneho rozmeru
    :type img_size: int
    :return test_data pole kde je na kazdom indexe zaznam o jednom obrazku pre testovanie, konkretne prislusny
            obrazok normovany na interval [0, 1] a jeho zaradenie do triedy vo forme one-hot vectora
    :rtype list
    """
    test_data = []

    # pocet tried sa rovna poctu podpriecinkov    
    path_dir = os.path.join(dataset_directory, 'test')
    count_classes = len(os.listdir(path_dir))
    # pocitadlo ze ktora je aktualne trieda (pre vyrabanie one-hot vectora pre triedu)
    image_class = 0
    # prechadzame jednotlive adresare tried
    for class_directory in os.listdir(path_dir):
        # vyrobenie one-hot vectora pre triedu
        label = np.zeros(count_classes, dtype=int)
        label[image_class] = 1

        path_img_dict = os.path.join(path_dir, class_directory)
        for img in os.listdir(path_img_dict):
            path = os.path.join(path_img_dict, img)
            img = np.array(Image.open(path).resize((img_size, img_size), Image.ANTIALIAS))
            img_norm = img / 65535
            test_data.append([np.array(img_norm), label])

        image_class += 1

    shuffle(test_data)
    return test_data


def load_train_test_data(dataset, img_size):
    """ Vrati z datasetu na zadanej ceste, polia s nacitanymi trenovacimi a trénovacimi obrazkami a ich priradeniami do prislusnych tried
    vo forme one-hot vectora, ktore je vo vhodnom formte aby mohlo sluzit ako vstup pre neuronovu siet.
    Predpoklada sa, že dataset ma nasledovnu adresarovu štrukturu:
        dataset /train
                    /class1
                    /class2
                    ...
                /test
                    /class1
                    /class2
                    ...

    :param dataset: cesta k datasetu, z ktoreho maju byt nacitane obrazky, predpokladame, ze maju 16-bitovu farebnu hlbku
    :type dataset: str
    :param img_size: rozmer aky maju mat obrazky po nacitani, upravia sa na stvorcovy tvar podla zadaneho rozmeru
    :type img_size: int
    :return train_data pole kde je na kazdom indexe zaznam o jednom obrazku pre trening, konkretne prislusny
            obrazok normovany na interval [0, 1] a jeho zaradenie do triedy vo forme one-hot vectora
        test_data pole kde je na kazdom indexe zaznam o jednom obrazku pre testovanie, konkretne prislusny
            obrazok normovany na interval [0, 1] a jeho zaradenie do triedy vo forme one-hot vectora
        count_classes pocet tried na trenovanie, ktory sa zisti podla poctu adresarov s obrazkami
    :rtype list
        list
        int
    """
    train_data, count_classes = load_train_data(dataset, img_size)
    test_data = load_test_data(dataset, img_size)
    return train_data, test_data, count_classes


def timestamp():
    date_time_obj = datetime.now()
    return date_time_obj.strftime("%m%d%H%M")
