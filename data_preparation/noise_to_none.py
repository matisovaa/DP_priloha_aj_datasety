import os

import numpy
import skimage.util
import random

from PIL import Image


def noise_to_one_image(image, noise, var=0.01):
    """ Prida do zadaneho image zadany sum so zadanou intenzitou pomocou python module skimage.util.random_noise.

    :param image: obrazok, do ktoreho sa ma pridat sum, predpokladame, ze ma 16-bitovu farebnu hlbku
    :type image: ndarray
    :param noise: aky typ sumu sa ma pridat. Bud 'poisson' alebo 'gaussian'.
    :type noise: str
    :param var: hodnota variancie nahodnej distribucie pri gaussian sume. (variance = (standard deviation) ** 2)
    :type var: float
    :return image s pridanym sumom vo formate ndarray s hodnotami v rozsahu uint16
    :rtype ndarray    
    """
    img = image / 65535.0
    if noise == 'poisson':
        img_noise = skimage.util.random_noise(img, mode=noise)
    if noise == 'gaussian':
        img_noise = skimage.util.random_noise(img, mode=noise, var=var)
    return (img_noise * 65535).astype(numpy.uint16)


def noise_to_none_difractograms(dir_imagedirs, target_dir_path, noise='gaussian', var=0.01):
    """ Vytvori novy adresar s obrazkami v target_dir_path, tak, ze skopiruje obrazky,
    ktore su v adresaroch adresara dir_imagedirs a prida do nich zadany sum
    so zadanou intenzitou pomocou python module skimage.util.random_noise.

    :param dir_imagedirs: cesta k adresaru, v ktorom su adresare s obrazkami, do ktorych chceme pridat sum.
    Predpoklada sa, ze nazov podadresarov s obrazkami zacina nazvom spolocneho oznacenia obrazkov 
    v danom podadresari, ktory je oddeleny "_" a je jedinecny.
    :type dir_imagedirs: str
    :param target_dir_path: cesta k adresaru kde sa ma ulozit adresar s obrazkami s pridanym sumom
    :type target_dir_path: str
    :param noise: aky typ sumu sa ma pridat. Bud 'poisson' alebo 'gaussian'.
    :type noise: str
    :param var: hodnota variancie nahodnej distribucie pri gaussian sume. (variance = (standard deviation) ** 2)
    :type var: float      
    """
    # nazov noveho adresara pre adresare so zasumenymi obrazkami vytvorime podla nazvu pridavaneho sumu
    name_target_dir = noise
    if noise == 'gaussian':
        name_target_dir += str(var)
    target_dir = os.path.join(target_dir_path, name_target_dir)
    os.mkdir(target_dir)

    for dict_one_class in os.listdir(dir_imagedirs):

        # zisti sa zaciatok nazvu podadresara a podla neho sa potom pomenuje podadresar so zasumenimi obrazkami
        pdb = dict_one_class.split('_')[0]

        target_dir_one_class = os.path.join(target_dir, pdb + '_' + name_target_dir)
        os.mkdir(target_dir_one_class)

        # cesta k podadresaru, v ktorom su obrazky
        directory_path_one_class = os.path.join(dir_imagedirs, dict_one_class)

        idx = 0
        for img in os.listdir(directory_path_one_class):
            path_source = os.path.join(directory_path_one_class, img)
            image = numpy.array(Image.open(path_source))

            # pridanie sumu do obrazku
            noisy_image = noise_to_one_image(image, noise, var)

            Image.fromarray(noisy_image).save(f"{target_dir_one_class}/{pdb}_{name_target_dir}__{idx}.png")
            idx += 1


def noise_to_dataset(path_to_dataset, target_dir_path, name_new_dataset, noise='gaussian', var=0.01, max_var=None):
    """ Vyrobi novy dataset zkopirovanim obrazkov z datasetu path_to_dataset, pricom do kazdeho obrazku
    vo vytvaranom datasete prida zadany sum so zadanou intenzitou pomocou python module skimage.util.random_noise.
    Ak je nastavena hodnota max_var tak sa do obrazkov prida sum z intervalu medzi var a max_var,
    kde sa predpoklada, ze var je nizsia hodnota nez max_var.

    :param path_to_dataset: cesta k datasetu, z ktoreho vyrobime novy dataset s pridanym sumom.
    Predpoklada sa, ze nazov podadresarov s obrazkami zacina nazvom spolocneho oznacenia obrazkov
    v danom podadresari, ktory je oddeleny "_" a je jedinecny.
    :type path_to_dataset: str
    :param target_dir_path: cesta k adresaru kde sa ma vytvorit dataset s pridanym sumom
    :type target_dir_path: str
    :param name_new_dataset: nazov pre vytvarany dataset
    :type name_new_dataset: str
    :param noise: aky typ sumu sa ma pridat. Bud 'poisson' alebo 'gaussian'.
    :type noise: str
    :param var: hodnota variancie nahodnej distribucie pri gaussian sume. (variance = (standard deviation) ** 2)
    :type var: float
    :param max_var: hodnota naximalnej variancie nahodnej distribucie pri gaussian sume. (variance = (standard deviation) ** 2)
        Ak je nastavena tak sa do obrazkov prida sum z intervalu medzi var a max_var, kde sa predpoklada, ze var je nizsia hodnota
    :type max_var: float
    """

    # vytvorenie adresarovej struktury noveho datasetu
    new_target_dir = os.path.join(target_dir_path, name_new_dataset)
    os.mkdir(new_target_dir)

    train_dir_new = os.path.join(new_target_dir, 'train')
    os.mkdir(train_dir_new)

    test_dir_new = os.path.join(new_target_dir, 'test')
    os.mkdir(test_dir_new)

    train_source = os.path.join(path_to_dataset, 'train')
    test_source = os.path.join(path_to_dataset, 'test')

    for dict_one_test_class in os.listdir(test_source):
        pdb = dict_one_test_class.split('_')[0]

        # cesta k jednej test triede
        path_dict_one_test_class = os.path.join(test_source, dict_one_test_class)

        # cesta k novemu test adresaru
        name_target_dir = f"{pdb}_{noise}"
        if noise == 'gaussian':
            name_target_dir += str(var)
            if max_var is not None:
                name_target_dir += '-' + str(max_var)
        path_dict_one_test_class_new = os.path.join(test_dir_new, name_target_dir)
        os.mkdir(path_dict_one_test_class_new)

        idx = 0
        for img in os.listdir(path_dict_one_test_class):
            path_source = os.path.join(path_dict_one_test_class, img)
            image = numpy.array(Image.open(path_source))

            var_actual = var
            # pridanie sumu do obrazku
            if max_var is None:
                noisy_image = noise_to_one_image(image, noise, var_actual)
            else:
                var_actual = random.uniform(var, max_var)
                noisy_image = noise_to_one_image(image, noise, var_actual)

            g_var = ''
            if noise == 'gaussian':
                g_var += str(round(var_actual, 3))
            Image.fromarray(noisy_image).save(f"{path_dict_one_test_class_new}/{pdb}_{noise}{g_var}__{idx}.png")
            idx += 1

    for dict_one_train_class in os.listdir(train_source):
        pdb = dict_one_train_class.split('_')[0]

        # cesta k jednej train triede
        path_dict_one_train_class = os.path.join(train_source, dict_one_train_class)

        # cesta k novemu train adresaru
        name_target_dir = f"{pdb}_{noise}"
        if noise == 'gaussian':
            name_target_dir += str(var)
            if max_var is not None:
                name_target_dir += '-' + str(max_var)
        path_dict_one_train_class_new = os.path.join(train_dir_new, name_target_dir)
        os.mkdir(path_dict_one_train_class_new)

        # je tu test_dir_new, lebo chceme zistit, ze po aky index uz vyrobilo obrazky aby sme vedeli, ze od kade indexovat
        test_count = len(os.listdir(os.path.join(test_dir_new, name_target_dir)))

        idx = test_count
        for img in os.listdir(path_dict_one_train_class):
            path_source = os.path.join(path_dict_one_train_class, img)
            image = numpy.array(Image.open(path_source))

            var_actual = var
            # pridanie sumu do obrazku
            if max_var is None:
                noisy_image = noise_to_one_image(image, noise, var_actual)
            else:
                var_actual = random.uniform(var, max_var)
                noisy_image = noise_to_one_image(image, noise, var_actual)

            g_var = ''
            if noise == 'gaussian':
                g_var += str(round(var_actual, 3))
            Image.fromarray(noisy_image).save(f"{path_dict_one_train_class_new}/{pdb}_{noise}{g_var}__{idx}.png")
            idx += 1
