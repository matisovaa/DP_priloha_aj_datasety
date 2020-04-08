import os

import h5py
import numpy

from PIL import Image


def nacitaj_cxi(path_to_cxi):
    """ Nacita difraktogramy z .cxi suboru, ktory bol vytvoreny vyuzitim 
    Condor (Hantke, Ekeberg and Maia. J. Appl. Cryst. (2016). 49, 1356-1362. doi:10.1107/S1600576716009213)

    :param path_to_cxi: cesta k cxi suboru, ktory ma byt nacitany
    :type path_to_cxi: str
    :return pole kde su nacitane difraktogramy so shape (n, x, y), kde n je pocet nacitanych difraktogramov a x, y je ich rozmer
    :rtype ndarray    
    """
    with h5py.File(path_to_cxi, 'r') as f:
        intensity_pattern = numpy.asarray(f["/entry_1/data_1/data"])
    return intensity_pattern


def log_z_matice(matica):
    """ Posunie rozsah hodnot matice aby zacinali od hodnoty 1 a potom ich zlogaritmuje 10-kovym logaritmom

    :param matica: ndarray, ktoreho hodnoty sa zlogaritmuju
    :type matica: ndarray
    :return ndarray, ktora ma rozsah hodnot posunuty aby zacinali od hodnoty 1 a potom boli hodnoty zlogaritmované 10-kovym logaritmom
    :rtype ndarray    
    """
    minimum = matica.min()
    log_matica = numpy.log10(matica - minimum + 1)

    return log_matica


def uint16_z_matice(matica):
    """ Pre vstupnu maticu rozmeru (n, x, y) rozrata hodnoty v kazdej podmatici rozmeru (x, y) na rozsah hodnot uint16.

    :param matica: ndarray so shape (n, x, y), kde n je pocet nacitanych difraktogramov a x, y je ich rozmer
    :type matica: ndarray
    :return ndarray, ktora ma rozmer (n, x, y) a rozratane hodnoty v kazdej podmatici rozmeru (x, y) na rozsah hodnot uint16.
    :rtype ndarray    
    """
    data_uint16 = numpy.copy(matica).astype(numpy.uint16)
    for i in range(matica.shape[0]):
        img = matica[i]
        data_uint16[i] = (65535 * (img - img.min()) / img.ptp()).astype(numpy.uint16)
    return data_uint16


def one_cxi_to_png(source_cxi, target_png_dir):
    """ Vyrobi zlogaritmovane obrazky vo formáte PNG16 zo vsetkych difraktogramov ulozenych v .cxi subore na zadanej ceste source_cxi. 
    .cxi subor musi mat strukturu ako robi 
    Condor (Hantke, Ekeberg and Maia. J. Appl. Cryst. (2016). 49, 1356-1362. doi:10.1107/S1600576716009213).

    :param source_cxi: cesta k cxi suboru
    :type source_cxi: str
    :param target_png_dir: cesta k adresaru, v ktorom tato funkcia vyrobi adresar na obrazky, ktore sa vyrobia zo zadaneho .cxi suboru.
    :type target_png_dir: str        
    """
    data = nacitaj_cxi(source_cxi)

    # zistenie nazvu suboru cxi
    base = os.path.basename(source_cxi)
    name = os.path.splitext(base)[0]

    # logaritmus z hodnot
    data_log = log_z_matice(data)

    # rozrata hodnoty v jednotlivych difraktogramoch na rozsah PNG16
    data_png = uint16_z_matice(data_log)

    # vytvorenie adresara na obrazky
    name_dir = name + '_png16'
    new_directory_path = os.path.join(target_png_dir, name_dir)
    os.mkdir(new_directory_path)

    for idx, np_img in enumerate(data_png):
        Image.fromarray(np_img).save(f"{new_directory_path}/{name}__{idx}.png")


def dir_cxi_to_png(source_cxi_dir, target_png_dir):
    """ Pre kazdy .cxi subor v adresari source_cxi_dir vyrobi zlogaritmovane obrazky vo formáte PNG16 zo vsetkych
    difraktogramov ulozenych v danom .cxi subore a pre kazdy .cxi subor ich ulozi do osobitneho adresara
    co si vyrobi v zadanom adresari target_png_dir.
    .cxi subory musia mat strukturu ako robi 
    Condor (Hantke, Ekeberg and Maia. J. Appl. Cryst. (2016). 49, 1356-1362. doi:10.1107/S1600576716009213).

    :param source_cxi_dir: cesta k adresaru s cxi subormi
    :type source_cxi_dir: str
    :param target_png_dir: cesta k adresaru, v ktorom tato funkcia vyrobi adresar na obrazky, ktore sa vyrobia
        zo zadanych .cxi suborov osobitne pre kazdy subor.
    :type target_png_dir: str        
    """

    for one_cxi in os.listdir(source_cxi_dir):
        if one_cxi.lower().endswith('.cxi'):
            one_cxi_path = os.path.join(source_cxi_dir, one_cxi)
            one_cxi_to_png(one_cxi_path, target_png_dir)
