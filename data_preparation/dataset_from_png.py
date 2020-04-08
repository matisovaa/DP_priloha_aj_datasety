import os
import shutil


def dataset_from_directory(path_source_dir, path_target_dir, name_new, count_train, count_test):
    """ Vytvori dataset s nasledujucou strukturov: 
    hlavny priecinok datasetu obsahuje priecinok train a test s podadresarmi pre kazdu z tried obrazkov.
    Obrazky do datasetu prekopiruje z adresara path_source_dir, v ktorom predpokladame, ze je podadresar pre kazdu triedu obrazkov.
    Predpokladame, ze v kazdom podadresari so zdrojovimi obrazkami (v kazdej triede) je pocet obrazkov >= count_train + count_test

    :param path_source_dir: cesta k adresaru, v ktorom je podadresar pre kazdu triedu, z ktorej budu train a test obrazky
    :type path_source_dir: str
    :param path_target_dir: cesta k adresaru kde sa vyrobi adresar s datasetom
    :type path_target_dir: str
    :param name_new: nazov pre vytvarany dataset
    :type name_new: str
    :param count_train: pocet train obrazkov co chceme vybrat z kazdej triedy 
        (count_train + count_test <= pocet obrazkov v triede s najmenej obrazkami)
    :type count_train: uint
    :param count_test: pocet test obrazkov co chceme vybrat z kazdej triedy
        (count_train + count_test <= pocet obrazkov v triede s najmenej obrazkami)
    :type count_test: uint        
    """

    # vytvorenie adresarovej struktury datasetu
    new_target_dir = os.path.join(path_target_dir, name_new)
    os.mkdir(new_target_dir)

    train_dir = os.path.join(new_target_dir, 'train')
    os.mkdir(train_dir)

    test_dir = os.path.join(new_target_dir, 'test')
    os.mkdir(test_dir)

    for dict_one_class in os.listdir(path_source_dir):
        # vytvorenie priecinku pre jednu trenovaciu triedu
        directory_path_train = os.path.join(train_dir, dict_one_class)
        os.mkdir(directory_path_train)

        # vytvorenie priecinku pre jednu testovaciu triedu
        directory_path_test = os.path.join(test_dir, dict_one_class)
        os.mkdir(directory_path_test)

        # naplnenie priecinku prislusnym poctom trenovacich obrazkov zo zdrojovej triedy        
        path_img_dict = os.path.join(path_source_dir, dict_one_class)

        for img in os.listdir(path_img_dict)[0:count_train]:
            path_source = os.path.join(path_img_dict, img)
            path_target = os.path.join(directory_path_train, img)
            shutil.copy(path_source, path_target)

        for img in os.listdir(path_img_dict)[count_train:count_train + count_test]:
            path_source = os.path.join(path_img_dict, img)
            path_target = os.path.join(directory_path_test, img)
            shutil.copy(path_source, path_target)
