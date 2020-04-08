import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.regularizers import l1_l2

from cnn.auxiliary_functions import *


def model_cnn(filters_count, img_size, dropout, classes_count, l1_l2_value):
    """ Vrati model konvolučnej neuronovej siete nastaveny podla zadanych parametrov.

    :param filters_count: Pole, v ktorom su zadane pocty konvolucnych jadier pre jednotlivé vrstvy konvolučnej siete
        v poradí od prvej konvolucnej vrstvy.
    :type filters_count: list
    :param img_size: rozmer obrazkov, ktore sluzia ako vstup sieti (dlzka strany stvorcoveho obrazku)
    :type img_size: int
    :param dropout: float hodnota medzi 0 a 1. Urcuje podiel, ze aka cast vstupnych neuronov do tejto vrstvy bude mat svoj
        vystup nastaveny na 0 (drop)
    :type dropout: float
    :param classes_count: pocet tried, do ktorych siet klasifikuje
    :type classes_count: int
    :param l1_l2_value: hodnota regularizacneho koeficientu, ktora bude nastavena pre L1 a L2 regularizaciu
        na každej konvolučnj vrstve siete
    :type l1_l2_value: float
    :return model konvolučnej siete s parametrami vrstiev nastavenymi podla zadanych parametrov
    :rtype keras.models.Sequential
    """
    model = Sequential()
    model.add(
        Conv2D(filters_count[0], kernel_size=(3, 3), kernel_regularizer=l1_l2(l1=l1_l2_value, l2=l1_l2_value),
               activation='relu',
               input_shape=(img_size, img_size, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    for i in range(1, len(filters_count)):
        model.add(Conv2D(filters_count[i], kernel_size=(3, 3), kernel_regularizer=l1_l2(l1=l1_l2_value, l2=l1_l2_value),
                         activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(dropout))

    model.add(Flatten())
    model.add(Dense(classes_count, activation='softmax'))

    return model


def print_model(filters_count, dataset, epochs, dropout, img_size, l1_l2_value, optimizer):
    """ Prehladne vypise informacie o spustanom trenovani siete na zaklade hodnot v parametri.

    :param filters_count: Pole, v ktorom su zadane pocty konvolucnych jadier pre jednotlivé vrstvy konvolučnej siete
        v poradí od prvej konvolucnej vrstvy.
    :type filters_count: list
    :param dataset: cesta k datasetu
    :type dataset: str
    :param epochs: počet epoch kolko bude bezat trenovanie siete
    :type epochs: int
    :param dropout: float hodnota medzi 0 a 1. Urcuje podiel, ze aka cast vstupnych neuronov do tejto vrstvy bude mat svoj
        vystup nastaveny na 0 (drop)
    :type dropout: float
    :param img_size: rozmer obrazkov, ktore sluzia ako vstup sieti (dlzka strany stvorcoveho obrazku)
    :type img_size: int
    :param l1_l2_value: hodnota regularizacneho koeficientu, ktora bude nastavena pre L1 a L2 regularizaciu
        na každej konvolučnj vrstve siete
    :type l1_l2_value: float
    :param optimizer: optimalizacny algoritmus pouzity pri kompilacii modelu. Je mozne pouzit lubovolny
        co poskytuje Keras (https://keras.io/optimizers/)
    :type optimizer: str
    :return file_to_save_name nazov suboru pre ulozenie naucenej siete tak aby obsahoval vsetky potrebne informacie
            o nastaveni siete pri trenovani
        network_structure informacia o tom, ze aky model siete bol trenovany
    :rtype str
        str
    """
    timestamp_value = timestamp()
    network_structure = str(filters_count) + '_1DR' + str(dropout)

    if l1_l2_value != 0.0:
        network_structure += '_L1_L2-' + str(l1_l2_value)

    if optimizer != 'adam':
        network_structure += '_' + optimizer

    file_to_save_name = network_structure + '_' + dataset.replace('/', '-') + '_EP' + str(
        epochs) + '_' + timestamp_value

    print(timestamp_value)
    print("NETWORK =", network_structure)
    print("DATASET =", dataset)
    print("EPOCHS =", epochs)
    print("IMG_SIZE =", img_size)
    return file_to_save_name, network_structure


def save_info(model, history, acc, file_to_save_name, network_structure, dataset):
    """ Ulozi informacie o naucenom modely. Samotny nauceny model ulozi do suboru formatu HDF5.
    Na koniec suboru s nazvom "output_summary.txt" zapise zhrnutie o trenovani siete
    (na akom datasete bola siet trenovana a ake na ňom dosiahla vysledky).
    Ulozi grafy accuracy a loss z trenovania siete vo forme obrazku.

    :param model: Nauceny model siete urceny na ulozenie
    :type model: keras.models.Sequential
    :param history: A History object. Its History.history attribute is a record of training loss values
        and metrics values at successive epochs, as well as validation loss values
        and validation metrics values (if applicable). (https://keras.io/models/model/)
    :type history: History object
    :param acc: accuracy nauceneho modelu na testovacom datasete
    :type acc: float
    :param file_to_save_name: nazov suboru pre ulozenie naucenej siete tak aby obsahoval vsetky potrebne informacie
            o nastaveni siete pri trenovani
    :type file_to_save_name: str
    :param network_structure: informacia o tom, ze aky model siete bol trenovany
    :type network_structure: str
    :param dataset: cesta k datasetu, na ktorom bol dany model natrenovany
    :type dataset: str
    """
    h_keys = list(history.history.keys())

    f = open("output_summary.txt", "a")
    f.write(file_to_save_name + "\n")

    maximum = max(history.history[h_keys[1]])
    epocha_max = history.history[h_keys[1]].index(maximum) + 1
    f.write(
        str(round(acc * 100, 2)) + "%" + '\t' + str(round(maximum * 100, 2)) + '% after ' + str(
            epocha_max) + '\t' + str(
            round(history.history[h_keys[3]][-1] * 100, 2)) + "%" + '\t' + dataset + '\t' + network_structure + "\n")
    f.write("------------------------------------" + "\n")
    f.close()

    # summarize history for accuracy
    fig = plt.figure()
    plt.plot(history.history[h_keys[3]])
    plt.plot(history.history[h_keys[1]])
    plt.title('model accuracy', fontsize=16)
    plt.ylabel('accuracy', fontsize=16)
    plt.xlabel('epoch', fontsize=16)
    plt.legend(['train', 'test'], fontsize=14, loc='lower right')
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.tight_layout()

    plt.savefig('acc_' + file_to_save_name + '.png')
    plt.show()

    # summarize history for loss
    fig = plt.figure()
    plt.plot(history.history[h_keys[2]])
    plt.plot(history.history[h_keys[0]])
    plt.title('model loss', fontsize=16)
    plt.ylabel('loss', fontsize=16)
    plt.xlabel('epoch', fontsize=16)
    plt.legend(['train', 'test'], fontsize=14, loc='upper right')
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.tight_layout()

    plt.savefig('loss_' + file_to_save_name + '.png')
    plt.show()

    model.save(file_to_save_name + '.h5')


def train_cnn(filters_count=[8, 16, 16, 32, 32, 64], dataset='../datasets/5_none_1000/', epochs=100,
              dropout=0.2, l1_l2_value=0.0, optimizer='adam'):
    """ Nacita dataset, spusti trenovanie siete a ulozi natrenovany model a informacie o trenovani.

    :param filters_count: Pole, v ktorom su zadane pocty konvolucnych jadier pre jednotlivé vrstvy konvolučnej siete
        v poradí od prvej konvolucnej vrstvy.
    :type filters_count: list
    :param dataset: cesta k datasetu
    :type dataset: str
    :param epochs: počet epoch kolko bude bezat trenovanie siete
    :type epochs: int
    :param dropout: float hodnota medzi 0 a 1. Urcuje podiel, ze aka cast vstupnych neuronov do tejto vrstvy bude mat svoj
        vystup nastaveny na 0 (drop)
    :type dropout: float
    :param l1_l2_value: hodnota regularizacneho koeficientu, ktora bude nastavena pre L1 a L2 regularizaciu
        na každej konvolučnj vrstve siete
    :type l1_l2_value: float
    :param optimizer: optimalizacny algoritmus pouzity pri kompilacii modelu. Je mozne pouzit lubovolny
        co poskytuje Keras (https://keras.io/optimizers/)
    :type optimizer: str
    """
    img_size = 256

    # vypis info o spustanom modely
    file_to_save_name, network_structure = print_model(filters_count, dataset, epochs, dropout, img_size,
                                                       l1_l2_value, optimizer)

    # priprava datasetu
    train_data, test_data, count_classes = load_train_test_data(dataset, img_size)
    train_images = np.array([i[0] for i in train_data]).reshape(-1, img_size, img_size, 1)
    train_labels = np.array([i[1] for i in train_data])
    test_images = np.array([i[0] for i in test_data]).reshape(-1, img_size, img_size, 1)
    test_labels = np.array([i[1] for i in test_data])

    model = model_cnn(filters_count, img_size, dropout, count_classes, l1_l2_value)
    print(model.summary())
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # trenovanie modelu
    history = model.fit(train_images, train_labels, batch_size=50,
                        validation_data=(test_images, test_labels),
                        epochs=epochs, verbose=2)

    loss, acc = model.evaluate(test_images, test_labels, verbose=2)
    print(acc * 100)

    # ulozenie natrenovaneho modelu, grafov acc a loss z trenovania
    save_info(model, history, acc, file_to_save_name, network_structure, dataset)
