import yaml
import argparse
from approaches import build_nn
from generators import image_generator, validation_image_generator, generator_nn, generator_cnn_lstm, validation_generator_cnn_lstm, generator_lstm, validation_lstm_generator
from pprint import pprint
from utils import make_dir
import shutil
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn


class Experiment:
    def __init__(self, **kwargs):
        approaches_dict = {'NN': build_nn}
        generators_dict = {'image_generator': image_generator, 
                           'validation_images_generator': validation_image_generator,
                           'generator_nn': generator_nn,
                           'generator_cnn_lstm': generator_cnn_lstm,
                           'validation_generator_cnn_lstm': validation_generator_cnn_lstm,
                           'generator_lstm': generator_lstm,
                           'validation_lstm_generator': validation_lstm_generator}

        self._sets = kwargs

        self._approach = approaches_dict[self._sets['approach']['name']](**self._sets['approach'])

        self._train_gen = generators_dict[self._sets['train_gen']['name']](**self._sets['train_gen'])
        self._val_gen = generators_dict[self._sets['val_gen']['name']](self._sets['val_gen']['name'], 
                                                                       self._sets['val_gen']['path'],
                                                                       self._sets['val_gen']['validation_list'])

    def start(self):

        self._approach.train(self._train_gen, self._val_gen)

        _1, _2, confusion_train = self._approach.evaluate(self._train_gen, True)
        _1, _2, confusion_val = self._approach.evaluate(self._val_gen, True)
        print(confusion_train)
        print(confusion_val)

        conf_dict = {0:'train', 1:'val'}
        words = ['bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'four', 'go', 'happy', 'house', 'left', 
        'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 
        'two', 'up', 'wow', 'yes', 'zero']
        for i, j in enumerate([confusion_train, confusion_val]):
            print(j)
            fig = plt.figure()
            df_cm = pd.DataFrame(j, words, words)
            sn.set(font_scale=0.5)
            sn.heatmap(df_cm, annot=True, annot_kws={"size": 8})
            plt.savefig(f"{self._sets['approach']['exp_dir']}\\confusion_matrix_{conf_dict[i]}.png")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('sets', help="settings file")

    args = parser.parse_args()

    with open(args.sets, 'r') as f:
        sets = yaml.safe_load(f)

    pprint(sets)

    folder_path = make_dir(sets['experiments_dir'])

    shutil.copy('C:\\Users\\oleksandr.vaitovych\\PycharmProjects\\course_project\\'+args.sets,
                folder_path+f"\\{args.sets}")

    sets['approach']['exp_dir'] = folder_path

    experiment = Experiment(**sets)
    experiment.start()

    shutil.copy('C:\\Users\\oleksandr.vaitovych\\PycharmProjects\\course_project\\log.txt', folder_path + "\\log.txt")


if __name__ == "__main__":
    main()