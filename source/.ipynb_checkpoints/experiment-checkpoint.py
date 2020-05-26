import yaml
import argparse
from approaches import build_cnn
from generators import image_generator, validation_image_generator
from pprint import pprint
from utils import make_dir
import shutil


class Experiment:
    def __init__(self, **kwargs):
        approaches_dict = {'CNN': build_cnn}
        generators_dict = {'image_generator': image_generator, 'validation_images_generator': validation_image_generator}

        self._sets = kwargs

        self._approach = approaches_dict[self._sets['approach']['name']](**self._sets['approach'])

        self._train_gen = generators_dict[self._sets['train_gen']['name']](**self._sets['train_gen'])
        self._val_gen = generators_dict[self._sets['val_gen']['name']](self._sets['val_gen']['path'],
                                                                       self._sets['val_gen']['validation_list'])
        self._test_gen = generators_dict[self._sets['test_gen']['name']](**self._sets['test_gen'])

    def start(self):
        self._approach.train(self._train_gen, self._val_gen)

        self._approach.evaluate(self._train_gen, True)
        self._approach.evaluate(self._val_gen, True)
        self._approach.evaluate(self._test_gen, True)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('sets', help="settings file")

    args = parser.parse_args()

    with open(args.sets, 'r') as f:
        sets = yaml.safe_load(f)

    pprint(sets)

    folder_path = make_dir(sets['experiments_dir'])

    shutil.copy('C:\\Users\\oleksandr.vaitovych\\PycharmProjects\\course_project\\settings.yaml',
                folder_path+"\\setting.yaml")

    sets['approach']['exp_dir'] = folder_path

    experiment = Experiment(**sets)
    experiment.start()

    shutil.copy('C:\\Users\\oleksandr.vaitovych\\PycharmProjects\\course_project\\log.txt', folder_path + "\\log.txt")


if __name__ == "__main__":
    main()