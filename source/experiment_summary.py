import yaml
import os
import re
import argparse
from approaches import build_cnn_v1
from generators import image_generator, validation_image_generator
from utils import make_dir
import shutil
import json
import matplotlib.pyplot as plt


def extract_metrics(log_file):
    def extract(metric):
        return re.compile(f'{metric}: (\d*.\d*)').findall(log_file)
    
    metrics = ['Loss', 'Acc', 'Val_loss', 'Val_acc', 'Final loss', 'Final acc']
    
    return {m:extract(m) for m in metrics}    


def save_plot(metrics, exp_dir_path):

    fig = plt.figure()
    plt.plot(list(map(lambda x:float(x), metrics['Loss'])))
    plt.plot(list(map(lambda x:float(x), metrics['Val_loss'])))
    plt.xlabel('epoch')
    plt.savefig(f'{exp_dir_path}loss.png')

    fig = plt.figure()
    plt.plot(list(map(lambda x:float(x), metrics['Acc'])))
    plt.plot(list(map(lambda x:float(x), metrics['Val_acc'])))
    plt.xlabel('epoch')
    plt.savefig(f'{exp_dir_path}acc.png')


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('sets', help="settings file")

    args = parser.parse_args()

    with open(args.sets, 'r') as f:
        sets = yaml.safe_load(f)

    exp_dir_path = args.sets[:-args.sets[::-1].index('\\')]
    log_file_path = exp_dir_path + 'log.txt'

    print(log_file_path)

    with open(log_file_path, 'r') as f:
    	log_file = f.read()

    metrics = extract_metrics(log_file)

    results = {'Final loss': metrics['Final loss'], 'Final acc': metrics['Final acc']}

    save_plot(metrics, exp_dir_path)

    with open(exp_dir_path+'results.json', 'w+') as f:
    	json.dump(results, f)



if __name__ == "__main__":
    main()
    



