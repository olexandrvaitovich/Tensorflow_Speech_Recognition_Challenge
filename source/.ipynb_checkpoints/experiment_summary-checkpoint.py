import yaml
import os
import re

from approaches import build_cnn
from generators import image_generator, validation_image_generator


def extract_metrics(log_file):
    def extract(metric):
        return re.compile(f'{metric}: (\d*.\d*)').findall(log_file)
    
    metrics = ['loss', 'acc', 'val_loss', 'val_acc', 'final loss', 'final acc']
    
    return {m:extract(m) for m in metrics}

def get_examples(sets):
    
    sets['approach']['mode'] = 'inference'
    
    



