from pickle import UnpicklingError
import torch
import gdown
import os

MODEL_URL = {'bb': 'https://drive.google.com/uc?export=download&id=1SO8ZorC_6CoPPlbmkBdqjDxUU6RK-IWk',
             'bb_flat': 'https://drive.google.com/uc?export=download&id=1F90sUPzYEB8nmaVlKr4XQgmWu0DXyUVt',
             'bb_dem_ablation': 'https://drive.google.com/uc?export=download&id=1wDwUEeMrV0BYLzlpppcMWZoNfbeTldb0',
             'sb': 'https://drive.google.com/uc?export=download&id=1cai3RJDxG5-XvixnZfT2tyELegj_l536', 
             'sb_corrp': 'https://drive.google.com/uc?export=download&id=1WHHQZdF8owZeZBPLuNnypEk7u5PTaxCu', 
             'sb_rulem': 'https://drive.google.com/uc?export=download&id=14bI3glQGhkDq5_4hXTylvNQQJEeLTU2-',
             'sb_epoch_4': 'https://drive.google.com/uc?export=download&id=1YdQ_29cSHHuUYms5L_DIZwLLsUEa6BJq'} 

METRICS_URL = {'bb': 'https://drive.google.com/uc?export=download&id=1YOO9UTSJw4PrnqOZSAfhAlJGXx3pA1-f', 
               'bb_flat': 'https://drive.google.com/uc?export=download&id=11ELUcEzVVuS6EVuUw--gGg6Uo2xLrmZY', 
               'bb_dem_ablation': 'https://drive.google.com/uc?export=download&id=1CRr8gISGvCNi74cW2Dj0vegwSj2bbGZA', 
               'sb': 'https://drive.google.com/uc?export=download&id=1k701vgUCn968PZOyNlOhUL5SSBtQoCNA', 
               'sb_corrp': 'https://drive.google.com/uc?export=download&id=1-l0JkBSWC-Luy4pjG-S9hhVHHEQR2DVI', 
               'sb_rulem': 'https://drive.google.com/uc?export=download&id=1TXZGfsjHQdnBQW7mV6EHF3s66M0qJkjD', 
               'sb_epoch_4': 'https://drive.google.com/uc?export=download&id=1kJoNfkaOvNCJCih46LNe9UQGyeUVbZ2_'} 

def get_model(name, dir):
    try:
        url = MODEL_URL[name]
    except KeyError:
        raise RuntimeError('Model "{}" is not in the available pretrained models.'.format(name))
    if not os.path.exists(dir):
        print('Creating directory {}'.format(dir))
        os.makedirs(dir)
    model_fn = gdown.download(url, os.path.join(dir, '{}_model_from_gdrive.pt'.format(name)))
    model_obj = torch.load(model_fn)
    return model_obj

def get_metrics(name, dir):
    try:
        url = METRICS_URL[name]
    except KeyError:
        raise RuntimeError('Experiment "{}" is not in the available metrics files.'.format(name))
    if not os.path.exists(dir):
        print('Creating directory {}'.format(dir))
        os.makedirs(dir)
    metrics_fn = gdown.download(url, os.path.join(dir, '{}_metrics_from_gdrive.pt'.format(name)))
    metrics_obj = torch.load(metrics_fn)
    return metrics_obj

