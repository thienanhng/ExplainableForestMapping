""" 
Functions to handle model checkpoints.
"""

import torch
import numpy as np

def extract_checkpoint(in_fn, out_fn, epoch):
    """
    Extract the model from training epoch 'epoch' in a training metrics file 'in_fn' and save it in a separate file 
    'out_fn'
    """
    d = torch.load(in_fn, map_location='cpu')
    args = d['args']
    checkpoint = {'model': d['model_checkpoints'][epoch],
                    'optimizer': d['optimizer_checkpoints'][epoch],
                    'epoch': epoch}
    try:
        checkpoint['random_state'] = d['random_state'][epoch]
    except KeyError:
        pass
    checkpoint['model_params'] = {}
    for key in 'decision', 'epsilon_rule':
        try:
            checkpoint['model_params'][key] = args[key]
        except KeyError:
            pass
    torch.save(checkpoint, out_fn)

def truncate_training_file(in_fn, out_fn, trunc_epoch):
    """
    Truncates the training metrics and checkpoints in 'in_fn' at epoch trunc_epoch (included) and saves the modified 
    file in 'out_fn'
    """
    print('Truncating file at epoch {} included'.format(trunc_epoch))
    d = torch.load(in_fn, map_location='cpu')
    d_out = {}
    d_out['args'] = d['args']
    d_out['args']['num_epochs'] = trunc_epoch + 1
    
    schedule_params = {'learning_schedule': ['lr', 'lambda_sem'],
                       'negative_sampling_schedule': ['n_negative_samples']}

    for key in schedule_params.keys():
        if key in d['args']:
            stop_idx = np.argmin(np.cumsum(d['args'][key]) <= trunc_epoch)+1
            d_out['args'][key] = d['args'][key][:stop_idx]
            d_out['args'][key][-1] += trunc_epoch + 1 - np.sum(d_out['args'][key])
            for subkey in schedule_params[key]:
                d_out['args'][subkey] = d['args'][subkey][:stop_idx]
            
        
    keys = list(d.keys())
    keys.remove('args')
    for key in keys:
        d_out[key] = d[key][:trunc_epoch+1]

    with open(out_fn, 'wb') as f:
        torch.save(d_out, f)

def validate_state_dicts(model_state_dict_1, model_state_dict_2):
    """From https://gist.github.com/rohan-varma/a0a75e9a0fbe9ccc7420b04bff4a7212"""
    if len(model_state_dict_1) != len(model_state_dict_2):
        print(
            f"Length mismatch: {len(model_state_dict_1)}, {len(model_state_dict_2)}"
        )
        return False

    # Replicate modules have "module" attached to their keys, so strip these off when comparing to local model.
    if next(iter(model_state_dict_1.keys())).startswith("module"):
        model_state_dict_1 = {
            k[len("module") + 1 :]: v for k, v in model_state_dict_1.items()
        }

    if next(iter(model_state_dict_2.keys())).startswith("module"):
        model_state_dict_2 = {
            k[len("module") + 1 :]: v for k, v in model_state_dict_2.items()
        }

    for ((k_1, v_1), (k_2, v_2)) in zip(
        model_state_dict_1.items(), model_state_dict_2.items()
    ):
        if k_1 != k_2:
            print(f"Key mismatch: {k_1} vs {k_2}")
            return False
        # convert both to the same CUDA device
        if str(v_1.device) != "cuda:0":
            v_1 = v_1.to("cuda:0" if torch.cuda.is_available() else "cpu")
        if str(v_2.device) != "cuda:0":
            v_2 = v_2.to("cuda:0" if torch.cuda.is_available() else "cpu")

        if not torch.allclose(v_1, v_2):
            print(f"Tensor mismatch: {v_1} vs {v_2}")
            return False

def compare_model_files(fn1, fn2):
    m1 = torch.load(fn1)['model']
    m2 = torch.load(fn2)['model']
    validate_state_dicts(m1, m2)

if __name__ == "__main__":
    experiment = 'sb_seed_0'
    epoch = 4
    extract_checkpoint('output/{}/training/{}_metrics.pt'.format(experiment, experiment), 
                       'output/{}/training/{}_model_epoch_{}.pt'.format(experiment, experiment, epoch), epoch)
    truncate_training_file('output/{}/training/{}_metrics.pt'.format(experiment, experiment),
                           'output/{}/training/{}_metrics_epoch_{}.pt'.format(experiment, experiment, epoch),
                           epoch)
