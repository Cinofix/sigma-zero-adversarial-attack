import datetime
import os
import re
import statistics
import warnings
from distutils.version import LooseVersion
from typing import Callable, Optional, Dict, Union
import random
import numpy as np
from torchvision import utils as vutils
from tqdm import tqdm
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from adv_lib.utils import ForwardCounter, BackwardCounter
from adv_lib.utils.attack_utils import _default_metrics


class Logger():
    # code taken by the official Sparse-RS repo: https://github.com/fra31/sparse-rs/blob/master/utils.py
    def __init__(self, log_path):
        self.log_path = log_path

    def log(self, str_to_log):
        print(str_to_log)
        if not self.log_path is None:
            with open(self.log_path, 'a') as f:
                f.write(str_to_log + '\n')
                f.flush()


def set_seed(seed):
    """ Random seed generation for PyTorch. See https://pytorch.org/docs/stable/notes/randomness.html
        for further details.
    Args:
        seed (int): the seed for pseudonumber generation.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def run_attack(model: nn.Module,
               loader: DataLoader,
               attack: tuple,
               targets: Optional[Union[int, Tensor]] = None,
               metrics: Dict[str, Callable] = _default_metrics,
               return_adv: bool = False
               ) -> dict:
    # code adapted from Official adversarial library repo:
    # https://github.com/jeromerony/adversarial-library/blob/main/adv_lib/utils/attack_utils.py

    torch.cuda.empty_cache()
    model.eval()
    device = next(model.parameters()).device
    targeted = True if targets is not None else False
    loader_length = len(loader)

    if device == torch.device("cuda"):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    else:
        start, end = 0, 0

    forward_counter, backward_counter = ForwardCounter(), BackwardCounter()
    model.register_forward_pre_hook(forward_counter)
    if LooseVersion(torch.__version__) >= LooseVersion('1.8'):
        model.register_full_backward_hook(backward_counter)
    else:
        model.register_backward_hook(backward_counter)
    forwards, backwards = [], []  # number of forward and backward calls per sample

    times, accuracies, ori_success, adv_success = [], [], [], []
    ori_labels, pred_ori, pred_adv = [], [], []
    distances = {k: [] for k in metrics.keys()}

    if return_adv:
        all_inputs, all_adv_inputs = [], []

    for i, (inputs, labels) in enumerate(tqdm(loader, ncols=80, total=loader_length)):
        ori_labels.append(labels.cpu().tolist())

        if return_adv:
            all_inputs.append(inputs.clone())

        # move data to device and get predictions for clean samples
        inputs, labels = inputs.to(device), labels.to(device)

        try:
            logits = model(inputs)
        except RuntimeError as e:
            if 'out of memory' in str(e) or 'valid cuDNN' in str(e):
                print('\n WARNING: ran out of memory, cannot perform experiments with this batch size')
                raise e
            else:
                raise e
        torch.cuda.empty_cache()

        predictions = logits.argmax(dim=1)
        if return_adv:
            pred_ori.append(predictions.cpu().tolist())
        accuracies.extend((predictions == labels).cpu().tolist())
        success = (predictions == targets) if targeted else (predictions != labels)
        ori_success.extend(success.cpu().tolist())

        forward_counter.reset(), backward_counter.reset()
        if device == torch.device('cuda'):
            start.record()
            torch.cuda.reset_peak_memory_stats(device=device)
        try:
            adv_inputs = attack[1](model, inputs, labels)

        except Exception as e:
            adv_inputs = inputs
            if 'out of memory' in str(e) or 'valid cuDNN' in str(e):
                print('\n WARNING: ran out of memory, cannot perform this specific attack with this batch size')
                exit()
            else:
                print(e)

        torch.cuda.empty_cache()
        elapsed_time = 0
        if device == torch.device('cuda'):
            end.record()
            torch.cuda.synchronize()
            elapsed_time = (start.elapsed_time(end)) / 1000  # times for cuda Events are in milliseconds
        times.append(elapsed_time)

        forwards.append(forward_counter.num_samples_called)
        backwards.append(backward_counter.num_samples_called)
        forward_counter.reset(), backward_counter.reset()

        if adv_inputs.min() < 0 or adv_inputs.max() > 1:
            warnings.warn('Values of produced adversarials are not in the [0, 1] range -> Clipping to [0, 1].')
            adv_inputs.clamp_(min=0, max=1)

        adv_logits = model(adv_inputs)
        adv_pred = adv_logits.argmax(dim=1)
        pred_adv.append(adv_pred.cpu().tolist())

        if return_adv:
            all_adv_inputs.append(adv_inputs.clone())
        success = (adv_pred == targets) if targeted else (adv_pred != labels)
        adv_success.extend(success.cpu().tolist())

        for metric, metric_func in metrics.items():
            distances[metric].extend(metric_func(adv_inputs, inputs).detach().cpu().tolist())

    max_memory = 0
    if device != torch.device('cpu'):
        max_memory = torch.cuda.max_memory_allocated(device=device) / 1024 / 1024

    data = {
        'targeted': targeted,
        'accuracy': sum(accuracies) / len(accuracies),
        'ori_success': ori_success,
        'adv_success': adv_success,
        'ASR': sum(adv_success) / len(adv_success),
        'times': times,
        'num_forwards': forwards,
        'num_backwards': backwards,
        'distances': distances,
        'max_memory': max_memory,
        'ori_labels': [item for sublist in ori_labels for item in sublist],
        'pred_ori': [item for sublist in pred_ori for item in sublist],
        'pred_adv': [item for sublist in pred_adv for item in sublist]
    }

    if return_adv:
        if len(all_inputs) > 1:
            all_inputs = torch.cat(all_inputs, dim=0)
            all_adv_inputs = torch.cat(all_adv_inputs, dim=0)
        data['inputs'] = all_inputs
        data['adv_inputs'] = all_adv_inputs

    return data


def save_examples(experiment_folder_path, stats, start_index=0, end_index=0):
    """
    Saves examples to a folder specified by experiment_folder_path
    """
    adv_inputs = stats["adv_inputs"]
    inputs = stats["inputs"]
    examples_folder_path = os.path.join(experiment_folder_path, "examples")
    os.makedirs(examples_folder_path, exist_ok=True)

    for i in range(start_index, end_index + 1):
        if len(adv_inputs) == 1:
            adv_inputs = adv_inputs[0]
            inputs = inputs[0]

        example_image = adv_inputs[i].cpu().detach()
        example_mask = (adv_inputs[i].cpu().detach() - inputs[i].cpu().detach()).abs()
        image_filename = f"example_{i}_{stats['pred_ori'][i]}_{stats['pred_adv'][i]}_{stats['distances']['l0'][i]}.png"
        mask_filename = f"mask_{i}_{stats['pred_ori'][i]}_{stats['pred_adv'][i]}_{stats['distances']['l0'][i]}.png"

        image_path = os.path.join(examples_folder_path, image_filename)
        mask_path = os.path.join(examples_folder_path, mask_filename)

        vutils.save_image(example_image, image_path)
        vutils.save_image(example_mask, mask_path)


def show_salient_statistics(experiment_results, name):
    """
    Show salient statistics given the result of an experiment
    """
    attack_name = name
    asr = experiment_results.get("ASR", "N/A") * 100 if experiment_results.get("ASR", "N/A") != "N/A" else "N/A"

    l0s = ([x for x, y, z in zip(experiment_results['distances']['l0'], experiment_results['adv_success'],
                                 experiment_results["ori_success"]) if y and not z])
    l0_median = statistics.median(l0s) if l0s != [] else 0
    l0_mean = statistics.mean(l0s) if l0s != [] else 0

    time = sum(experiment_results.get("times", [])) / len(experiment_results['adv_success'])

    num_queries = (sum(experiment_results.get('num_forwards', [])) + sum(
        experiment_results.get('num_backwards', []))) / len(experiment_results["ori_success"])

    vram = experiment_results.get("max_memory", "N/A")

    summary = {
        "attack_name": attack_name,
        "ASR": round(asr, 4),
        "L0 Median": round(l0_median, 4),
        "L0 Mean": round(l0_mean, 4),
        "Time (seconds)": round(time, 4),
        "Number of Queries": num_queries,
        "VRAM Usage (MB)": round(vram, 2)
    }

    print(summary)

    return summary


def generate_experiment_name():
    """
    Generates a unique name for the experiment and returns it as a string
    """
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%d-%m-%Y_%H-%M-%S")
    experiment_name = f"exp_{formatted_time}"

    # Replace characters that may interfere with file manager
    experiment_name = re.sub(r"[:]", "_", experiment_name)

    return experiment_name
