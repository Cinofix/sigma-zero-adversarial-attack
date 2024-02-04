# python main.py --device=cuda:0 --config=config.json

import json
from functools import partial
import os
import pickle
import torch
from utilities import set_seed, save_examples, show_salient_statistics, generate_experiment_name
import argparse

from adv_lib.attacks.fast_minimum_norm import fmn
from adv_lib.attacks.stochastic_sparse_attacks import vfga
from adv_lib.attacks.primal_dual_gradient_descent import pdpgd
from utils.attack_wrappers import sparsefool, PGD0, binary_PGD0, binary_sparse_rs, sparse_rs, dataset_BB_attack, \
    BB_attack, EAD_attack
from sigma_zero import sigma_zero

from model import get_local_model
from dataset import get_dataset_loaders
from utilities import run_attack


def read_config_file(config_file_path):
    with open(config_file_path, "r") as config_file:
        config_data = json.load(config_file)
    return config_data


def process_and_save_results(experiment_name, stats, config, dataset, model_name, attack_name, num_samples, batch_size,
                             show_preview=True, save_adversarial=False):
    """
    Processes and saves the results of the experiments, it creates folders to save salient statistics, results and
    example images
    """
    # Create directories for dataset, model, and attack if they don't exist
    if not os.path.exists("results"):
        os.makedirs("results")
    if not os.path.exists(f"results/{dataset}"):
        os.makedirs(f"results/{dataset}")
    if not os.path.exists(f"results/{dataset}/{model_name}"):
        os.makedirs(f"results/{dataset}/{model_name}")
    if not os.path.exists(f"results/{dataset}/{model_name}/{attack_name}"):
        os.makedirs(f"results/{dataset}/{model_name}/{attack_name}")

    # Create an experiment folder
    experiment_folder_path = f"results/{dataset}/{model_name}/{attack_name}/{experiment_name}"
    os.makedirs(experiment_folder_path)

    # Save the result of show_salient_statistics as a JSON file
    summary = show_salient_statistics(stats, attack_name)
    summary_file_path = f"{experiment_folder_path}/{num_samples}_s_{batch_size}_bs_summary.json"
    with open(summary_file_path, "w") as summary_file:
        json.dump(summary, summary_file, indent=4)

    # Save the inputs and adv_inputs tensors with pickle
    if "inputs" in stats and "adv_inputs" in stats:

        if show_preview == True:
            save_examples(experiment_folder_path, stats, 0, 4)

        if save_adversarial == True:
            inputs_file_path = f"{experiment_folder_path}/inputs.pkl"
            adv_inputs_file_path = f"{experiment_folder_path}/adv_inputs.pkl"
            with open(inputs_file_path, "wb") as inputs_file:
                pickle.dump(stats["inputs"], inputs_file)
            with open(adv_inputs_file_path, "wb") as adv_inputs_file:
                pickle.dump(stats["adv_inputs"], adv_inputs_file)
        del stats["inputs"]
        del stats["adv_inputs"]

    # Save the stats to a JSON file
    results_file_path = f"{experiment_folder_path}/{num_samples}_s_{batch_size}_bs_results.json"
    with open(results_file_path, "w") as results_file:
        json.dump(stats, results_file, indent=4)

    config_file_path = f"{experiment_folder_path}/{num_samples}_s_{batch_size}_bs_config.json"
    with open(config_file_path, "w") as config_file:
        json.dump(config, config_file, indent=4)


def main():
    parser = argparse.ArgumentParser(description='Run experiments based on a configuration file.')
    parser.add_argument('--config', type=str, help='Path to the configuration file')
    parser.add_argument('--device', default='cpu', type=str, help='Device to use for computation')

    args = parser.parse_args()

    config_file_path = args.config
    config_data = read_config_file(config_file_path)

    device = torch.device(args.device)

    print(f"Trying computations on {device}")

    for experiment in config_data["experiments"]:
        set_seed(config_data["seed"])

        attack_name = experiment["attack"]["name"]
        attack_params = experiment["attack"]["params"]
        dataset = experiment["dataset"]
        model_name = experiment["model"]
        batch_size = experiment["batch_size"]
        n_samples = experiment["n_samples"]

        dataloaders = get_dataset_loaders(dataset, batch_size=batch_size, n_examples=n_samples,
                                          seed=config_data["seed"])
        model = get_local_model(model_name, dataset)
        model.eval()
        model = model.to(device)

        print(f"Chosen attack: {experiment['attack']['name']}")
        test_attacks = {
            'DTBB': partial(dataset_BB_attack),
            'BB': partial(BB_attack),
            'PGD0': partial(binary_PGD0),
            'fixed-PGD0': partial(PGD0),
            'Sparse-RS': partial(binary_sparse_rs),
            'fixed-Sparse-RS': partial(sparse_rs),
            'SPARSEFOOL': partial(sparsefool),
            'EAD': partial(EAD_attack),
            'VFGA': partial(vfga),
            'PDPGD': partial(pdpgd),
            'FMN': partial(fmn),
            'sigma_zero': partial(sigma_zero),
        }

        if attack_name in test_attacks:
            attack_func = test_attacks[attack_name]
        else:
            raise ValueError("Unknown attack: " + attack_name)

        torch.cuda.empty_cache()
        stats = run_attack(
            model=model,
            loader=dataloaders["val"],
            attack=(attack_name, partial(attack_func, **attack_params)),
            return_adv=True
        )
        torch.cuda.empty_cache()

        experiment_name = generate_experiment_name()
        process_and_save_results(experiment_name, stats, {"seed": config_data["seed"], "config": experiment}, dataset,
                                 model_name, attack_name, n_samples, batch_size)


if __name__ == "__main__":
    main()
