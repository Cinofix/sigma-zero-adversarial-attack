import json


def generate_config(config_file_path, attack_name, attack_params, dataset, models, batch_sizes, n_samples):
    """
    Generate configuration file for executing experiments.
    """
    config = {
        "seed": 1233,
        "experiments": []
    }

    for model, batch_size in zip(models, batch_sizes):
        experiment = {
            "attack": {
                "name": attack_name,
                "params": attack_params
            },
            "dataset": dataset,
            "model": model,
            "n_samples": n_samples,
            "batch_size": batch_size
        }
        config["experiments"].append(experiment)

    with open(config_file_path, "w") as config_file:
        json.dump(config, config_file, indent=4)

    print(f"Configuration file {config_file_path} generated and saved.")


def main():
    # attack name
    attack_name = "sigma_zero"
    # attack parameters
    attack_params = {
        "steps": 1000
    }
    # number of samples (0 means all validation set)
    n_samples = 32
    dataset = "cifar10"
    models = [
        "standard",
        "chen2020"
        "addepalli2022"
    ]
    batch_sizes = [16, 16]

    config_file_path = f"configs/{attack_name}_{dataset}_config.json"

    generate_config(config_file_path, attack_name, attack_params, dataset, models, batch_sizes, n_samples)


if __name__ == "__main__":
    main()
