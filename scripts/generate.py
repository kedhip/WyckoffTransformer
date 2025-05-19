from functools import partial
import time
from multiprocessing import Pool
import argparse
import pickle
import torch
# torch.set_float32_matmul_precision('high')
import json
import gzip
import wandb
import logging
from pathlib import Path
from omegaconf import OmegaConf

from wyckoff_transformer.generator import WyckoffGenerator
from wyckoff_transformer.trainer import WyckoffTrainer
from cascade_transformer.model import CascadeTransformer
from cascade_transformer.dataset import AugmentedCascadeDataset
from wyckoff_transformer.tokenization import (
    load_tensors_and_tokenisers, tensor_to_pyxtal,
    get_letter_from_ss_enum_idx, get_wp_index)


def main():
    parser = argparse.ArgumentParser(description="Generate structures using a Wyckoff transformer.")
    parser.add_argument("wandb_run", type=str, help="The W&B run ID.")
    parser.add_argument("output", type=Path, help="The output file.")
    parser.add_argument("--initial-n-samples", type=int, help="The number of samples to try"
        " before filtering out the invalid ones.")
    parser.add_argument("--firm-n-samples", type=int, help="The number of samples after generation, "
        "subsampling the valid ones if nesessary.")
    parser.add_argument("--update-wandb", action="store_true", help="Update the W&B run with the "
        "generated structures and quality metrics.")
    parser.add_argument("--device", type=torch.device, default=torch.device("cpu"), help="The device to use.")
    parser.add_argument("--calibrate", action="store_true", help="Calibrate the generator.")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode.")
    args = parser.parse_args()
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    if args.output.suffixes != [".json", ".gz"]:
        raise ValueError("Output file must be a .json.gz file.")

    if args.update_wandb:
        wandb_run = wandb.init(project="WyckoffTransformer", id=args.wandb_run, resume=True)
    else:
        wandb_run = wandb.Api().run(f"WyckoffTransformer/{args.wandb_run}")
    config = OmegaConf.create(dict(wandb_run.config))

    # The start tokens will be sampled from the train+validation datasets,
    # to preserve the sanctity of the test dataset and ex nihilo generation.
    #tensors, tokenisers, engineers = load_tensors_and_tokenisers(config.dataset, config.tokeniser.name)
    generation_start_time = time.time()
    trainer = WyckoffTrainer.from_config(
        config, device=args.device, run_path=Path(__file__).parent.parent / "runs" / args.wandb_run) # Adjusted path
    trainer.model.load_state_dict(torch.load(trainer.run_path / "best_model_params.pt", weights_only=True))
    generated_wp = trainer.generate_structures(args.initial_n_samples, args.calibrate)
    generation_end_time = time.time()
    print(f"Generation in total took {generation_end_time - generation_start_time} seconds")
    # print(f"Tensor generation took {tensor_generated_time - generation_start_time} seconds")
    # print(f"Detokenizing took {generation_end_time - tensor_generated_time} seconds")
    # wp_formal_validity = len(generated_wp) / generation_size
    # print(f"Wyckchoffs formal validity: {wp_formal_validity}")
    if args.firm_n_samples is not None:
        if len(generated_wp) >= args.firm_n_samples:
            generated_wp = generated_wp[:args.firm_n_samples]
        else:
            raise ValueError("Not enough valid structures to subsample.")
    with gzip.open(args.output, "wt") as f:
        json.dump(generated_wp, f)


if __name__ == "__main__":
    main()
