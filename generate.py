from functools import partial
from multiprocessing import Pool
import argparse
import torch
import json
import gzip
import wandb
import logging
from pathlib import Path
from omegaconf import OmegaConf

from wyckoff_transformer.generator import WyckoffGenerator
from cascade_transformer.model import CascadeTransformer
from cascade_transformer.dataset import AugmentedCascadeDataset
from wyckoff_transformer.tokenization import (
    load_tensors_and_tokenisers, tensor_to_pyxtal,
    get_letter_from_ss_enum_idx, get_wp_index)


def main():
    parser = argparse.ArgumentParser(description="Generate structures using a Wyckoff transformer.")
    parser.add_argument("wandb_run", type=str, help="The W&B run ID.")
    parser.add_argument("output", type=Path, help="The output file.")
    parser.add_argument("--calibrate", action="store_true", help="Calibrate the generator.")
    parser.add_argument("--rare-sg-threshold", type=int, default=0,
                        help="The threshold of underrepresented start_tokens (usually space groups)"
                        " to be dropped.")
    parser.add_argument("--n-tries", type=int, default=100,
                        help="The number of generation tries before giving up on invalid structures.")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode.")
    args = parser.parse_args()
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    if args.output.suffixes != [".json", ".gz"]:
        raise ValueError("Output file must be a .json.gz file.")
    device = torch.device("cpu")
    wandb_run = wandb.Api().run(f"WyckoffTransformer/{args.wandb_run}")
    config = OmegaConf.create(dict(wandb_run.config))

    # The start tokens will be sampled from the train+validation datasets,
    # to preserve the sanctity of the test dataset and ex nihilo generation.
    tensors, tokenisers = load_tensors_and_tokenisers(config.dataset, config.tokeniser.name)    
    generation_size = 15000#len(next(iter(tensors["test"].values())))*2
    del tensors["test"]
    
    max_start = len(tokenisers[config.model.start_token])
    start_counts = torch.bincount(tensors["train"][config.model.start_token], minlength=max_start)
    underrepresented = start_counts < args.rare_sg_threshold
    start_counts[underrepresented] = 0
    print(f"Excluded {underrepresented.sum()} underrepresented start tokens.")
    start_distribution = torch.distributions.Categorical(probs=start_counts.float())
    
    model = CascadeTransformer.from_config_and_tokenisers(config, tokenisers, device)
    model.load_state_dict(torch.load(Path("runs", args.wandb_run, "best_model_params.pt"), map_location=device))
    # We need to grab any tensor from the train dataset
    max_sequence_len = tensors["train"][config.model.cascade_order[0]].size(1)
    
    masks_dict = {field: tokenisers[field].mask_token for field in config.model.cascade_order}
    pad_dict = {field: tokenisers[field].pad_token for field in config.model.cascade_order}
    stops_dict = {field: tokenisers[field].stop_token for field in config.model.cascade_order}
    num_classes_dict = {field: len(tokenisers[field]) for field in config.model.cascade_order}

    generator = WyckoffGenerator(model, config.model.cascade_order, masks_dict, max_sequence_len)

    if args.calibrate:
        selected = ~underrepresented[tensors["val"][config.model.start_token]]
        filtered_tensors = {k: v[selected] for k, v in tensors["val"].items()}
        validation_dataset = AugmentedCascadeDataset(
            data=filtered_tensors,
            cascade_order=config.model.cascade_order,
            masks=masks_dict,
            pads=pad_dict,
            stops=stops_dict,
            num_classes=num_classes_dict,
            start_field=config.model.start_token,
            augmented_field=config.tokeniser.augmented_token_fields[0],
            dtype=torch.long,
            device=device)
        generator.calibrate(validation_dataset)
    start = start_distribution.sample((generation_size,))
    generated_tensors = torch.stack(generator.generate_tensors(start), dim=-1)

    letter_from_ss_enum_idx = get_letter_from_ss_enum_idx(tokenisers['sites_enumeration'])
    to_pyxtal = partial(tensor_to_pyxtal,
                        tokenisers=tokenisers,
                        cascade_order=config.model.cascade_order,
                        letter_from_ss_enum_idx=letter_from_ss_enum_idx,
                        wp_index=get_wp_index())
    with Pool() as p:
        generated_wp = p.starmap(to_pyxtal, zip(start.detach().cpu(), generated_tensors.detach().cpu()))
    #from itertools import starmap
    #generated_wp = list(starmap(to_pyxtal, zip(start.detach().cpu(), generated_tensors.detach().cpu())))
    generated_wp = [s for s in generated_wp if s is not None]
    wp_formal_validity = len(generated_wp) / generation_size
    print(f"Wyckchoffs formal validity: {wp_formal_validity}")
    with gzip.open(args.output, "wt") as f:
        json.dump(generated_wp, f)


if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    main()
