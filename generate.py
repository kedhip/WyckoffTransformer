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
    parser.add_argument("--update-wandb", action="store_true", help="Update the W&B run with the "
        "generated structures and quality metrics.")
    parser.add_argument("--calibrate", action="store_true", help="Calibrate the generator.")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode.")
    args = parser.parse_args()
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    if args.output.suffixes != [".json", ".gz"]:
        raise ValueError("Output file must be a .json.gz file.")
    device = torch.device("cpu")
    if args.update_wandb:
        wandb_run = wandb.init(project="WyckoffTransformer", id=args.wandb_run, resume=True)
    else:
        wandb_run = wandb.Api().run(f"WyckoffTransformer/{args.wandb_run}")
    config = OmegaConf.create(dict(wandb_run.config))

    # The start tokens will be sampled from the train+validation datasets,
    # to preserve the sanctity of the test dataset and ex nihilo generation.
    tensors, tokenisers, engineers = load_tensors_and_tokenisers(config.dataset, config.tokeniser.name)    
    generation_size = config.evaluation.n_structures_to_generate
    del tensors["test"]
    
    model = CascadeTransformer.from_config_and_tokenisers(config, tokenisers, device)
    model.load_state_dict(torch.load(Path("runs", args.wandb_run, "best_model_params.pt"), map_location=device))
    # We need to grab any tensor from the train dataset
    max_sequence_len = tensors["train"][config.model.cascade.order[0]].size(1)
    
    masks_dict = {field: tokenisers[field].mask_token for field in config.model.cascade.order}
    pad_dict = {field: tokenisers[field].pad_token for field in config.model.cascade.order}
    stops_dict = {field: tokenisers[field].stop_token for field in config.model.cascade.order}
    num_classes_dict = {field: len(tokenisers[field]) for field in config.model.cascade.order}

    generator = WyckoffGenerator(model,
        config.model.cascade.order,
        config.model.cascade.is_target,
        engineers,
        masks_dict, max_sequence_len)
    if config.model.CascadeTransformer_args.start_type == "one_hot":
        start_dtype = torch.float
    elif config.model.CascadeTransformer_args.start_type == "categorial":
        start_dtype = torch.long
    if args.calibrate:
        validation_dataset = AugmentedCascadeDataset(
            data=tensors["val"],
            cascade_order=config.model.cascade.order,
            masks=masks_dict,
            pads=pad_dict,
            stops=stops_dict,
            num_classes=num_classes_dict,
            start_field=config.model.start_token,
            augmented_field=config.tokeniser.augmented_token_fields[0],
            dtype=torch.long,
            start_dtype=start_dtype,
            device=device)
        generator.calibrate(validation_dataset)

    # Should we maybe sample wih replacement?
    all_starts = torch.cat([tensors["train"][config.model.start_token], tensors["val"][config.model.start_token]], dim=0)
    if generation_size > len(all_starts):
        raise NotImplementedError("Naive sampling doesn't support the larger generation sizes.")
    permutation = torch.randperm(all_starts.size(0))
    start = all_starts[permutation[:generation_size]].to(dtype=start_dtype, device=device)
    generated_tensors = torch.stack(generator.generate_tensors(start), dim=-1)

    letter_from_ss_enum_idx = get_letter_from_ss_enum_idx(tokenisers['sites_enumeration'])
    to_pyxtal = partial(tensor_to_pyxtal,
                        tokenisers=tokenisers,
                        cascade_order=config.model.cascade.order,
                        letter_from_ss_enum_idx=letter_from_ss_enum_idx,
                        wp_index=get_wp_index())
    with Pool() as p:
        generated_wp = p.starmap(to_pyxtal, zip(start.detach().cpu(), generated_tensors.detach().cpu()))

    generated_wp = [s for s in generated_wp if s is not None]
    wp_formal_validity = len(generated_wp) / generation_size
    print(f"Wyckchoffs formal validity: {wp_formal_validity}")
    with gzip.open(args.output, "wt") as f:
        json.dump(generated_wp, f)


if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    main()
