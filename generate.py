# One way to be sure we don't grab any cuda
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from functools import partial
from multiprocessing import Pool
from io import StringIO
import argparse
import torch
import wandb
from pathlib import Path
import ast
from omegaconf import OmegaConf

from wyckoff_transformer.generator import WyckoffGenerator
from cascade_transformer.model import CascadeTransformer
from wyckoff_transformer.tokenization import (
    load_tensors_and_tokenisers, tensor_to_pyxtal,
    get_letter_from_ss_enum_idx, get_wp_index)


def main():
    parser = argparse.ArgumentParser(description="Generate structures using a Wyckoff transformer.")
    parser.add_argument("wandb_run", type=str, help="The W&B run ID.")
    parser.add_argument("--rare-sg-threshold", type=int, default=10,
                        help="The threshold of underrepresented start_tokens (usually space groups)"
                        " to be dropped.")
    parser.add_argument("--n-tries", type=int, default=1000,
                        help="The number of generation tries before giving up on invalid structures.")
    args = parser.parse_args()
    device = torch.device("cpu")
    run = wandb.init(id=args.wandb_run, resume=True)
    config = OmegaConf.create(dict(wandb.config))
    # The start tokens will be sampled from the train+validation datasets,
    # to preserve the sanctity of the test dataset and ex nihilo generation.
    config.tokeniser = ast.literal_eval(config.tokeniser)
    config.model = ast.literal_eval(config.model)
    tensors, tokenisers = load_tensors_and_tokenisers(config.dataset, config.tokeniser.name)    
    generation_size = len(next(iter(tensors["test"].values())))
    del tensors["test"]
    # First, we want to compute the unbiased validity number
    # Then we drop the underrepresented groups
    max_start = len(tokenisers[config.model.start_token])
    start_counts = torch.bincount(tensors["train"][config.model.start_token], minlength=max_start) + \
                   torch.bincount(tensors["val"][config.model.start_token], minlength=max_start)
    underrepresented = start_counts < args.rare_sg_threshold
    start_counts[underrepresented] = 0
    start_distribution = torch.distributions.Categorical(probs=start_counts.float())
    
    model = CascadeTransformer.from_config_and_tokenisers(config, tokenisers, device)
    model.load_state_dict(torch.load(Path("runs", args.wandb_run, "best_model_params.pt"), map_location=device))
    # We need to grab any tensor from the train dataset
    max_sequence_len = tensors["train"][config.model.cascade_order[0]].size(1)
    masks = {field: tokenisers[field].mask_token for field in config.model.cascade_order}
    generator = WyckoffGenerator(model, config.model.cascade_order, masks, max_sequence_len)
    start = start_distribution.sample((generation_size,))
    generated_tensors = torch.stack(generator.generate_tensors(start), dim=-1)

    letter_from_ss_enum_idx = get_letter_from_ss_enum_idx(tokenisers['sites_enumeration'])
    to_pyxtal = partial(tensor_to_pyxtal,
                        tokenisers=tokenisers,
                        cascade_order=config.model.cascade_order,
                        letter_from_ss_enum_idx=letter_from_ss_enum_idx,
                        wp_index=get_wp_index())
    import time
    start_time = time.time()
    #with Pool() as p:
    #    generated_wp = p.starmap(to_pyxtal, zip(start.detach().cpu(), generated_tensors.detach().cpu()))
    from itertools import starmap
    generated_wp = list(starmap(to_pyxtal, zip(start.detach().cpu(), generated_tensors.detach().cpu())))
    end = time.time()
    print(end - start_time)
    generated_wp = [s for s in generated_wp if s is not None]
    wp_formal_validity = len(generated_wp) / generation_size
    print(f"Wyckchoffs formal validity: {wp_formal_validity}")
    import numpy as np
    print(np.mean([len(s['species']) for s in generated_wp]))
    #print("Generating structutes with Pyxtal.")
    #with Pool() as p:
    #    valid_structures = p.map(pyxtal_generate, generated_wp)
    #valid_structures = [s for s in valid_structures if s is not None]
    #pyxtal_success_rate = len(valid_structures) / len(generated_wp)
    #print(f"Pyxtal success rate: {pyxtal_success_rate}")
    #valid_strucutes = [wandb.Molecule(s, fmt="cif") for s in valid_structures]
    #wandb.log({"structures": valid_strucutes}, commit=True)

    #is_valid = torch.zeros(generation_size, dtype=bool, device=device, requires_grad=False)
    #valid_wykchoffs = [None] * generation_size
    #while not is_valid.all():
    #    generated_tensors = torch.stack(generator.generate_tensors(space_groups[~is_valid]), dim=-1)
    #    indices = torch.nonzero(~is_valid).squeeze(1)
    #    for i, tensors, space_group in zip(indices, generated_tensors, space_groups[~is_valid]):
    #        generated = tensors_to_pyxtal(tensors, space_group,
    #            enforced_max_elements=enforced_max_elements, enforced_min_elements=enforced_min_elements)
    #        if generated is not None:
    #            valid_generated[i] = generated
    #            is_valid[i] = True
    #    print(is_valid.sum().item(), len(space_groups), is_valid.sum().item() / len(space_groups))
    #    print(torch.unique(space_groups[~is_valid]))
if __name__ == "__main__":
    main()
