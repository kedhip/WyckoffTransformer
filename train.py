import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "backend:cudaMallocAsync"
import torch
import argparse
from mp_20_utils import load_all_data
from cascade_transformer.model import CascadeTransformer
from wyckoff_transformer import WyckoffTrainer
from tokenization import PAD_TOKEN, MASK_TOKEN


def main():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument("device", type=str, help="Device to train on")
    parser.add_argument("--dataset", type=str, default="mp_20_biternary", help="Dataset to use")
    parser.add_argument("--n-head", type=int, default=1, help="Number of attention heads")
    parser.add_argument("--d-hid", type=int, default=512, help="Hidden dimension")
    parser.add_argument("--n-layers", type=int, default=8, help="Number of layers")
    args = parser.parse_args()

    if args.device.startswith("cuda"):
        # UserWarning: TensorFloat32 tensor cores for float32 matrix multiplication available but not enabled. Consider setting `torch.set_float32_matmul_precision('high')` for better performance.
        torch.set_float32_matmul_precision('high')

    datasets_pd, torch_datasets, site_to_ids, element_to_ids, spacegroup_to_ids, max_len, \
        max_enumeration, enumeration_stop, enumeration_pad = \
            load_all_data(dataset=args.dataset)
    del datasets_pd
    del torch_datasets["test"]
    print(max_len, max_enumeration, enumeration_stop, enumeration_pad)

    n_space_groups = len(spacegroup_to_ids)
    # Not all 230 space groups are present in the data
    # Embedding doesn't support uint8. Sad!
    dtype = torch.int64
    cascade_order = ("elements", "symmetry_sites", "symmetry_sites_enumeration")
    # (N_i, d_i, pad_i)
    assert max_enumeration + 1 == enumeration_stop
    assert max_enumeration + 2 == enumeration_pad
    enumeration_mask = max_enumeration + 3
    assert enumeration_mask < torch.iinfo(dtype).max

    cascade = (
        (len(element_to_ids), 128, torch.tensor(element_to_ids[PAD_TOKEN], dtype=dtype, device=args.device)),
        (len(site_to_ids), 128 - 1, torch.tensor(site_to_ids[PAD_TOKEN], dtype=dtype, device=args.device)),
        (enumeration_mask + 1, None, torch.tensor(enumeration_pad, dtype=dtype, device=args.device))
    )
    model = CascadeTransformer(
        n_start=n_space_groups,
        cascade=cascade,
        n_head=args.n_head,
        d_hid=args.d_hid,
        n_layers=args.n_layers,
        dropout=0.1,
        use_mixer=True).to(args.device)

    pad_dict = {
        "elements": element_to_ids[PAD_TOKEN],
        "symmetry_sites": site_to_ids[PAD_TOKEN],
        "symmetry_sites_enumeration": enumeration_pad
    }
    mask_dict = {
        "elements": element_to_ids[MASK_TOKEN],
        "symmetry_sites": site_to_ids[MASK_TOKEN],
        "symmetry_sites_enumeration": enumeration_mask
    }
    trainer = WyckoffTrainer(
        model, torch_datasets, pad_dict, mask_dict, cascade_order, "spacegroup_number", max_len, args.device, dtype=dtype)
    trainer.train(epochs=20000)

if __name__ == '__main__':
    main()