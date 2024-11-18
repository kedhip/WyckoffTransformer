from pathlib import Path
import argparse
import torch
from omegaconf import OmegaConf
import wandb
from wyckoff_transformer.trainer import WyckoffTrainer

def main():
    parser = argparse.ArgumentParser(description='Compute test loss for a WanDB run')
    parser.add_argument("--device", type=torch.device, default=torch.device("cpu"), help="Device to run on")
    parser.add_argument("wandb_run", type=str, help="The W&B run ID.")
    args = parser.parse_args()
    wandb_run = wandb.Api().run(f"WyckoffTransformer/{args.wandb_run}")
    config = OmegaConf.create(dict(wandb_run.config))
    trainer = WyckoffTrainer.from_config(config, args.device, Path("runs", args.wandb_run))
    trainer.model.load_state_dict(torch.load(trainer.run_path / "best_model_params.pt",
        weights_only=False, map_location=args.device))
    print(trainer.evaluate(trainer.test_dataset))

if __name__ == '__main__':
    main()