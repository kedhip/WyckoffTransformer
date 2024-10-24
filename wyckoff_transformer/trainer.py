from typing import Tuple, Dict, Optional
from random import randint
from functools import partial
from pathlib import Path
import gzip
import json
import pickle
import logging
from multiprocessing import Pool
import numpy as np
import torch
from torch import nn
from torch import Tensor
from omegaconf import OmegaConf
import wandb
from tqdm import trange


from cascade_transformer.dataset import AugmentedCascadeDataset, TargetClass
from cascade_transformer.model import CascadeTransformer
from wyckoff_transformer.tokenization import (
    load_tensors_and_tokenisers, tensor_to_pyxtal,
    get_letter_from_ss_enum_idx, get_wp_index)
from wyckoff_transformer.generator import WyckoffGenerator
from wyckoff_transformer.evaluation import (
    evaluate_and_log, StatisticalEvaluator, smac_validity_from_counter)


class WyckoffTrainer():
    def __init__(
        self,
        model: nn.Module,
        train_dataset: Dict[str, torch.tensor],
        val_dataset: Dict[str, torch.tensor],
        tokenisers: dict,
        token_engineers: dict,
        cascade_order: Tuple[str],
        cascade_is_target: Dict[str, bool],
        augmented_field: str|None,
        start_name: str,
        start_dtype: torch.dtype,
        target: TargetClass|str,
        evaluation_samples: int,
        multiclass_next_token_with_order_permutation: bool,
        optimisation_config: dict,
        device: torch.DeviceObjType,
        batch_size: Optional[int] = None,
        run_path: Optional[Path] = Path("runs"),
        target_name = None,
        weights_path: Optional[Path] = None
    ):
        """
        Args:
            multiclass_next_token_with_order_permutation: Train a permutation-invariant model by permuting the sequences,
                    If the target is the next token, predict the 0-th cascade field as a multiclass target.
                    You might also want to ensure that the model is permutation-invariant; pay attention to the positional encoding.
            target: the target to predict. One of:
                next_token: predict the next token in the cascade&sequence.
                num_unique_tokens: predict the number of unique tokens in the cascade&sequence encountered so far.
                    Intended for debugging the ability of the model to count.
        """
        is_target_in_order = [cascade_is_target[field] for field in cascade_order]
        if not all(is_target_in_order[:-1]):
            raise NotImplementedError("Only one not targret field is supported at the moment and it must be the last")
        self.cascade_target_count = sum(is_target_in_order)
        self.token_engineers = token_engineers
        self.cascade_is_target = cascade_is_target
        # Nothing else will work in foreseeable future
        self.dtype = torch.int64
        self.batch_size = batch_size
        self.run_path = run_path / wandb.run.id
        if isinstance(target, str):
            target = TargetClass[target]
        if target == TargetClass.NextToken:
            # Sequences have difference lengths, so we need to make sure that
            # long sequences don't dominate the loss, so we don't average the loss
            self.criterion = nn.CrossEntropyLoss(reduction="sum")
        elif target == TargetClass.NumUniqueTokens:
            if not multiclass_next_token_with_order_permutation:
                raise NotImplementedError("NumUniqueTokens is not implemented without permutations")
            self.criterion = nn.MSELoss(reduction="none")
        elif target == TargetClass.Scalar:
            self.criterion = nn.MSELoss(reduction='mean')
            self.testing_criterion = nn.L1Loss(reduction='mean')  
        else:
            raise ValueError(f"Unknown target: {target}")
        
        self.model = model
        self.tokenisers = tokenisers
        self.device = device
        self.augmented_field = augmented_field

        masks_dict = {field: tokenisers[field].mask_token for field in cascade_order}
        pad_dict = {field: tokenisers[field].pad_token for field in cascade_order}
        stops_dict = {field: tokenisers[field].stop_token for field in cascade_order}
        num_classes_dict = {field: len(tokenisers[field]) for field in cascade_order}

        self.train_dataset = AugmentedCascadeDataset(
            data=train_dataset,
            cascade_order=cascade_order,
            masks=masks_dict,
            pads=pad_dict,
            stops=stops_dict,
            num_classes=num_classes_dict,
            start_field=start_name,
            augmented_field=augmented_field,
            batch_size=batch_size,
            dtype=self.dtype,
            start_dtype=start_dtype,
            device=self.device,
            target_name=target_name
            )
        
        if "lr_per_sqrt_n_samples" in optimisation_config.optimiser:
            if "config" in optimisation_config.optimiser and "lr" in optimisation_config.optimiser.config:
                raise ValueError("Cannot specify both lr and lr_per_sqrt_n_samples")
            if batch_size is None:
                samples_per_step = len(self.train_dataset)
            else:
                samples_per_step = batch_size
            optimisation_config.optimiser.update(
                {"config": {"lr": optimisation_config.optimiser.lr_per_sqrt_n_samples * samples_per_step**0.5}})
                
        self.optimizer = getattr(torch.optim, optimisation_config.optimiser.name)(
            model.parameters(), **optimisation_config.optimiser.config)
        self.scheduler = getattr(torch.optim.lr_scheduler, optimisation_config.scheduler.name)(
            self.optimizer, 'min', **optimisation_config.scheduler.config)

        self.val_dataset = AugmentedCascadeDataset(
            data=val_dataset,
            cascade_order=cascade_order,
            masks=masks_dict,
            pads=pad_dict,
            stops=stops_dict,
            num_classes=num_classes_dict,
            start_field=start_name,
            augmented_field=augmented_field,
            dtype=self.dtype,
            start_dtype=start_dtype,
            device=device,
            target_name=target_name
            )

        assert self.train_dataset.max_sequence_length == self.val_dataset.max_sequence_length
    
        self.clip_grad_norm = optimisation_config.clip_grad_norm
        self.cascade_len = len(cascade_order)
        self.cascade_order = cascade_order
        self.epochs = optimisation_config.epochs
        if weights_path is not None:
            self.model.load_state_dict(torch.load(weights_path))

        self.validation_period = optimisation_config.validation_period
        self.early_stopping_patience_epochs = optimisation_config.early_stopping_patience_epochs    
        self.target = target
        self.multiclass_next_token_with_order_permutation = multiclass_next_token_with_order_permutation
        self.evaluation_samples = evaluation_samples
        if batch_size is None:
            self.batches_per_epoch = 1
        else:
            # Round up, as the last batch might be smaller, but is still a batch
            self.batches_per_epoch = -(-len(self.train_dataset) // batch_size)


    def get_loss(self,
        dataset: AugmentedCascadeDataset,
        known_seq_len: int,
        known_cascade_len: int,
        no_batch: bool,
        testing: bool = False) -> Tensor:
        logging.debug("Known sequence length: %i", known_seq_len)
        logging.debug("Known cascade lenght: %i", known_cascade_len)
        if self.multiclass_next_token_with_order_permutation:
            if self.target == TargetClass.Scalar:
                ValueError("Scalar prediction is computed for the whole sequence, "
                    "it doesn't make sense with multiclass_next_token_with_order_permutation")
            if self.target == TargetClass.NextToken:
                # Once we have sampled the first cascade field, the prediction target is no longer mutliclass
                # However, we still need to permute the sequence so that the autoregression is
                # permutation-invariant.
                start_tokens, masked_data, target = dataset.get_masked_multiclass_cascade_data(
                    known_seq_len, known_cascade_len, multiclass_target=(known_cascade_len == 0),
                    target_type=self.target)
            elif self.target == TargetClass.NumUniqueTokens:
                start_tokens, masked_data, target = dataset.get_masked_multiclass_cascade_data(
                    known_seq_len, known_cascade_len, multiclass_target=False, target_type=self.target)
                logging.debug("Target: %s", target)
                # Counts are integers, as they should be, but MSE needs a float
                target = target.float()
        else:
            if self.target == TargetClass.Scalar:
                start_tokens, masked_data, target, mask = dataset.get_masked_cascade_data(known_seq_len, known_cascade_len)
            else:
                start_tokens, masked_data, target = dataset.get_masked_cascade_data(known_seq_len, known_cascade_len)
        if self.target == TargetClass.NextToken:    
            # No padding, as we have already discarded the padding
            prediction = self.model(start_tokens, masked_data, None, known_cascade_len)
        elif self.target == TargetClass.NumUniqueTokens:
            # No padding, as we have already discarded the padding
            prediction = self.model(start_tokens, masked_data, None, None)
        elif self.target == TargetClass.Scalar:
            prediction = self.model(start_tokens, masked_data, mask, None).squeeze()
        if testing:
            return self.testing_criterion(prediction, target)
        return self.criterion(prediction, target)


    def train_epoch(self):
        self.model.train()
        for _ in range(self.batches_per_epoch):
            self.optimizer.zero_grad(set_to_none=True)
            known_seq_len = randint(0, self.train_dataset.max_sequence_length - 1)
            if self.target == TargetClass.NextToken:
                known_cascade_len = randint(0, self.cascade_target_count - 1)
            elif self.target == TargetClass.NumUniqueTokens:
                known_cascade_len = 0
            elif self.target == TargetClass.Scalar:
                known_cascade_len = self.cascade_target_count - 1
                known_seq_len = self.train_dataset.max_sequence_length - 1
            loss = self.get_loss(self.train_dataset, known_seq_len, known_cascade_len, False)
            if self.target == TargetClass.NumUniqueTokens:
                # Predictions are [batch_size, cascade_size]
                # Unreduced MSE is [batch_size, cascade_size]
                # We avoid averaging them at the level of self.criterion, so we can log
                # the loss for each cascade field separately.
                loss = loss.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
            self.optimizer.step()
            wandb.log({"loss.batch.train": loss,
                       "known_seq_len": known_seq_len,
                       "known_cascade_len": known_cascade_len})


    def evaluate(self) -> Tensor:
        """
        Evaluates the model by calculating the average loss on the validation dataset.

        Returns:
            The average loss on the validation dataset.
        """
        self.model.eval()
        with torch.no_grad():
            if self.target == TargetClass.Scalar:
                known_cascade_len = self.cascade_target_count - 1
                known_train_seq_len = self.train_dataset.max_sequence_length - 1
                known_test_seq_len = self.val_dataset.max_sequence_length - 1
                train_loss = self.get_loss(
                    self.train_dataset, known_train_seq_len, known_cascade_len, False, True
                )
                val_loss = self.get_loss(
                    self.val_dataset, known_test_seq_len, known_cascade_len, False, True
                )
                return val_loss, train_loss

            val_loss = torch.zeros(self.cascade_len, device=self.device)
            train_loss = torch.zeros(self.cascade_len, device=self.device)
            # Since we are sampling permutations and augmentations, and we rely on the
            # validation loss for early stopping and learning rate scheduling, we need to
            # average the loss over multiple samples.
            for _ in range(self.evaluation_samples):
                for known_seq_len in range(0, self.train_dataset.max_sequence_length):
                    if self.target == TargetClass.NextToken:
                        for known_cascade_len in range(0, self.cascade_target_count):
                            train_loss[known_cascade_len] += self.get_loss(
                                self.train_dataset, known_seq_len, known_cascade_len, True)
                            val_loss[known_cascade_len] += self.get_loss(
                                self.val_dataset, known_seq_len, known_cascade_len, True)
                    else:
                        train_loss += self.get_loss(self.train_dataset, known_seq_len, 0, True).sum(dim=0)
                        val_loss += self.get_loss(self.val_dataset, known_seq_len, 0, True).sum(dim=0)
            # ln(P) = ln p(t_n|t_n-1, ..., t_1) + ... + ln p(t_2|t_1)
            # We are minimising the negative log likelihood of the whole sequences
            val_loss /= self.evaluation_samples * len(self.val_dataset)
            train_loss /= self.evaluation_samples * len(self.train_dataset)
            return val_loss, train_loss


    def train(self):
        best_val_loss = float('inf')
        best_val_epoch = 0
        self.run_path.mkdir(exist_ok=False)
        best_model_params_path = self.run_path / "best_model_params.pt"
        wandb.define_metric("loss.epoch.val.total", step_metric="epoch", goal="minimize")
        wandb.define_metric("loss.epoch.train.total", step_metric="epoch", goal="minimize")
        wandb.define_metric("loss.epoch.val_best", step_metric="epoch", goal="minimize")
        wandb.define_metric("lr", step_metric="epoch")
        wandb.define_metric("known_seq_len", hidden=True)
        wandb.define_metric("known_cascade_len", hidden=True)
        if self.epochs == 0:
            val_loss_epoch, train_loss_epoch = self.evaluate()
            print(val_loss_epoch, train_loss_epoch)
            return

        for epoch in trange(self.epochs):
            self.train_epoch()
            if epoch % self.validation_period == 0 or epoch == self.epochs - 1:
                if self.target == TargetClass.Scalar:
                    val_loss_epoch, train_loss_epoch = self.evaluate()
                    total_val_loss = val_loss_epoch
                    wandb.log({"train_mae": train_loss_epoch.item(),
                           "val_mae": val_loss_epoch.item(),
                           "lr": self.optimizer.param_groups[0]['lr'],
                           "epoch": epoch}, commit=False)
                else:
                    val_loss_dict = {name: val_loss_epoch[i] for i, name in enumerate(self.cascade_order)}
                    total_val_loss = val_loss_epoch.sum()
                    val_loss_dict["total"] = total_val_loss.item()
                    train_loss_dict = {name: train_loss_epoch[i] for i, name in enumerate(self.cascade_order)}
                    train_loss_dict["total"] = train_loss_epoch.sum().item()
                    wandb.log({"loss.epoch": {"val": val_loss_dict, "train": train_loss_dict},
                                "lr": self.optimizer.param_groups[0]['lr'],
                                "epoch": epoch}, commit=False)
                if total_val_loss < best_val_loss:
                    best_val_loss = total_val_loss
                    best_val_epoch = epoch
                    torch.save(self.model.state_dict(), best_model_params_path)
                    wandb.save(best_model_params_path, base_path=self.run_path, policy="live")
                    print(f"Epoch {epoch}; loss_epoch.val {total_val_loss} saved to {best_model_params_path}")
                    wandb.log({"loss.epoch.val_best": best_val_loss}, commit=False)
                if epoch - best_val_epoch > self.early_stopping_patience_epochs:
                    print(f"Early stopping at epoch {epoch} after more than {self.early_stopping_patience_epochs}"
                           " epochs without improvement")
                    break
                # Don't step the scheduler on the tail epoch, to presereve
                # patience behaviour
                if epoch % self.validation_period == 0:
                    self.scheduler.step(total_val_loss)
        # Make sure we log the last evaluation results
        wandb.log(dict(), commit=True)


    def generate_structures(self, n_structures: int, calibrate: bool):
        generator = WyckoffGenerator(
            self.model, self.cascade_order, self.cascade_is_target, self.token_engineers,
             self.train_dataset.masks, self.train_dataset.max_sequence_length)
        if calibrate:
            generator.calibrate(self.val_dataset)
        if self.model.start_type == "categorial":
            max_start = self.model.start_embedding.num_embeddings
            start_counts = torch.bincount(
                self.train_dataset.start_tokens, minlength=max_start) + torch.bincount(
                    self.val_dataset.start_tokens, minlength=max_start)
            start_tensor = torch.distributions.Categorical(probs=start_counts.float()).sample((n_structures,))
        elif self.model.start_type == "one_hot":
            # We are going to sample with replacement from train+val datasets
            all_starts = torch.cat([self.train_dataset.start_tokens, self.val_dataset.start_tokens], dim=0)
            start_tensor = all_starts[torch.randint(len(all_starts), (n_structures,))]
        generated_tensors = torch.stack(
            generator.generate_tensors(start_tensor), dim=-1)

        letter_from_ss_enum_idx = get_letter_from_ss_enum_idx(self.tokenisers['sites_enumeration'])
        to_pyxtal = partial(tensor_to_pyxtal,
                            tokenisers=self.tokenisers,
                            cascade_order=self.cascade_order,
                            letter_from_ss_enum_idx=letter_from_ss_enum_idx,
                            wp_index=get_wp_index())
        with Pool() as p:
            structures = p.starmap(to_pyxtal, zip(start_tensor.detach().cpu(), generated_tensors.detach().cpu()))
        valid_structures = [s for s in structures if s is not None]
        return valid_structures
    
    def generate_evaluate_and_log_wp(
        self,
        generation_name: str,
        calibrate: bool,
        n_structures: int,
        evaluator: StatisticalEvaluator):
        
        generated_wp = self.generate_structures(n_structures, calibrate)
        file_name = self.run_path / f"generated_wp_{generation_name}.json.gz"
        with gzip.open(file_name, "wt") as f:
            json.dump(generated_wp, f)
        wandb.save(file_name, base_path=self.run_path, policy="now")
        evaluate_and_log(generated_wp, generation_name, n_structures, evaluator)


def train_from_config(config_dict: dict, device: torch.device, run_path: Optional[Path] = Path("runs")):
    config = OmegaConf.create(config_dict)
    if config.model.WyckoffTrainer_args.multiclass_next_token_with_order_permutation and not config.model.CascadeTransformer_args.learned_positional_encoding_only_masked:
        raise ValueError("Multiclass target with order permutation requires learned positional encoding only masked, ",
                         "otherwise the Transformer is not permutation invariant.")
    tensors, tokenisers, token_engineers = load_tensors_and_tokenisers(config.dataset, config.tokeniser.name)
    model = CascadeTransformer.from_config_and_tokenisers(config, tokenisers, device)
    # Our hihgly dynamic concat-heavy workflow doesn't benefit much from compilation
    # torch._dynamo.config.cache_size_limit = 128
    # model = torch.compile(model, dynamic=True)
    if "augmented_token_fields" in config.tokeniser and len(config.tokeniser.augmented_token_fields) == 1:
        augmented_field = config.tokeniser.augmented_token_fields[0]
    else:
        augmented_field = None
    if config.model.CascadeTransformer_args.start_type == "categorial":
        start_dtype = torch.int64
    # one-hots are encoded by a linear layer
    elif config.model.CascadeTransformer_args.start_type == "one_hot":
        start_dtype = torch.float32
    else:
        raise ValueError(f"Unknown start type: {config.model.CascadeTransformer_args.start_type}")
    trainer = WyckoffTrainer(
        model, tensors["train"], tensors["val"], tokenisers, token_engineers, config.model.cascade.order,
        config.model.cascade.is_target,
        augmented_field,
        config.model.start_token,
        optimisation_config=config.optimisation, device=device,
        run_path=run_path,
        start_dtype=start_dtype,
        **config.model.WyckoffTrainer_args)
    trainer.train()
    if config.model.WyckoffTrainer_args.target == "NextToken":
        print("Training complete, loading the best model")
        model.load_state_dict(torch.load(trainer.run_path / "best_model_params.pt"))
        data_cache_path = Path(__file__).parent.parent.resolve() / "cache" / config.dataset / "data.pkl.gz"
        with gzip.open(data_cache_path, "rb") as f:
            datasets_pd = pickle.load(f)
        del datasets_pd["train"]
        del datasets_pd["val"]

        test_no_sites = datasets_pd['test']['site_symmetries'].map(len).values
        num_sites_bins = np.arange(0, 21)
        test_no_sites_hist = np.histogram(test_no_sites, bins=num_sites_bins)
        wandb.run.summary["num_sites"] = {"test": {"hist": wandb.Histogram(np_histogram=test_no_sites_hist)}}

        print(f"Test dataset size: {len(datasets_pd['test']['site_symmetries'])}")
        wandb.run.summary["test_dataset_size"] = len(datasets_pd["test"]["site_symmetries"])
        evaluator = StatisticalEvaluator(datasets_pd["test"])
        test_smact_validity = datasets_pd["test"]["composition"].map(smac_validity_from_counter).mean()
        print(f"SMAC-T validity on the test dataset: {test_smact_validity}")
        wandb.run.summary["smact_validity"] = {"test": test_smact_validity}
        wandb.run.summary["formal_validity"] = {}
        wandb.run.summary["wp"] = {}
        trainer.generate_evaluate_and_log_wp(
            "no_calibration", calibrate=False, n_structures=config.evaluation.n_structures_to_generate, evaluator=evaluator)
        trainer.generate_evaluate_and_log_wp(
            "temperature_calibration", calibrate=True, n_structures=config.evaluation.n_structures_to_generate, evaluator=evaluator)