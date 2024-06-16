from typing import Tuple, List, Dict
import numpy as np
import logging
import torch
from torch import nn, Tensor
from sklearn.isotonic import IsotonicRegression

from cascade_transformer.dataset import AugmentedCascadeDataset, TargetClass

logger = logging.getLogger(__name__)

class TemperatureScaling(nn.Module):
    def __init__(self):
        super(TemperatureScaling, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits):
        # Apply temperature scaling
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        # Temperature scaling of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    def fit(self, logits, labels):
        # Fit the temperature parameter using the validation data
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval():
            optimizer.zero_grad()
            loss = criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss

        logger.info("Cross entropy before calibration: %f", criterion(logits, labels).item())
        optimizer.step(eval)
        logger.info("Cross entropy after calibration: %f", criterion(self.temperature_scale(logits), labels).item())
        logger.info("Temperature: %f", self.temperature.item())
        return self


class WyckoffGenerator():
    def __init__(self,
                 model: nn.Module,
                 cascade_order: Tuple,
                 cascade_is_target: Dict,
                 token_engineers: Dict,
                 masks: dict,
                 max_sequence_len: int):
        self.model = model
        self.max_sequence_len = max_sequence_len
        self.cascade_order = cascade_order
        self.cascade_is_target = cascade_is_target
        self.masks = masks
        self.calibrators = None
        self.tail_calibrators = None
        self.token_engineers = token_engineers


    def calibrate(self, dataset: AugmentedCascadeDataset, calibration_element_count_threshold: int = 100):
        """
        The calibraiton is going to be per cascade field.
        We will generate p_predicted and p_true for each cascade field for each
        known sequence length.
        """
        assert dataset.cascade_order == self.cascade_order
        with torch.no_grad():
            self.model.eval()
            self.calibrators = []
            self.tail_calibrators = []
            for known_cascade_len, cascade_name in enumerate(self.cascade_order):
                tail_predictions = []
                tail_targets = []
                self.calibrators.append([])
                if not self.cascade_is_target[cascade_name]:
                    logging.info("Cascade field %s is not a target doesn't need calibration", cascade_name)
                    continue
                logging.info("Calibrating cascade field %s", cascade_name)
                for known_seq_len in range(dataset.max_sequence_length):
                    if known_cascade_len == 0:
                        start_tokens, masked_data, target = dataset.get_masked_multiclass_cascade_data(
                            known_seq_len, known_cascade_len,
                            target_type=TargetClass.NextToken, multiclass_target=True)
                    else:
                        start_tokens, masked_data, target = dataset.get_masked_multiclass_cascade_data(
                            known_seq_len, known_cascade_len,
                            target_type=TargetClass.NextToken, multiclass_target=False)
                    model_output = self.model(start_tokens, masked_data, None, known_cascade_len)
                    # Enought data for separate calibration
                    if target.size(0) >= calibration_element_count_threshold:
                        # If model_output and target are on different devices, not our problem
                        self.calibrators[known_cascade_len].append(TemperatureScaling().to(target.device).fit(
                            model_output, target))
                    # Not enough data for separate calibration, so we gather all the data in the tail
                    else:
                        tail_predictions.append(model_output)
                        tail_targets.append(target)
                tail_predictions = torch.concatenate(tail_predictions, axis=0)
                tail_targets = torch.concatenate(tail_targets, axis=0)
                if tail_predictions.size(0) < calibration_element_count_threshold:
                    logger.warning("Tail too small, %i when requested %i", tail_predictions.size(0), calibration_element_count_threshold)
                self.tail_calibrators.append(TemperatureScaling().to(tail_targets.device).fit(tail_predictions, tail_targets))
        

    def generate_tensors(self, start: Tensor) -> List[Tensor]:
        """
        Generates a sequence of tokens.

        Arguments:
            start: The start token. It should be a tensor of shape [batch_size].

        Returns:
            The generated sequence of tokens. It has shape [batch_size, max_len, len(cascade_order)].
            It doesn't include the start token.
        """
        with torch.no_grad():
            self.model.eval()
            batch_size = start.size(0)
            # Since we are doing in-place operations, we can just pre-assign
            # everything to be a mask
            generated = []
            for field in self.cascade_order:
                generated.append(torch.full((batch_size, self.max_sequence_len), self.masks[field],
                                            dtype=torch.int64, device=start.device))
            cascade_index_by_name = {name: idx for idx, name in enumerate(self.cascade_order)}
            if len(start.size()) > 1:
                start_converted = list(map(tuple, start.tolist()))
            else:
                start_converted = start.tolist()
            for known_seq_len in range(self.max_sequence_len):
                for known_cascade_len, cascade_name in enumerate(self.cascade_order):
                    if self.cascade_is_target[cascade_name]:
                        this_generation_input = [generated_cascade[:, :known_seq_len + 1] for generated_cascade in generated]
                        logits = self.model(start, this_generation_input, None, known_cascade_len)
                        if self.calibrators is not None:
                            if known_seq_len < len(self.calibrators[known_cascade_len]):
                                logits = self.calibrators[known_cascade_len][known_seq_len](logits)
                            else:
                                logits = self.tail_calibrators[known_cascade_len](logits)
                        # binary/ternary hack
                        # if known_cascade_len == 0 and known_seq_len > 2:
                        #    logits *= 3.
                        # CONSIDER: remove probas for all special tokens aside from STOP
                        calibrated_probas = torch.nn.functional.softmax(logits, dim=1)
                        # calibrated_probas = probas.numpy()
                        # calibrated_probas = calibrator.predict_proba(probas.numpy())
                        generated[known_cascade_len][:, known_seq_len] = \
                            torch.multinomial(calibrated_probas, num_samples=1).squeeze()
                    else:
                        if known_cascade_len != len(self.cascade_order) - 1:
                            raise NotImplementedError("Only the last cascade field can be non-target")
                        # If a field is not in the cascade, by exclusion it is the start token
                        this_engineer_input = [generated[cascade_index_by_name[name]][:, known_seq_len].tolist() 
                                                for name in self.token_engineers[cascade_name].inputs
                                                    if name in cascade_index_by_name]
                        #print(this_engineer_input)
                        #print(len(this_engineer_input))
                        #print(this_generation_input[0].shape)
                        generated[known_cascade_len][:, known_seq_len] = \
                            torch.from_numpy(
                                self.token_engineers[cascade_name].get_feature_from_token_batch(start_converted, this_engineer_input))
            return generated
