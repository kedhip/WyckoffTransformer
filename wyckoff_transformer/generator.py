from typing import Tuple, List, Dict
import numpy as np
import torch
from torch import nn, Tensor
from sklearn.isotonic import IsotonicRegression

from cascade_transformer.dataset import AugmentedCascadeDataset, TargetClass


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

        print(f"Cross entropy before calibration: {criterion(logits, labels).item()}")
        optimizer.step(eval)
        print(f"Cross entropy after calibration: {criterion(self.temperature_scale(logits), labels).item()}")
        print(f"Temperature: {self.temperature.item()}")
        return self


class VennAbersNaive():
    def __init__(self):
        self.isotonic_models = []

    def fit(self, logits: np.array, targets: np.array):
        for class_idx in range(targets.shape[1]):
            ir = IsotonicRegression(out_of_bounds='clip')
            self.isotonic_models.append(ir.fit(logits[:, class_idx], targets[:, class_idx]))
        return self
    
    def predict_proba(self, logits: np.array):
        calibrated_probs = np.zeros_like(logits)
        for class_idx, ir in enumerate(self.isotonic_models):
            calibrated_probs[:, class_idx] = ir.transform(logits[:, class_idx])
        calibrated_probs /= calibrated_probs.sum(axis=1, keepdims=True)
        return calibrated_probs


def cross_entropy(predictions, targets, epsilon=1e-8):
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    return (-np.log(predictions) * targets).sum(axis=1).mean()


class WyckoffGenerator():
    def __init__(self,
                 model: nn.Module,
                 cascade_order: Tuple,
                 masks: dict,
                 max_sequence_len: int):
        self.model = model
        self.max_sequence_len = max_sequence_len
        self.cascade_order = cascade_order
        self.masks = masks
        self.calibrators = []


    def calibrate(self, dataset: AugmentedCascadeDataset):
        """
        The calibraiton is going to be per cascade field.
        We will generate p_predicted and p_true for each cascade field for each
        known sequence length.
        """
        assert dataset.cascade_order == self.cascade_order
        with torch.no_grad():
            self.model.eval()
            for known_cascade_len, cascade_name in enumerate(self.cascade_order):
                p_predicted = []
                true_targets = []
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
                    # probas = torch.nn.functional.softmax(model_output, dim=1)
                    probas = model_output
                    p_predicted.append(probas)
                    true_targets.append(target)
                p_predicted = torch.concatenate(p_predicted, axis=0)
                true_targets = torch.concatenate(true_targets, axis=0)
                self.calibrators.append(TemperatureScaling().fit(p_predicted, true_targets))
                #print(f"Cross entropy for {cascade_name} before calibration: {cross_entropy(p_predicted, true_targets)}")
                #self.calibrators.append(VennAbersNaive().fit(p_predicted, true_targets))
                #print(f"Cross entropy for {cascade_name} after calibration: {cross_entropy(self.calibrators[-1].predict_proba(p_predicted), true_targets)}")
        

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
                                            dtype=start.dtype, device=start.device))
            for known_seq_len in range(self.max_sequence_len):
                for known_cascade_len, calibrator in enumerate(self.calibrators):
                    this_generation_input = [generated_cascade[:, :known_seq_len + 1] for generated_cascade in generated]
                    # print(this_generation_input[0].size())
                    model_output = self.model(start, this_generation_input, None, known_cascade_len)
                    logits = calibrator(model_output)
                    # CONSIDER: remove probas for all special tokens aside from STOP
                    calibrated_probas = torch.nn.functional.softmax(logits, dim=1)
                    # calibrated_probas = probas.numpy()
                    # calibrated_probas = calibrator.predict_proba(probas.numpy())
                    generated[known_cascade_len][:, known_seq_len] = \
                        torch.multinomial(calibrated_probas, num_samples=1).squeeze()
            return generated
