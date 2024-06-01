from typing import Tuple, List
import torch
from torch import nn, Tensor


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


    def generate_tensors(self, start: Tensor) -> List[Tensor]:
        """
        Generates a sequence of tokens.

        Arguments:
            start: The start token. It should be a tensor of shape [batch_size].

        Returns:
            The generated sequence of tokens. It has shape [batch_size, max_len, len(cascade_order)].
            It doesn't include the start token.
        """
        self.model.eval()
        with torch.no_grad():
            batch_size = start.size(0)
            # Since we are doing in-place operations, we can just pre-assign
            # everything to be a mask
            generated = []
            for field in self.cascade_order:
                generated.append(torch.full((batch_size, self.max_sequence_len), self.masks[field],
                                            dtype=start.dtype, device=start.device))
            for known_seq_len in range(self.max_sequence_len):
                for known_cascade_len in range(len(self.cascade_order)):
                    this_generation_input = [generated_cascade[:, :known_seq_len + 1] for generated_cascade in generated]
                    # print(this_generation_input[0].size())
                    model_output = self.model(start, this_generation_input, None, known_cascade_len)
                    # CONSIDER: remove probas for all special tokens aside from STOP
                    probas = torch.nn.functional.softmax(model_output, dim=1)
                    # print(probas.size())
                    generated[known_cascade_len][:, known_seq_len] = torch.multinomial(probas, num_samples=1).squeeze()
            return generated
