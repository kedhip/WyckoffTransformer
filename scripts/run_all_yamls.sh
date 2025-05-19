#!/bin/bash
# ls yamls/models/NextToken/v5/*.yaml | CUDA_VISIBLE_DEVICES=0 xargs -I "{}" python train.py "{}" mp_20_biternary cuda --pilot
# pilot is prone to SMACT timeouts, so no
# ls yamls/models/NextToken/v5/*.yaml | parallel --tmux -j4 'export CUDA_VISIBLE_DEVICES=$(({%}%2)); python train.py {} mp_20_biternary cuda --pilot && python train.py {} mp_20_biternary cuda'
# ls yamls/models/NextToken/v5/*.yaml | parallel --tmux -j4 'export CUDA_VISIBLE_DEVICES=$(({%}%2)); python train.py {} mp_20_biternary cuda'
ls yamls/models/NextToken/v6/*.yaml | parallel --tmux -j4 'export CUDA_VISIBLE_DEVICES=$(({%}%2)); python train.py {} mp_20_biternary cuda'