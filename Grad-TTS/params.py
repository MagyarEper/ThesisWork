# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import os
from model.utils import fix_len_compatibility



# data parameters
train_filelist_path = "resources/files/metadata_train.txt"
valid_filelist_path = "resources/files/metadata_val.txt"
test_filelist_path = "resources/files/metadata_test.txt"
cmudict_path = None
add_blank = True
n_feats = 80
n_spks = 26  # Number of speakers in Hungarian dysarthric dataset
spk_emb_dim = 64
n_feats = 80
n_fft = 1024
sample_rate = 22050
hop_length = 256
win_length = 1024
f_min = 0
f_max = 8000

# encoder parameters
n_enc_channels = 192
filter_channels = 768
filter_channels_dp = 256
n_enc_layers = 6
enc_kernel = 3
enc_dropout = 0.1
n_heads = 2
window_size = 4

# decoder parameters
dec_dim = 64
beta_min = 0.05
beta_max = 20.0
pe_scale = 1000  # 1 for `grad-tts-old.pt` checkpoint

# training parameters
log_dir = 'logs/hungarian_dysarthric'
test_size = 4
n_epochs = 1000  # Reduced from 10000 - typically 500-1000 epochs is sufficient
batch_size = 32  # Increased from 16 - utilize 16GB VRAM better
learning_rate = 1e-4
seed = 37
save_every = 50  # Save checkpoints every 50 epochs instead of every epoch
out_size = fix_len_compatibility(2*22050//256)
