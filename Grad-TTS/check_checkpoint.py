#!/usr/bin/env python3
"""Check checkpoint for number of speakers and model architecture."""

import torch

checkpoint_path = 'checkpts/grad_1000.pt'

print(f"Loading checkpoint: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location='cpu')

print("\n" + "="*60)
print("CHECKPOINT INSPECTION")
print("="*60)

# Check if it's a dict or direct state dict
if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
    print(f"\nCheckpoint type: Training checkpoint with metadata")
    if 'epoch' in checkpoint:
        print(f"Epoch: {checkpoint['epoch']}")
else:
    state_dict = checkpoint
    print(f"\nCheckpoint type: Direct state dict")

print(f"\nTotal parameters: {len(state_dict)}")

# Look for speaker embedding
print("\n" + "="*60)
print("SPEAKER-RELATED PARAMETERS")
print("="*60)

speaker_params = {}
for key, value in state_dict.items():
    if 'spk' in key.lower() or 'speaker' in key.lower():
        speaker_params[key] = value.shape
        print(f"{key}: {value.shape}")

if speaker_params:
    # Look for encoder.spk_emb which tells us n_spks
    if 'encoder.spk_emb.weight' in speaker_params:
        n_spks = speaker_params['encoder.spk_emb.weight'][0]
        print(f"\n✓ Found encoder speaker embedding: {n_spks} speakers")
    
    # Look for decoder spk_mlp
    decoder_spk_keys = [k for k in speaker_params.keys() if 'decoder' in k]
    if decoder_spk_keys:
        print(f"✓ Found decoder speaker conditioning layers: {len(decoder_spk_keys)}")
else:
    print("WARNING: No speaker-related parameters found!")
    print("This might be a single-speaker model!")

# Sample other parameters
print("\n" + "="*60)
print("SAMPLE OF OTHER PARAMETERS")
print("="*60)
for i, (key, value) in enumerate(list(state_dict.items())[:10]):
    print(f"{key}: {value.shape}")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
if 'encoder.spk_emb.weight' in state_dict:
    n_spks = state_dict['encoder.spk_emb.weight'].shape[0]
    emb_dim = state_dict['encoder.spk_emb.weight'].shape[1]
    print(f"✓ Multi-speaker model")
    print(f"  - Number of speakers (n_spks): {n_spks}")
    print(f"  - Speaker embedding dimension: {emb_dim}")
else:
    print("✗ Single-speaker model or no speaker embeddings found")
