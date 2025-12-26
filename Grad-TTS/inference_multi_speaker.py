#!/usr/bin/env python3
# Multi-speaker inference script for Hungarian Grad-TTS model

import argparse
import json
import datetime as dt
import numpy as np
from scipy.io.wavfile import write

import torch

import params
from model import GradTTS
from text.symbols import symbols
from utils import intersperse

import sys
sys.path.append('./hifi-gan/')
from env import AttrDict
from models import Generator as HiFiGAN


def get_phoneme_ids(sampa_string):
    """Convert SAMPA phoneme string to list of symbol IDs."""
    phoneme_ids = []
    i = 0
    while i < len(sampa_string):
        # Try multi-character phonemes first (longest match)
        matched = False
        for length in [2, 1]:  # Try 2-char, then 1-char
            if i + length <= len(sampa_string):
                candidate = sampa_string[i:i+length]
                if candidate in symbols:
                    phoneme_ids.append(symbols.index(candidate))
                    i += length
                    matched = True
                    break
        if not matched:
            print(f"Warning: Unknown phoneme '{sampa_string[i]}' at position {i}")
            i += 1  # Skip unknown character
    return phoneme_ids


def load_models(checkpoint_path, vocoder_path, vocoder_config_path, n_spks=26, spk_emb_dim=64):
    """Load Grad-TTS and HiFi-GAN models."""
    print('Initializing Grad-TTS...')
    generator = GradTTS(
        len(symbols) + 1, n_spks, spk_emb_dim,
        params.n_enc_channels, params.filter_channels,
        params.filter_channels_dp, params.n_heads, params.n_enc_layers,
        params.enc_kernel, params.enc_dropout, params.window_size,
        params.n_feats, params.dec_dim, params.beta_min, params.beta_max, params.pe_scale
    )
    generator.load_state_dict(torch.load(checkpoint_path, map_location=lambda loc, storage: loc))
    _ = generator.cuda().eval()
    print(f'Number of parameters: {generator.nparams}')
    
    print('Initializing HiFi-GAN...')
    with open(vocoder_config_path) as f:
        h = AttrDict(json.load(f))
    vocoder = HiFiGAN(h)
    vocoder.load_state_dict(torch.load(vocoder_path, map_location=lambda loc, storage: loc)['generator'])
    _ = vocoder.cuda().eval()
    vocoder.remove_weight_norm()
    
    return generator, vocoder


def synthesize(generator, vocoder, sampa_text, speaker_id, n_timesteps=50, temperature=1.5, length_scale=0.91):
    """
    Synthesize speech from SAMPA phonemes.
    
    Args:
        generator: Grad-TTS model
        vocoder: HiFi-GAN vocoder
        sampa_text: String of SAMPA phonemes (e.g., "hOlnOp")
        speaker_id: Integer speaker ID (0-25 for 26 speakers)
        n_timesteps: Number of diffusion steps (more = better quality, slower)
        temperature: Sampling temperature (higher = more variation)
        length_scale: Speech duration scale (lower = faster speech)
    """
    print(f'Synthesizing: "{sampa_text}" with speaker {speaker_id}')
    
    # Convert SAMPA to phoneme IDs
    x = get_phoneme_ids(sampa_text)
    if not x:
        print("Error: No valid phonemes found!")
        return None
    
    print(f'Phoneme IDs: {x}')
    
    # Add blank tokens between phonemes
    x = intersperse(x, len(symbols))
    x = torch.LongTensor(x)[None].cuda()
    x_lengths = torch.LongTensor([x.shape[-1]]).cuda()
    
    # Speaker ID
    spk = torch.LongTensor([speaker_id]).cuda()
    
    # Generate mel-spectrogram
    t = dt.datetime.now()
    with torch.no_grad():
        y_enc, y_dec, attn = generator.forward(
            x, x_lengths, n_timesteps=n_timesteps, 
            temperature=temperature, stoc=False, spk=spk, 
            length_scale=length_scale
        )
    t = (dt.datetime.now() - t).total_seconds()
    print(f'Grad-TTS RTF: {t * 22050 / (y_dec.shape[-1] * 256):.4f}')
    
    # Generate audio with vocoder
    with torch.no_grad():
        audio = vocoder.forward(y_dec).cpu().squeeze().clamp(-1, 1).numpy()
    
    rtf = t * 22050 / (len(audio))
    print(f'Total RTF: {rtf:.4f}')
    
    return audio, 22050


def main():
    parser = argparse.ArgumentParser(description='Multi-speaker Grad-TTS inference')
    parser.add_argument('--checkpoint', type=str, default='./checkpts/grad_1000.pt',
                        help='Path to Grad-TTS checkpoint')
    parser.add_argument('--vocoder', type=str, default='./checkpts/hifigan.pt',
                        help='Path to HiFi-GAN vocoder checkpoint')
    parser.add_argument('--vocoder-config', type=str, default='./checkpts/hifigan-config.json',
                        help='Path to HiFi-GAN config')
    parser.add_argument('--sampa', type=str, required=True,
                        help='SAMPA phoneme string (e.g., "hOlnOp")')
    parser.add_argument('--speaker', type=int, default=0,
                        help='Speaker ID (0-25, model trained on 26 speakers)')
    parser.add_argument('--output', type=str, default='~/Documents/Thesis/generated_audio/output.wav',
                        help='Output WAV file path')
    parser.add_argument('--output-dir', type=str, default='~/Documents/Thesis/generated_audio',
                        help='Output directory for generated audio files')
    parser.add_argument('--timesteps', type=int, default=10,
                        help='Number of diffusion timesteps (10=fast, 50=quality)')
    parser.add_argument('--temperature', type=float, default=1.5,
                        help='Sampling temperature')
    parser.add_argument('--length-scale', type=float, default=0.91,
                        help='Speech duration scale')
    parser.add_argument('--n-spks', type=int, default=26,
                        help='Number of speakers in the model')
    
    args = parser.parse_args()
    
    # Expand and create output directory
    import os
    args.output = os.path.expanduser(args.output)
    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # If output path is relative, make it relative to output_dir
    if not os.path.isabs(args.output):
        args.output = os.path.join(output_dir, args.output)
    
    # Load models
    generator, vocoder = load_models(
        args.checkpoint, args.vocoder, args.vocoder_config,
        n_spks=args.n_spks, spk_emb_dim=params.spk_emb_dim
    )
    
    # Synthesize
    audio, sr = synthesize(
        generator, vocoder, args.sampa, args.speaker,
        n_timesteps=args.timesteps, temperature=args.temperature,
        length_scale=args.length_scale
    )
    
    if audio is not None:
        # Save audio
        write(args.output, sr, audio)
        print(f'Audio saved to: {args.output}')


if __name__ == '__main__':
    main()
