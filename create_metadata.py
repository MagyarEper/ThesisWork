#!/usr/bin/env python3
"""
Create metadata files for Grad-TTS training from Excel/CSV source.

This script:
1. Reads your original dataset metadata (Excel/CSV)
2. Generates train/val/test splits
3. Creates clean metadata files in the correct format
4. Verifies all audio files exist

Output format: filepath|text|speaker|sampa_phonemes
"""

import pandas as pd
import numpy as np
import os
import re
from pathlib import Path

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Input file (your original metadata)
INPUT_FILE = '/home/berta/data/HungarianDysartriaDatabase/SAMPA_transcripts_all.xlsx'  # or .csv

# Audio files location
AUDIO_ROOT = '/home/berta/data/HungarianDysartriaDatabase/wav'

# Output directory for metadata files
OUTPUT_DIR = './Grad-TTS/resources/files'

# Split ratios
TRAIN_RATIO = 0.75
VAL_RATIO = 0.15
TEST_RATIO = 0.10  # Remaining after train and val

# Random seed for reproducibility
RANDOM_SEED = 42

# ==============================================================================
# FUNCTIONS
# ==============================================================================

def clean_text(text):
    """Remove brackets and their contents (false starts) from text."""
    if pd.isna(text):
        return ""
    # Remove [word] patterns
    text = re.sub(r'\[.*?\]', '', str(text))
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_sampa(sampa):
    """Clean SAMPA phoneme string."""
    if pd.isna(sampa):
        return ""
    # Remove spaces
    sampa = str(sampa).replace(' ', '')
    return sampa

def load_metadata(input_file):
    """Load metadata from Excel or CSV file."""
    print(f"Loading metadata from: {input_file}")
    
    file_ext = Path(input_file).suffix.lower()
    
    if file_ext == '.xlsx' or file_ext == '.xls':
        df = pd.read_excel(input_file, header=0)  # Use first row as header
    elif file_ext == '.csv':
        df = pd.read_csv(input_file, header=0)  # Use first row as header
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")
    
    # The Excel has columns with headers, rename them to standard names
    # Expected columns: ID, Utterance ID, Code, Full_ID, Full_ID.wav, Utterance structure, Transcript, Sampa
    column_mapping = {
        'ID': 'speaker',
        'Sampa': 'sampa',
        'Transcript': 'text',
        'Full_ID.wav': 'full_id1'  # This column has the actual filename
    }
    df = df.rename(columns=column_mapping)
    
    # Verify we have the required columns
    required_cols = ['speaker', 'text', 'sampa', 'full_id1']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        print(f"Available columns: {df.columns.tolist()}")
        raise ValueError("Please verify your input file structure")
    
    print(f"Loaded {len(df)} records")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    return df

def prepare_dataframe(df, audio_root):
    """Prepare dataframe with cleaned text, SAMPA, and audio paths."""
    print("\nPreparing data...")
    
    # Clean text (remove brackets)
    df['text_clean'] = df['text'].apply(clean_text)
    
    # Clean SAMPA (remove spaces)
    df['sampa_clean'] = df['sampa'].apply(clean_sampa)
    
    # Create full audio file paths
    df['wav_path'] = df['full_id1'].apply(lambda x: os.path.join(audio_root, f"{x}.wav"))
    
    # Verify files exist
    df['file_exists'] = df['wav_path'].apply(os.path.exists)
    
    # Report missing files
    missing = df[~df['file_exists']]
    if len(missing) > 0:
        print(f"\nWarning: {len(missing)} audio files not found!")
        print("First 10 missing files:")
        for idx, row in missing.head(10).iterrows():
            print(f"  {row['wav_path']}")
    
    # Keep only rows with existing files
    df_valid = df[df['file_exists']].copy()
    print(f"\nValid samples: {len(df_valid)}/{len(df)}")
    
    # Remove rows with empty text or SAMPA
    df_valid = df_valid[
        (df_valid['text_clean'].str.len() > 0) & 
        (df_valid['sampa_clean'].str.len() > 0)
    ]
    print(f"After removing empty text/SAMPA: {len(df_valid)}")
    
    return df_valid

def create_splits(df, train_ratio, val_ratio, random_seed):
    """Create train/val/test splits stratified by speaker."""
    print("\nCreating splits...")
    
    np.random.seed(random_seed)
    
    train_dfs = []
    val_dfs = []
    test_dfs = []
    
    # Split by speaker to maintain speaker distribution
    for speaker in df['speaker'].unique():
        speaker_df = df[df['speaker'] == speaker].copy()
        speaker_df = speaker_df.sample(frac=1, random_state=random_seed)  # Shuffle
        
        n = len(speaker_df)
        train_size = int(train_ratio * n)
        val_size = int(val_ratio * n)
        # test gets remaining
        
        train_dfs.append(speaker_df.iloc[:train_size])
        val_dfs.append(speaker_df.iloc[train_size:train_size+val_size])
        test_dfs.append(speaker_df.iloc[train_size+val_size:])
    
    train_df = pd.concat(train_dfs, ignore_index=True)
    val_df = pd.concat(val_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)
    
    print(f"\nSplit results:")
    print(f"  Train: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Val:   {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test:  {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")
    
    # Speaker distribution check
    print(f"\nSpeakers per split:")
    print(f"  Train: {train_df['speaker'].nunique()} speakers")
    print(f"  Val:   {val_df['speaker'].nunique()} speakers")
    print(f"  Test:  {test_df['speaker'].nunique()} speakers")
    
    return train_df, val_df, test_df

def save_metadata(df, output_path):
    """Save metadata in Grad-TTS format: path|text|speaker|sampa"""
    print(f"\nSaving to: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            # Format: wav_path|text|speaker|sampa
            line = f"{row['wav_path']}|{row['text_clean']}|{row['speaker']}|{row['sampa_clean']}\n"
            f.write(line)
    
    print(f"  Saved {len(df)} samples")

def main():
    """Main execution function."""
    print("="*80)
    print("Grad-TTS Metadata Generator for Hungarian Dysarthric Speech")
    print("="*80)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load data
    df = load_metadata(INPUT_FILE)
    
    # Prepare data
    df_valid = prepare_dataframe(df, AUDIO_ROOT)
    
    if len(df_valid) == 0:
        print("\nError: No valid samples found!")
        return
    
    # Create splits
    train_df, val_df, test_df = create_splits(
        df_valid, 
        TRAIN_RATIO, 
        VAL_RATIO, 
        RANDOM_SEED
    )
    
    # Save metadata files
    save_metadata(train_df, os.path.join(OUTPUT_DIR, 'metadata_train.txt'))
    save_metadata(val_df, os.path.join(OUTPUT_DIR, 'metadata_val.txt'))
    save_metadata(test_df, os.path.join(OUTPUT_DIR, 'metadata_test.txt'))
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total samples processed: {len(df_valid)}")
    print(f"Train: {len(train_df)}")
    print(f"Val:   {len(val_df)}")
    print(f"Test:  {len(test_df)}")
    print(f"\nOutput files:")
    print(f"  - {OUTPUT_DIR}/metadata_train.txt")
    print(f"  - {OUTPUT_DIR}/metadata_val.txt")
    print(f"  - {OUTPUT_DIR}/metadata_test.txt")
    print("="*80)
    print("\nNext steps:")
    print("1. Verify the metadata files look correct")
    print("2. Update DATASET_ROOT in params.py if needed")
    print("3. Run: python train.py")
    print("="*80)

if __name__ == '__main__':
    main()
