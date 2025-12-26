#!/usr/bin/env python3
"""
Extract SAMPA examples from Excel file.
Creates a dictionary of text -> SAMPA mappings from your Excel data.
"""

import argparse
from collections import defaultdict
import pandas as pd

def extract_sampa_mappings(excel_file, text_column='text', sampa_column='sampa'):
    """Extract text-to-SAMPA mappings from Excel file."""
    mappings = {}
    
    print(f"Reading Excel file: {excel_file}")
    
    # Try to read Excel file
    try:
        df = pd.read_excel(excel_file)
        print(f"Loaded {len(df)} rows from Excel")
        print(f"Columns: {list(df.columns)}")
    except Exception as e:
        print(f"Error reading Excel: {e}")
        print("Trying to read as CSV...")
        df = pd.read_csv(excel_file)
        print(f"Loaded {len(df)} rows from CSV")
        print(f"Columns: {list(df.columns)}")
    
    # Check if columns exist
    if text_column not in df.columns or sampa_column not in df.columns:
        print(f"\nError: Could not find columns '{text_column}' and/or '{sampa_column}'")
        print(f"Available columns: {list(df.columns)}")
        print("\nPlease specify correct column names with --text-col and --sampa-col")
        return {}
    
    # Extract mappings
    for idx, row in df.iterrows():
        text = str(row[text_column]).strip()
        sampa = str(row[sampa_column]).strip()
        
        # Skip empty or NaN values
        if text == 'nan' or sampa == 'nan' or not text or not sampa:
            continue
        
        # Store mapping (text -> sampa)
        if text not in mappings:
            mappings[text] = sampa
        elif mappings[text] != sampa:
            # Different SAMPA for same text (could be variant pronunciations)
            if idx < 10:  # Only show first 10 warnings
                print(f"Warning: Multiple SAMPA for '{text}'")
                print(f"  1: {mappings[text]}")
                print(f"  2: {sampa}")
    
    return mappings


def save_dictionary(mappings, output_file):
    """Save text-SAMPA dictionary."""
    with open(output_file, 'w', encoding='utf-8') as f:
        for text, sampa in sorted(mappings.items()):
            f.write(f"{text}\t{sampa}\n")
    print(f"\nSaved {len(mappings)} mappings to: {output_file}")


def search_sampa(mappings, query):
    """Search for text containing query string."""
    results = []
    query_lower = query.lower()
    for text, sampa in mappings.items():
        if query_lower in text.lower():
            results.append((text, sampa))
    return results


def main():
    parser = argparse.ArgumentParser(description='Extract SAMPA mappings from Excel file')
    parser.add_argument('--excel', type=str, required=True,
                        help='Path to Excel file (.xlsx or .csv)')
    parser.add_argument('--text-col', type=str, default='text',
                        help='Name of column containing text (default: text)')
    parser.add_argument('--sampa-col', type=str, default='sampa',
                        help='Name of column containing SAMPA (default: sampa)')
    parser.add_argument('--output', type=str, default='hungarian_sampa_dict.txt',
                        help='Output dictionary file')
    parser.add_argument('--search', type=str, help='Search for text containing this string')
    
    args = parser.parse_args()
    
    # Extract mappings
    mappings = extract_sampa_mappings(args.excel, args.text_col, args.sampa_col)
    
    print(f"\nFound {len(mappings)} unique text entries")
    
    # Show some examples
    print("\nSample mappings:")
    for i, (text, sampa) in enumerate(list(mappings.items())[:10]):
        print(f"  {text} -> {sampa}")
    
    # Save dictionary
    save_dictionary(mappings, args.output)
    
    # Search if requested
    if args.search:
        results = search_sampa(mappings, args.search)
        print(f"\nSearch results for '{args.search}': {len(results)} matches")
        for text, sampa in results[:20]:
            print(f"  {text} -> {sampa}")
        if len(results) > 20:
            print(f"  ... and {len(results) - 20} more")
    
    print(f"\n✓ Dictionary saved to: {args.output}")
    print(f"  Use it to look up SAMPA transcriptions for Hungarian text!")


if __name__ == '__main__':
    main()
