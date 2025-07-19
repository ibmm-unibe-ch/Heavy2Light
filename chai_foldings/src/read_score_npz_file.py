"""
NPZ File Reader and Display Script
Reads and displays the contents of NumPy .npz files in a structured format.
"""

import numpy as np
import sys
import os
from pathlib import Path

def format_array_info(arr):
    """Format array information for display."""
    info = {
        'shape': arr.shape,
        'dtype': str(arr.dtype),
        'size': arr.size,
        'ndim': arr.ndim
    }
    
    # Add statistics for numeric arrays
    if np.issubdtype(arr.dtype, np.number):
        if arr.size > 0:
            info.update({
                'min': np.min(arr),
                'max': np.max(arr),
                'mean': np.mean(arr),
                'std': np.std(arr)
            })
    
    return info

def display_array_values(arr, key, max_display=20):
    """Display array values with intelligent formatting."""
    print(f"\n  Values for '{key}':")
    
    if arr.size == 0:
        print("    <empty array>")
        return
    
    # For scalar values
    if arr.ndim == 0:
        print(f"    {arr.item()}")
        return
    
    # For 1D arrays
    if arr.ndim == 1:
        if arr.size <= max_display:
            print(f"    {arr}")
        else:
            print(f"    First 10: {arr[:10]}")
            print(f"    Last 10:  {arr[-10:]}")
        return
    
    # For 2D arrays
    if arr.ndim == 2:
        if arr.size <= max_display:
            print(f"    {arr}")
        else:
            print(f"    Shape: {arr.shape}")
            print(f"    First few rows:")
            rows_to_show = min(5, arr.shape[0])
            for i in range(rows_to_show):
                row = arr[i]
                if len(row) > 10:
                    print(f"      Row {i}: [{row[0]:.4f}, {row[1]:.4f}, ..., {row[-1]:.4f}]")
                else:
                    print(f"      Row {i}: {row}")
        return
    
    # For higher dimensional arrays
    print(f"    Shape: {arr.shape}")
    print(f"    Sample values: {arr.flat[:min(10, arr.size)]}")

def read_npz_file(filepath, show_values=True, max_display=20):
    """
    Read and display contents of an NPZ file.
    
    Args:
        filepath (str): Path to the .npz file
        show_values (bool): Whether to display array values
        max_display (int): Maximum number of elements to display for arrays
    """
    
    # Check if file exists
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found.")
        return None
    
    print(f"Reading NPZ file: {filepath}")
    print("=" * 60)
    
    try:
        # Load the NPZ file
        data = np.load(filepath)
        
        print(f"File contains {len(data.keys())} arrays:")
        print(f"Keys: {list(data.keys())}")
        print("\n" + "=" * 60)
        
        # Process each array
        for i, key in enumerate(data.keys(), 1):
            arr = data[key]
            info = format_array_info(arr)
            
            print(f"\n{i}. Array: '{key}'")
            print("-" * 40)
            print(f"  Shape: {info['shape']}")
            print(f"  Data type: {info['dtype']}")
            print(f"  Size: {info['size']} elements")
            print(f"  Dimensions: {info['ndim']}")
            
            # Show statistics for numeric arrays
            if 'min' in info:
                print(f"  Min: {info['min']:.6f}")
                print(f"  Max: {info['max']:.6f}")
                print(f"  Mean: {info['mean']:.6f}")
                print(f"  Std: {info['std']:.6f}")
            
            # Display values if requested
            if show_values:
                display_array_values(arr, key, max_display)
        
        # Close the file
        data.close()
        print("\n" + "=" * 60)
        print("File reading completed successfully.")
        
        return True
        
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def main():
    """Main function to handle command line arguments."""
    
    default_file = "/storage/homefs/lb24i892/chai_folding/outputs/chai_folding_test_with_msa/scores.model_idx_0.npz"
    
    # Check command line arguments
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = default_file
        print(f"No file specified, using default: {filepath}")
    
    # Options for display
    show_values = True
    max_display = 20
    
    # Check for additional arguments
    if len(sys.argv) > 2:
        if '--no-values' in sys.argv:
            show_values = False
        if '--summary' in sys.argv:
            show_values = False
    
    # Read and display the file
    success = read_npz_file(filepath, show_values, max_display)
    
    if success:
        print(f"\nTo use this script:")
        print(f"  python {sys.argv[0]} <path_to_npz_file>")
        print(f"  python {sys.argv[0]} <path_to_npz_file> --no-values  # Summary only")
        print(f"  python {sys.argv[0]} <path_to_npz_file> --summary     # Summary only")

if __name__ == "__main__":
    main()