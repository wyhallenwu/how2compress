import numpy as np
import pandas as pd
from pathlib import Path

def analyze_results(file_path: str):
    """Analyze the results file and compute average statistics.
    
    Args:
        file_path (str): Path to the results file
    """
    # Read the results file
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            # Parse the line: map50_95,map75,map50,map50_95_gt,map75_gt,map50_gt,frame_size,time
            values = line.strip().split(',')
            if len(values) == 8:  # Ensure we have all expected values
                data.append({
                    'map50_95': float(values[0]),
                    'map75': float(values[1]),
                    'map50': float(values[2]),
                    'map50_95_gt': float(values[3]),
                    'map75_gt': float(values[4]),
                    'map50_gt': float(values[5]),
                    'frame_size': int(values[6]),
                    'time': float(values[7])
                })

    # Convert to pandas DataFrame for easier analysis
    df = pd.DataFrame(data)
    
    # Compute statistics
    stats = {
        'Metrics': ['mAP@0.5:0.95', 'mAP@0.75', 'mAP@0.5', 
                   'mAP@0.5:0.95 (GT)', 'mAP@0.75 (GT)', 'mAP@0.5 (GT)',
                   'Frame Size (bytes)', 'Processing Time (s)'],
        'Mean': [
            df['map50_95'].mean(),
            df['map75'].mean(),
            df['map50'].mean(),
            df['map50_95_gt'].mean(),
            df['map75_gt'].mean(),
            df['map50_gt'].mean(),
            df['frame_size'].mean(),
            df['time'].mean()
        ],
        'Std': [
            df['map50_95'].std(),
            df['map75'].std(),
            df['map50'].std(),
            df['map50_95_gt'].std(),
            df['map75_gt'].std(),
            df['map50_gt'].std(),
            df['frame_size'].std(),
            df['time'].std()
        ],
        'Min': [
            df['map50_95'].min(),
            df['map75'].min(),
            df['map50'].min(),
            df['map50_95_gt'].min(),
            df['map75_gt'].min(),
            df['map50_gt'].min(),
            df['frame_size'].min(),
            df['time'].min()
        ],
        'Max': [
            df['map50_95'].max(),
            df['map75'].max(),
            df['map50'].max(),
            df['map50_95_gt'].max(),
            df['map75_gt'].max(),
            df['map50_gt'].max(),
            df['frame_size'].max(),
            df['time'].max()
        ]
    }
    
    # Convert to DataFrame for nice printing
    stats_df = pd.DataFrame(stats)
    
    # Print results
    print("\nResults Analysis:")
    print("=" * 80)
    print(stats_df.to_string(index=False))
    print("=" * 80)
    
    # Print compression ratio (if original frame size is known)
    # Assuming original frame size is 1920x1080x3 bytes
    original_size = 1920 * 1080 * 3
    avg_compression_ratio = original_size / df['frame_size'].mean()
    print(f"\nAverage Compression Ratio: {avg_compression_ratio:.2f}x")
    
    # Print average processing time per frame
    print(f"Average Processing Time per Frame: {df['time'].mean():.3f} seconds")
    
    # Save results to CSV
    output_file = Path(file_path).with_suffix('.analysis.csv')
    stats_df.to_csv(output_file, index=False)
    print(f"\nDetailed analysis saved to: {output_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Analyze results file')
    parser.add_argument('file_path', type=str, help='Path to the results file')
    args = parser.parse_args()
    
    analyze_results(args.file_path) 