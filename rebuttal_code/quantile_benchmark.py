import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numba
from numba import jit, float64, int32, boolean, prange

# Try to import CuPy for GPU support
try:
    import cupy as cp
    HAS_CUPY = True
    # Check CUDA version
    CUDA_VERSION = cp.cuda.runtime.runtimeGetVersion()
    if CUDA_VERSION < 12060:  # 12.6.0
        print(f"Warning: Current CUDA version {CUDA_VERSION//1000}.{(CUDA_VERSION%1000)//10} is lower than recommended 12.6")
except ImportError:
    HAS_CUPY = False
    CUDA_VERSION = None
    print("Warning: CuPy not installed. For GPU acceleration, install with: pip install cupy-cuda12x")

# Try to import PyTorch for CUDA support
try:
    import torch
    HAS_TORCH_CUDA = torch.cuda.is_available()
    if HAS_TORCH_CUDA:
        print(f"PyTorch CUDA available: {torch.cuda.get_device_name(0)}")
        # Set PyTorch to use TF32 for better performance
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
except ImportError:
    HAS_TORCH_CUDA = False
    print("Warning: PyTorch not installed. For GPU acceleration, install with: pip install torch torchvision torchaudio")

# ============= Quantile Categorization Implementations =============

def categorize_by_quantiles_cpu(matrix, num_categories=5):
    """
    CPU version of quantile categorization using NumPy.
    """
    if not isinstance(matrix, np.ndarray):
        matrix = np.array(matrix)
    
    quantiles = np.linspace(0, 100, num_categories + 1)
    boundaries = np.percentile(matrix, quantiles)
    
    categorized = np.zeros_like(matrix, dtype=np.int32)
    
    for i in range(num_categories):
        mask = (matrix >= boundaries[i]) & (matrix < boundaries[i + 1])
        categorized[mask] = i
    
    categorized[matrix == boundaries[-1]] = num_categories - 1
    
    return categorized

@jit(nopython=True, parallel=True)
def categorize_by_quantiles_numba(matrix, boundaries):
    """
    Numba-accelerated version of quantile categorization.
    """
    rows, cols = matrix.shape
    categorized = np.zeros_like(matrix, dtype=np.int32)
    num_categories = len(boundaries) - 1
    
    # Use parallel loop over rows
    for i in prange(rows):
        for j in range(cols):
            val = matrix[i, j]
            # Find the appropriate category
            for k in range(num_categories):
                if val >= boundaries[k] and val < boundaries[k + 1]:
                    categorized[i, j] = k
                    break
            # Handle the last boundary case
            if val == boundaries[-1]:
                categorized[i, j] = num_categories - 1
    
    return categorized

def categorize_by_quantiles_gpu(matrix, num_categories=5):
    """
    GPU-accelerated version of quantile categorization using CuPy with CUDA 12.6 optimizations.
    """
    if not HAS_CUPY:
        raise ImportError("CuPy is not installed. Please install it with: pip install cupy-cuda12x")
    
    if not isinstance(matrix, cp.ndarray):
        matrix = cp.array(matrix)
    
    # Use CUDA streams for asynchronous execution
    stream = cp.cuda.Stream()
    with stream:
        # Calculate quantiles using GPU
        quantiles = cp.linspace(0, 100, num_categories + 1)
        boundaries = cp.percentile(matrix, quantiles)
        
        # Pre-allocate output array
        categorized = cp.zeros_like(matrix, dtype=cp.int32)
        
        # Use GPU-accelerated operations
        for i in range(num_categories):
            mask = (matrix >= boundaries[i]) & (matrix < boundaries[i + 1])
            categorized[mask] = i
        
        categorized[matrix == boundaries[-1]] = num_categories - 1
        
        # Synchronize the stream
        stream.synchronize()
    
    return categorized

def categorize_by_quantiles_torch(matrix, num_categories=5):
    """
    PyTorch version of quantile categorization with CUDA support.
    """
    if not isinstance(matrix, torch.Tensor):
        matrix = torch.tensor(matrix, dtype=torch.float32)
    
    # Move to GPU if available
    if HAS_TORCH_CUDA:
        matrix = matrix.cuda()
    
    # Calculate quantiles
    quantiles = torch.linspace(0, 100, num_categories + 1, device=matrix.device)
    boundaries = torch.quantile(matrix, quantiles / 100)
    
    # Initialize output tensor
    categorized = torch.zeros_like(matrix, dtype=torch.int32)
    
    # Use vectorized operations for better performance
    for i in range(num_categories):
        mask = (matrix >= boundaries[i]) & (matrix < boundaries[i + 1])
        categorized[mask] = i
    
    # Handle the last boundary case
    categorized[matrix == boundaries[-1]] = num_categories - 1
    
    # Move back to CPU if needed
    if HAS_TORCH_CUDA:
        categorized = categorized.cpu()
    
    return categorized.numpy()

def categorize_by_quantiles(matrix, num_categories=5, use_gpu=False, use_torch=False):
    """
    Main function that selects the appropriate implementation based on the flags.
    """
    if use_torch:
        if not HAS_TORCH_CUDA:
            print("Warning: PyTorch CUDA not available, falling back to CPU version")
            return categorize_by_quantiles_cpu(matrix, num_categories)
        return categorize_by_quantiles_torch(matrix, num_categories)
    
    if use_gpu:
        if not HAS_CUPY:
            print("Warning: CuPy not available, falling back to CPU version")
            return categorize_by_quantiles_cpu(matrix, num_categories)
        if CUDA_VERSION is not None and CUDA_VERSION < 12060:
            print("Warning: Using GPU with CUDA version lower than 12.6")
        result = categorize_by_quantiles_gpu(matrix, num_categories)
        return cp.asnumpy(result)  # Convert back to numpy array
    
    # For CPU version, use Numba acceleration
    if not isinstance(matrix, np.ndarray):
        matrix = np.array(matrix)
    
    quantiles = np.linspace(0, 100, num_categories + 1)
    boundaries = np.percentile(matrix, quantiles)
    
    return categorize_by_quantiles_numba(matrix, boundaries)

# ============= Benchmarking Functions =============

def benchmark_implementation(func, matrix, num_warmup=5, num_runs=10):
    """
    Benchmark a specific implementation of the quantile categorization.
    """
    # Warm-up runs
    for _ in range(num_warmup):
        _ = func(matrix)
    
    # Actual timing runs
    times = []
    for _ in range(num_runs):
        start_time = time.perf_counter()
        _ = func(matrix)
        end_time = time.perf_counter()
        elapsed_time = (end_time - start_time) * 1000  # Convert to milliseconds
        times.append(elapsed_time)
    
    return np.mean(times), np.std(times)

def create_visualizations(df):
    """
    Create visualizations of the benchmark results.
    """
    # Use a valid matplotlib style
    plt.style.use('seaborn-v0_8')
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    # Plot 1: Mean time by matrix size
    sns.barplot(data=df, x='Matrix Size', y='Mean Time (ms)', hue='Implementation', ax=ax1)
    ax1.set_title('Mean Processing Time by Matrix Size')
    ax1.set_xlabel('Matrix Size')
    ax1.set_ylabel('Time (ms)')
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Time per element
    sns.barplot(data=df, x='Matrix Size', y='Time per Element (µs)', hue='Implementation', ax=ax2)
    ax2.set_title('Processing Time per Element')
    ax2.set_xlabel('Matrix Size')
    ax2.set_ylabel('Time per Element (µs)')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('benchmark_results.png')
    plt.close()

def save_results_to_txt(df, filename='benchmark_results.txt'):
    """
    Save benchmark results to a nicely formatted text file.
    """
    with open(filename, 'w') as f:
        f.write("Quantile Categorization Benchmark Results\n")
        f.write("=" * 50 + "\n\n")
        
        # Write system information
        f.write("System Information:\n")
        f.write("-" * 20 + "\n")
        f.write(f"NumPy version: {np.__version__}\n")
        f.write(f"Numba version: {numba.__version__}\n")
        if HAS_CUPY:
            f.write(f"CuPy version: {cp.__version__}\n")
            f.write(f"CUDA version: {CUDA_VERSION//1000}.{(CUDA_VERSION%1000)//10}\n")
        if HAS_TORCH_CUDA:
            f.write(f"PyTorch version: {torch.__version__}\n")
            f.write(f"PyTorch CUDA device: {torch.cuda.get_device_name(0)}\n")
        f.write("\n")
        
        # Write detailed results for each matrix size
        for size in df['Matrix Size'].unique():
            f.write(f"Matrix Size: {size}\n")
            f.write("-" * 20 + "\n")
            
            size_data = df[df['Matrix Size'] == size]
            for _, row in size_data.iterrows():
                f.write(f"\nImplementation: {row['Implementation']}\n")
                f.write(f"Mean Time: {row['Mean Time (ms)']:.2f} ± {row['Std Time (ms)']:.2f} ms\n")
                f.write(f"Time per Element: {row['Time per Element (µs)']:.6f} µs\n")
            
            f.write("\n" + "=" * 50 + "\n\n")
        
        # Write summary statistics
        f.write("Summary Statistics\n")
        f.write("-" * 20 + "\n")
        
        # Calculate speedup ratios
        pivot_df = df.pivot(index='Matrix Size', columns='Implementation', values='Mean Time (ms)')
        speedup_df = pivot_df.div(pivot_df.min(axis=1), axis=0)
        
        f.write("\nSpeedup Ratios (lower is better):\n")
        for size in speedup_df.index:
            f.write(f"\nMatrix Size: {size}\n")
            for impl in speedup_df.columns:
                f.write(f"{impl}: {speedup_df.loc[size, impl]:.2f}x\n")
        
        f.write("\n" + "=" * 50 + "\n")
        f.write("\nNote: All times are in milliseconds (ms) unless specified otherwise.\n")
        f.write("Time per element is in microseconds (µs).\n")

def run_benchmarks():
    """
    Run benchmarks for all implementations and create visualizations.
    """
    # Test different matrix sizes based on common video resolutions
    matrix_sizes = [
        (480, 640),     # 480p (SD)
        (720, 1280),    # 720p (HD)
        (900, 1600),    # 900p
        (1080, 1920),   # 1080p (Full HD)
        (1440, 2560),   # 1440p (2K)
        (2160, 3840),   # 2160p (4K)
    ]
    
    # Create a list to store results
    results = []
    
    print("Benchmarking quantile categorization performance")
    print("=" * 70)
    print("Testing with common video resolutions:")
    for size in matrix_sizes:
        print(f"- {size[0]}p ({size[0]}x{size[1]})")
    print("=" * 70)
    
    for size in matrix_sizes:
        print(f"\nMatrix size: {size[0]}p ({size[0]}x{size[1]})")
        matrix = np.random.rand(*size)
        num_elements = size[0] * size[1]
        
        # CPU (NumPy) version
        mean_time, std_time = benchmark_implementation(
            lambda x: categorize_by_quantiles_cpu(x), matrix
        )
        results.append({
            'Matrix Size': f"{size[0]}p ({size[0]}x{size[1]})",
            'Elements': num_elements,
            'Implementation': 'CPU (NumPy)',
            'Mean Time (ms)': mean_time,
            'Std Time (ms)': std_time,
            'Time per Element (µs)': (mean_time / num_elements) * 1000
        })
        print(f"\nCPU (NumPy) version:")
        print(f"Average time: {mean_time:.2f} ± {std_time:.2f} ms")
        print(f"Time per element: {(mean_time / num_elements) * 1000:.6f} µs")
        
        # Numba version
        mean_time, std_time = benchmark_implementation(
            lambda x: categorize_by_quantiles(x, use_gpu=False), matrix
        )
        results.append({
            'Matrix Size': f"{size[0]}p ({size[0]}x{size[1]})",
            'Elements': num_elements,
            'Implementation': 'Numba',
            'Mean Time (ms)': mean_time,
            'Std Time (ms)': std_time,
            'Time per Element (µs)': (mean_time / num_elements) * 1000
        })
        print(f"\nNumba version:")
        print(f"Average time: {mean_time:.2f} ± {std_time:.2f} ms")
        print(f"Time per element: {(mean_time / num_elements) * 1000:.6f} µs")
        
        # PyTorch version (if available)
        if HAS_TORCH_CUDA:
            mean_time, std_time = benchmark_implementation(
                lambda x: categorize_by_quantiles(x, use_torch=True), matrix
            )
            results.append({
                'Matrix Size': f"{size[0]}p ({size[0]}x{size[1]})",
                'Elements': num_elements,
                'Implementation': 'PyTorch',
                'Mean Time (ms)': mean_time,
                'Std Time (ms)': std_time,
                'Time per Element (µs)': (mean_time / num_elements) * 1000
            })
            print(f"\nPyTorch version:")
            print(f"Average time: {mean_time:.2f} ± {std_time:.2f} ms")
            print(f"Time per element: {(mean_time / num_elements) * 1000:.6f} µs")
        
        # GPU version (if available)
        if HAS_CUPY:
            mean_time, std_time = benchmark_implementation(
                lambda x: categorize_by_quantiles(x, use_gpu=True), matrix
            )
            results.append({
                'Matrix Size': f"{size[0]}p ({size[0]}x{size[1]})",
                'Elements': num_elements,
                'Implementation': 'GPU (CuPy)',
                'Mean Time (ms)': mean_time,
                'Std Time (ms)': std_time,
                'Time per Element (µs)': (mean_time / num_elements) * 1000
            })
            print(f"\nGPU (CuPy) version:")
            print(f"Average time: {mean_time:.2f} ± {std_time:.2f} ms")
            print(f"Time per element: {(mean_time / num_elements) * 1000:.6f} µs")
        
        print("-" * 70)
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Save results to CSV
    df.to_csv('benchmark_results.csv', index=False)
    print("\nResults saved to benchmark_results.csv")
    
    # Save results to text file
    save_results_to_txt(df)
    print("Results saved to benchmark_results.txt")
    
    # Create visualizations
    create_visualizations(df)

if __name__ == "__main__":
    # Run the benchmarks
    run_benchmarks()
    
    # Test with a sample matrix
    print("\nTesting with a sample matrix...")
    test_matrix = np.random.rand(1000, 1000)
    
    # Test CPU version
    print("\nTesting CPU version...")
    categorized_cpu = categorize_by_quantiles(test_matrix, use_gpu=False)
    
    # Test PyTorch version if available
    if HAS_TORCH_CUDA:
        print("\nTesting PyTorch version...")
        categorized_torch = categorize_by_quantiles(test_matrix, use_torch=True)
        print("\nVerifying PyTorch results match CPU:", np.array_equal(categorized_cpu, categorized_torch))
    
    # Test GPU version if available
    if HAS_CUPY:
        print("\nTesting GPU version...")
        categorized_gpu = categorize_by_quantiles(test_matrix, use_gpu=True)
        print("\nVerifying GPU results match CPU:", np.array_equal(categorized_cpu, categorized_gpu))
    
    # Print some statistics
    print("\nNumber of values in each category:")
    for i in range(5):
        count = np.sum(categorized_cpu == i)
        print(f"Category {i}: {count} values") 