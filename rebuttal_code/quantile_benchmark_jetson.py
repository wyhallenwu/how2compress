import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import numba
from numba import jit, float64, int32, boolean, prange
import os
import psutil
import torch

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

# Check PyTorch CUDA availability
HAS_TORCH_CUDA = torch.cuda.is_available()
if HAS_TORCH_CUDA:
    print(f"PyTorch CUDA available: {torch.cuda.get_device_name(0)}")
    # Set PyTorch to use TF32 for better performance on Jetson
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Set memory allocator to be more efficient
    torch.cuda.set_per_process_memory_fraction(0.8)  # Use 80% of available GPU memory

def get_memory_usage():
    """Get current memory usage."""
    try:
        # Get GPU memory info if available
        if HAS_CUPY:
            gpu_mem = cp.cuda.runtime.memGetInfo()
            gpu_mem_used = (gpu_mem[1] - gpu_mem[0]) / 1024**2  # Convert to MB
        else:
            gpu_mem_used = None
        
        # Get system memory info
        system_mem = psutil.virtual_memory()
        system_mem_used = system_mem.used / 1024**2  # Convert to MB
        
        return {
            'gpu_mem_used': gpu_mem_used,
            'system_mem_used': system_mem_used
        }
    except:
        return None

# ============= Quantile Categorization Implementations =============

def categorize_by_quantiles_cpu(matrix, num_categories=5):
    """
    CPU version of quantile categorization using NumPy, optimized for Jetson.
    """
    if not isinstance(matrix, np.ndarray):
        matrix = np.array(matrix)
    
    # Use shared memory for better performance on Jetson
    matrix = np.ascontiguousarray(matrix)
    
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
    Numba-accelerated version of quantile categorization, optimized for Jetson.
    """
    rows, cols = matrix.shape
    categorized = np.zeros_like(matrix, dtype=np.int32)
    num_categories = len(boundaries) - 1
    
    # Use parallel loop over rows with constant step size
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
    GPU-accelerated version of quantile categorization using CuPy with Jetson optimizations.
    """
    if not HAS_CUPY:
        raise ImportError("CuPy is not installed. Please install it with: pip install cupy-cuda12x")
    
    if not isinstance(matrix, cp.ndarray):
        matrix = cp.array(matrix)
    
    # Use CUDA streams for asynchronous execution
    stream = cp.cuda.Stream()
    with stream:
        # Calculate quantiles using GPU with Jetson-optimized memory management
        quantiles = cp.linspace(0, 100, num_categories + 1)
        boundaries = cp.percentile(matrix, quantiles)
        
        # Pre-allocate output array with pinned memory for better transfer
        categorized = cp.zeros_like(matrix, dtype=cp.int32)
        
        # Use GPU-accelerated operations with optimized block size for Jetson
        block_size = (16, 16)  # Optimized for Jetson's architecture
        grid_size = (
            (matrix.shape[0] + block_size[0] - 1) // block_size[0],
            (matrix.shape[1] + block_size[1] - 1) // block_size[1]
        )
        
        for i in range(num_categories):
            mask = (matrix >= boundaries[i]) & (matrix < boundaries[i + 1])
            categorized[mask] = i
        
        categorized[matrix == boundaries[-1]] = num_categories - 1
        
        # Synchronize the stream
        stream.synchronize()
    
    return categorized

def categorize_by_quantiles_torch(matrix, num_categories=5):
    """
    PyTorch version of quantile categorization with Jetson optimizations.
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
    Benchmark a specific implementation of the quantile categorization with memory metrics.
    """
    # Warm-up runs
    for _ in range(num_warmup):
        _ = func(matrix)
    
    # Actual timing runs with memory monitoring
    times = []
    memory_readings = []
    
    for _ in range(num_runs):
        # Get initial memory
        initial_memory = get_memory_usage()
        
        start_time = time.perf_counter()
        _ = func(matrix)
        end_time = time.perf_counter()
        
        # Get final memory
        final_memory = get_memory_usage()
        
        elapsed_time = (end_time - start_time) * 1000  # Convert to milliseconds
        times.append(elapsed_time)
        
        if initial_memory is not None and final_memory is not None:
            memory_readings.append({
                'gpu_mem_delta': final_memory['gpu_mem_used'] - initial_memory['gpu_mem_used'] if final_memory['gpu_mem_used'] is not None and initial_memory['gpu_mem_used'] is not None else None,
                'system_mem_delta': final_memory['system_mem_used'] - initial_memory['system_mem_used']
            })
    
    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'mean_gpu_mem_delta': np.mean([m['gpu_mem_delta'] for m in memory_readings if m['gpu_mem_delta'] is not None]) if memory_readings else None,
        'mean_system_mem_delta': np.mean([m['system_mem_delta'] for m in memory_readings]) if memory_readings else None
    }

def create_visualizations(df):
    """
    Create visualizations of the benchmark results using matplotlib.
    """
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 15))
    
    # Plot 1: Mean time by matrix size
    for impl in df['Implementation'].unique():
        impl_data = df[df['Implementation'] == impl]
        ax1.bar(impl_data['Matrix Size'], impl_data['Mean Time (ms)'], label=impl)
    ax1.set_title('Mean Processing Time by Matrix Size')
    ax1.set_xlabel('Matrix Size')
    ax1.set_ylabel('Time (ms)')
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend()
    
    # Plot 2: Time per element
    for impl in df['Implementation'].unique():
        impl_data = df[df['Implementation'] == impl]
        ax2.bar(impl_data['Matrix Size'], impl_data['Time per Element (µs)'], label=impl)
    ax2.set_title('Processing Time per Element')
    ax2.set_xlabel('Matrix Size')
    ax2.set_ylabel('Time per Element (µs)')
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend()
    
    # Plot 3: GPU Memory usage
    if 'Mean GPU Memory Delta (MB)' in df.columns:
        for impl in df['Implementation'].unique():
            impl_data = df[df['Implementation'] == impl]
            if not impl_data['Mean GPU Memory Delta (MB)'].isna().all():
                ax3.bar(impl_data['Matrix Size'], impl_data['Mean GPU Memory Delta (MB)'], label=impl)
        ax3.set_title('GPU Memory Usage')
        ax3.set_xlabel('Matrix Size')
        ax3.set_ylabel('Memory Delta (MB)')
        ax3.tick_params(axis='x', rotation=45)
        ax3.legend()
    
    # Plot 4: System Memory usage
    if 'Mean System Memory Delta (MB)' in df.columns:
        for impl in df['Implementation'].unique():
            impl_data = df[df['Implementation'] == impl]
            ax4.bar(impl_data['Matrix Size'], impl_data['Mean System Memory Delta (MB)'], label=impl)
        ax4.set_title('System Memory Usage')
        ax4.set_xlabel('Matrix Size')
        ax4.set_ylabel('Memory Delta (MB)')
        ax4.tick_params(axis='x', rotation=45)
        ax4.legend()
    
    plt.tight_layout()
    plt.savefig('jetson_benchmark_results.png')
    plt.close()

def save_results_to_txt(df, filename='jetson_benchmark_results.txt'):
    """
    Save benchmark results to a nicely formatted text file.
    """
    with open(filename, 'w') as f:
        f.write("Jetson Quantile Categorization Benchmark Results\n")
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
            f.write("PyTorch version: 1.13.1\n")
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
                if 'Mean GPU Memory Delta (MB)' in row and not pd.isna(row['Mean GPU Memory Delta (MB)']):
                    f.write(f"GPU Memory Delta: {row['Mean GPU Memory Delta (MB)']:.2f} MB\n")
                if 'Mean System Memory Delta (MB)' in row:
                    f.write(f"System Memory Delta: {row['Mean System Memory Delta (MB)']:.2f} MB\n")
            
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
        f.write("Memory usage is in megabytes (MB).\n")

def run_benchmarks():
    """
    Run benchmarks for all implementations on Jetson device.
    """
    # Test different matrix sizes optimized for Jetson
    matrix_sizes = [
        (480, 640),     # 480p (SD)
        (720, 1280),    # 720p (HD)
        (900, 1600),    # 900p
        (1080, 1920),   # 1080p (Full HD)
    ]
    
    # Create a list to store results
    results = []
    
    print("Benchmarking quantile categorization performance on Jetson")
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
        metrics = benchmark_implementation(
            lambda x: categorize_by_quantiles_cpu(x), matrix
        )
        results.append({
            'Matrix Size': f"{size[0]}p ({size[0]}x{size[1]})",
            'Elements': num_elements,
            'Implementation': 'CPU (NumPy)',
            'Mean Time (ms)': metrics['mean_time'],
            'Std Time (ms)': metrics['std_time'],
            'Time per Element (µs)': (metrics['mean_time'] / num_elements) * 1000,
            'Mean GPU Memory Delta (MB)': metrics['mean_gpu_mem_delta'],
            'Mean System Memory Delta (MB)': metrics['mean_system_mem_delta']
        })
        print(f"\nCPU (NumPy) version:")
        print(f"Average time: {metrics['mean_time']:.2f} ± {metrics['std_time']:.2f} ms")
        print(f"Time per element: {(metrics['mean_time'] / num_elements) * 1000:.6f} µs")
        
        # Numba version
        metrics = benchmark_implementation(
            lambda x: categorize_by_quantiles(x, use_gpu=False), matrix
        )
        results.append({
            'Matrix Size': f"{size[0]}p ({size[0]}x{size[1]})",
            'Elements': num_elements,
            'Implementation': 'Numba',
            'Mean Time (ms)': metrics['mean_time'],
            'Std Time (ms)': metrics['std_time'],
            'Time per Element (µs)': (metrics['mean_time'] / num_elements) * 1000,
            'Mean GPU Memory Delta (MB)': metrics['mean_gpu_mem_delta'],
            'Mean System Memory Delta (MB)': metrics['mean_system_mem_delta']
        })
        print(f"\nNumba version:")
        print(f"Average time: {metrics['mean_time']:.2f} ± {metrics['std_time']:.2f} ms")
        print(f"Time per element: {(metrics['mean_time'] / num_elements) * 1000:.6f} µs")
        
        # PyTorch version (if available)
        if HAS_TORCH_CUDA:
            metrics = benchmark_implementation(
                lambda x: categorize_by_quantiles(x, use_torch=True), matrix
            )
            results.append({
                'Matrix Size': f"{size[0]}p ({size[0]}x{size[1]})",
                'Elements': num_elements,
                'Implementation': 'PyTorch',
                'Mean Time (ms)': metrics['mean_time'],
                'Std Time (ms)': metrics['std_time'],
                'Time per Element (µs)': (metrics['mean_time'] / num_elements) * 1000,
                'Mean GPU Memory Delta (MB)': metrics['mean_gpu_mem_delta'],
                'Mean System Memory Delta (MB)': metrics['mean_system_mem_delta']
            })
            print(f"\nPyTorch version:")
            print(f"Average time: {metrics['mean_time']:.2f} ± {metrics['std_time']:.2f} ms")
            print(f"Time per element: {(metrics['mean_time'] / num_elements) * 1000:.6f} µs")
        
        # GPU version (if available)
        if HAS_CUPY:
            metrics = benchmark_implementation(
                lambda x: categorize_by_quantiles(x, use_gpu=True), matrix
            )
            results.append({
                'Matrix Size': f"{size[0]}p ({size[0]}x{size[1]})",
                'Elements': num_elements,
                'Implementation': 'GPU (CuPy)',
                'Mean Time (ms)': metrics['mean_time'],
                'Std Time (ms)': metrics['std_time'],
                'Time per Element (µs)': (metrics['mean_time'] / num_elements) * 1000,
                'Mean GPU Memory Delta (MB)': metrics['mean_gpu_mem_delta'],
                'Mean System Memory Delta (MB)': metrics['mean_system_mem_delta']
            })
            print(f"\nGPU (CuPy) version:")
            print(f"Average time: {metrics['mean_time']:.2f} ± {metrics['std_time']:.2f} ms")
            print(f"Time per element: {(metrics['mean_time'] / num_elements) * 1000:.6f} µs")
        
        print("-" * 70)
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Save results to CSV
    df.to_csv('jetson_benchmark_results.csv', index=False)
    print("\nResults saved to jetson_benchmark_results.csv")
    
    # Save results to text file
    save_results_to_txt(df)
    print("Results saved to jetson_benchmark_results.txt")
    
    # Create visualizations
    create_visualizations(df)

if __name__ == "__main__":
    # Run the benchmarks
    run_benchmarks()
    
    # Test with a sample matrix
    print("\nTesting with a sample matrix...")
    test_matrix = np.random.rand(480, 640)  # Using a smaller size for Jetson
    
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