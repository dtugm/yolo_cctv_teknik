#!/usr/bin/env python3
"""
CUDA Device Checker Script
This script checks for available CUDA devices and provides detailed information.
"""

import torch
import subprocess
import sys
from pathlib import Path

def check_cuda_installation():
    """Check if CUDA is properly installed on the system."""
    print("🔍 Checking CUDA Installation...")
    print("=" * 50)
    
    try:
        # Check nvidia-smi
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ NVIDIA Driver found:")
            print(result.stdout)
        else:
            print("❌ nvidia-smi not found or failed")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ nvidia-smi not found - NVIDIA drivers may not be installed")
        return False
    
    return True

def check_pytorch_cuda():
    """Check PyTorch CUDA support."""
    print("\n🐍 Checking PyTorch CUDA Support...")
    print("=" * 50)
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"Number of CUDA devices: {torch.cuda.device_count()}")
        
        # List all available devices
        print("\n📱 Available CUDA Devices:")
        for i in range(torch.cuda.device_count()):
            device_props = torch.cuda.get_device_properties(i)
            print(f"  Device {i}: {device_props.name}")
            print(f"    - Memory: {device_props.total_memory / (1024**3):.1f} GB")
            print(f"    - Compute Capability: {device_props.major}.{device_props.minor}")
            print(f"    - Multiprocessors: {device_props.multi_processor_count}")
            
        # Current device
        current_device = torch.cuda.current_device()
        print(f"\n🎯 Current CUDA device: {current_device}")
        
        # Memory info
        print("\n💾 Memory Information:")
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_device(i)
            allocated = torch.cuda.memory_allocated(i) / (1024**3)
            cached = torch.cuda.memory_reserved(i) / (1024**3)
            total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"  Device {i}:")
            print(f"    - Allocated: {allocated:.2f} GB")
            print(f"    - Cached: {cached:.2f} GB")
            print(f"    - Total: {total:.2f} GB")
            print(f"    - Free: {total - cached:.2f} GB")
    else:
        print("❌ CUDA not available in PyTorch")
        print("   This could mean:")
        print("   - PyTorch was installed without CUDA support")
        print("   - CUDA drivers are not installed")
        print("   - CUDA version mismatch")

def test_cuda_performance():
    """Test basic CUDA operations."""
    print("\n⚡ Testing CUDA Performance...")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available, skipping performance test")
        return
    
    try:
        # Create tensors on CPU and GPU
        device = torch.device('cuda:0')
        size = (1000, 1000)
        
        # CPU test
        import time
        cpu_tensor = torch.randn(size)
        start_time = time.time()
        cpu_result = torch.matmul(cpu_tensor, cpu_tensor)
        cpu_time = time.time() - start_time
        
        # GPU test
        gpu_tensor = torch.randn(size, device=device)
        torch.cuda.synchronize()  # Wait for GPU operations to complete
        start_time = time.time()
        gpu_result = torch.matmul(gpu_tensor, gpu_tensor)
        torch.cuda.synchronize()
        gpu_time = time.time() - start_time
        
        print(f"Matrix multiplication ({size[0]}x{size[1]}):")
        print(f"  CPU time: {cpu_time:.4f} seconds")
        print(f"  GPU time: {gpu_time:.4f} seconds")
        print(f"  Speedup: {cpu_time/gpu_time:.2f}x")
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")

def check_yolo_cuda_compatibility():
    """Check if YOLO can use CUDA."""
    print("\n🎯 Checking YOLO CUDA Compatibility...")
    print("=" * 50)
    
    try:
        # Import ultralytics
        from ultralytics import YOLO
        
        # Check if model can be loaded on CUDA
        if torch.cuda.is_available():
            try:
                # Try to load a small model on CUDA
                model = YOLO('yolov8n.pt')
                device = torch.device('cuda:0')
                
                # Test inference on CUDA
                import numpy as np
                dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
                results = model(dummy_image, device=device)
                
                print("✅ YOLO CUDA inference successful")
                print(f"   Model device: {next(model.model.parameters()).device}")
                
            except Exception as e:
                print(f"❌ YOLO CUDA inference failed: {e}")
        else:
            print("❌ CUDA not available for YOLO")
            
    except ImportError:
        print("❌ ultralytics not installed")
    except Exception as e:
        print(f"❌ YOLO compatibility check failed: {e}")

def main():
    """Main function to run all checks."""
    print("🚀 CUDA Device Checker")
    print("=" * 50)
    
    # Check system CUDA installation
    cuda_installed = check_cuda_installation()
    
    # Check PyTorch CUDA support
    check_pytorch_cuda()
    
    # Test CUDA performance
    test_cuda_performance()
    
    # Check YOLO compatibility
    check_yolo_cuda_compatibility()
    
    print("\n📋 Summary:")
    print("=" * 50)
    print(f"System CUDA: {'✅ Available' if cuda_installed else '❌ Not Available'}")
    print(f"PyTorch CUDA: {'✅ Available' if torch.cuda.is_available() else '❌ Not Available'}")
    
    if torch.cuda.is_available():
        print(f"CUDA Devices: {torch.cuda.device_count()}")
        print(f"Current Device: {torch.cuda.current_device()}")
        print(f"Device Name: {torch.cuda.get_device_name()}")
    
    print("\n💡 Tips:")
    print("- Use 'python predict_simple.py --device cuda' to use GPU")
    print("- Use 'python predict_simple.py --device cpu' to force CPU")
    print("- Use 'python predict_simple.py --device cuda:0' for specific GPU")

if __name__ == "__main__":
    main()
