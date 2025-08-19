#!/usr/bin/env python3
"""
FPS Measurement Script for DeLA S3DIS Model
==========================================

This script measures the inference FPS (Frames Per Second) of DeLA models
on the S3DIS dataset for semantic segmentation.

Usage:
    python measure_fps.py --model_path S3DIS/pretrained/best.pt
    python measure_fps.py --batch_size 16 --num_samples 200
    python measure_fps.py --warmup 10 --runs 5
"""

import argparse
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent))

def setup_s3dis(model_path, batch_size=1):
    """Setup S3DIS model and dataloader"""
    sys.path.append(str(Path(__file__).absolute().parent / "S3DIS"))
    from S3DIS.delasemseg import DelaSemSeg
    from S3DIS.s3dis import S3DIS, s3dis_collate_fn
    from S3DIS.config import s3dis_args, dela_args
    import utils.util as util
    
    # Create model
    model = DelaSemSeg(dela_args).cuda()
    if model_path and Path(model_path).exists():
        util.load_state(model_path, model=model)
        print(f"Loaded model from {model_path}")
    else:
        print("Using randomly initialized model")
    
    # Create dataloader
    dataset = S3DIS(s3dis_args, partition="5", loop=30, train=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, 
                           collate_fn=s3dis_collate_fn, 
                           pin_memory=True, num_workers=4)
    
    return model, dataloader

def measure_fps_s3dis(model, dataloader, num_samples=100, warmup=10, use_amp=True):
    """Measure FPS for S3DIS semantic segmentation model"""
    model.eval()
    torch.cuda.synchronize()
    
    print(f"Warming up with {warmup} samples...")
    sample_count = 0
    batch_size = dataloader.batch_size
    
    # Warmup phase
    with torch.no_grad():
        for batch_data in dataloader:
            if sample_count >= warmup:
                break
                
            xyz, feature, indices, pts, y = batch_data
            xyz = xyz.cuda(non_blocking=True)
            feature = feature.cuda(non_blocking=True)
            indices = [ii.cuda(non_blocking=True).long() for ii in indices[::-1]]
            pts = pts.tolist()[::-1] if isinstance(pts, torch.Tensor) else pts[::-1]
            
            if use_amp:
                with autocast():
                    p = model(xyz, feature, indices, pts)
            else:
                p = model(xyz, feature, indices, pts)
            
            sample_count += batch_size
    
    print(f"Starting FPS measurement with {num_samples} samples...")
    torch.cuda.synchronize()
    
    # Actual timing
    sample_count = 0
    start_time = time.time()
    
    with torch.no_grad():
        for batch_data in dataloader:
            if sample_count >= num_samples:
                break
                
            xyz, feature, indices, pts, y = batch_data
            xyz = xyz.cuda(non_blocking=True)
            feature = feature.cuda(non_blocking=True)
            indices = [ii.cuda(non_blocking=True).long() for ii in indices[::-1]]
            pts = pts.tolist()[::-1] if isinstance(pts, torch.Tensor) else pts[::-1]
            
            if use_amp:
                with autocast():
                    p = model(xyz, feature, indices, pts)
            else:
                p = model(xyz, feature, indices, pts)
            
            sample_count += batch_size
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    total_time = end_time - start_time
    fps = sample_count / total_time
    
    return fps, total_time, sample_count

def get_model_info(model):
    """Get model parameter count and memory usage"""
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    model_size = (param_size + buffer_size) / (1024 * 1024)  # MB
    
    return param_count, model_size

def main():
    parser = argparse.ArgumentParser(description='Measure FPS of DeLA S3DIS model')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to pretrained model (optional)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for inference')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of samples to process for FPS measurement')
    parser.add_argument('--warmup', type=int, default=10,
                        help='Number of warmup samples')
    parser.add_argument('--no_amp', action='store_true',
                        help='Disable automatic mixed precision')
    parser.add_argument('--runs', type=int, default=3,
                        help='Number of runs to average over')
    
    args = parser.parse_args()
    
    torch.set_float32_matmul_precision("high")
    
    print(f"=== DeLA S3DIS Model FPS Measurement ===")
    print(f"Model path: {args.model_path or 'Random initialization'}")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Warmup samples: {args.warmup}")
    print(f"Mixed precision: {not args.no_amp}")
    print(f"Number of runs: {args.runs}")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print("=" * 50)
    
    # Setup S3DIS model and dataloader
    try:
        model, dataloader = setup_s3dis(args.model_path, args.batch_size)
    except ImportError as e:
        print(f"Error importing S3DIS modules: {e}")
        print("Make sure the S3DIS directory exists and has the required files.")
        return
    except Exception as e:
        print(f"Error setting up S3DIS: {e}")
        return
    
    # Get model information
    param_count, model_size = get_model_info(model)
    print(f"Model parameters: {param_count:,}")
    print(f"Model size: {model_size:.2f} MB")
    print()
    
    # Run FPS measurement multiple times
    fps_results = []
    
    for run in range(args.runs):
        print(f"Run {run + 1}/{args.runs}")
        try:
            fps, total_time, sample_count = measure_fps_s3dis(
                model, dataloader, 
                args.num_samples, args.warmup, 
                not args.no_amp
            )
            fps_results.append(fps)
            print(f"  FPS: {fps:.2f}")
            print(f"  Total time: {total_time:.3f}s")
            print(f"  Samples processed: {sample_count}")
            print()
        except Exception as e:
            print(f"Error in run {run + 1}: {e}")
            continue
    
    if fps_results:
        # Calculate statistics
        fps_mean = np.mean(fps_results)
        fps_std = np.std(fps_results)
        fps_min = np.min(fps_results)
        fps_max = np.max(fps_results)
        
        print("=" * 50)
        print(f"RESULTS SUMMARY ({len(fps_results)} successful runs)")
        print("=" * 50)
        print(f"Mean FPS: {fps_mean:.2f} ± {fps_std:.2f}")
        print(f"Min FPS: {fps_min:.2f}")
        print(f"Max FPS: {fps_max:.2f}")
        print(f"Throughput: {fps_mean * args.batch_size:.2f} samples/sec")
        print(f"Latency: {1000/fps_mean:.2f} ms per sample")
        
        # Memory usage
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
            memory_reserved = torch.cuda.memory_reserved() / (1024**3)   # GB
            print(f"GPU Memory Allocated: {memory_allocated:.2f} GB")
            print(f"GPU Memory Reserved: {memory_reserved:.2f} GB")
    else:
        print("No successful runs completed.")

if __name__ == "__main__":
    main()
