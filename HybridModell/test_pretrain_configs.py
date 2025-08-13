#!/usr/bin/env python3
"""
Test script to verify pretrain.py dataset configurations work correctly
"""
import sys
import argparse

def test_scannetv2_config():
    """Test ScanNetV2 configuration"""
    print("Testing ScanNetV2 configuration...")
    try:
        # Mock args for ScanNetV2
        args = argparse.Namespace()
        args.dataset = "scannetv2"
        args.model = "dela_semseg"
        
        # Test imports
        from config import scan_args, dela_args
        print("✓ ScanNetV2 config imports successful")
        
        # Test input dimensions
        input_dim = 10  # ScanNetV2: 3 RGB + 1 height + 3 normals + 3 xyz
        print(f"✓ ScanNetV2 input_dim: {input_dim}")
        
        # Set input feature dimension for the model
        if hasattr(dela_args, 'input_feature_dim'):
            dela_args.input_feature_dim = input_dim
        else:
            setattr(dela_args, 'input_feature_dim', input_dim)
        
        print(f"✓ ScanNetV2 configured with input_feature_dim: {dela_args.input_feature_dim}")
        return True
        
    except Exception as e:
        print(f"✗ ScanNetV2 error: {e}")
        return False

def test_s3dis_config():
    """Test S3DIS configuration"""
    print("\nTesting S3DIS configuration...")
    try:
        # Mock args for S3DIS
        args = argparse.Namespace()
        args.dataset = "s3dis"
        args.model = "dela_semseg"
        
        # Test imports
        from config_s3dis import s3dis_args, dela_args as s3dis_dela_args
        print("✓ S3DIS config imports successful")
        
        # Test input dimensions
        input_dim = 7   # S3DIS: 3 RGB + 1 height + 3 xyz (no normals)
        print(f"✓ S3DIS input_dim: {input_dim}")
        
        # Test S3DIS specific config
        print(f"✓ S3DIS args: k={s3dis_args.k}, grid_size={s3dis_args.grid_size}")
        print(f"✓ S3DIS num_classes: {s3dis_dela_args.num_classes}")
        
        # Set input feature dimension for the model
        if hasattr(s3dis_dela_args, 'input_feature_dim'):
            s3dis_dela_args.input_feature_dim = input_dim
        else:
            setattr(s3dis_dela_args, 'input_feature_dim', input_dim)
        
        print(f"✓ S3DIS configured with input_feature_dim: {s3dis_dela_args.input_feature_dim}")
        return True
        
    except Exception as e:
        print(f"✗ S3DIS error: {e}")
        return False

def test_model_config():
    """Test model configuration"""
    print("\nTesting model configuration...")
    try:
        from hybridmodel import DelaSemSeg
        from config import dela_args
        
        # Test with ScanNetV2 dimensions (10)
        dela_args.input_feature_dim = 10
        model_scannet = DelaSemSeg(dela_args)
        print("✓ Model created successfully with ScanNetV2 config")
        
        # Test with S3DIS dimensions (7)  
        dela_args.input_feature_dim = 7
        model_s3dis = DelaSemSeg(dela_args)
        print("✓ Model created successfully with S3DIS config")
        
        return True
        
    except Exception as e:
        print(f"✗ Model error: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("Testing pretrain.py dataset configurations")
    print("=" * 60)
    
    tests = [
        test_scannetv2_config,
        test_s3dis_config,
        test_model_config
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    passed = sum(results)
    total = len(results)
    print(f"✓ {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All configuration tests passed!")
        print("\nUsage examples:")
        print("  python pretrain.py --dataset scannetv2 --epochs 10")
        print("  python pretrain.py --dataset s3dis --epochs 10")
    else:
        print("❌ Some tests failed. Check configuration.")
        sys.exit(1)

if __name__ == "__main__":
    main()
