#!/usr/bin/env python3
"""
Test script to verify Teacher Model installation and dependencies
"""

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__} imported successfully")
        print(f"  - CUDA available: {torch.cuda.is_available()}")
        print(f"  - MPS available: {torch.backends.mps.is_available()}")
        
        import numpy as np
        print(f"✓ NumPy {np.__version__} imported successfully")
        
        import pandas as pd
        print(f"✓ Pandas {pd.__version__} imported successfully")
        
        import librosa
        print(f"✓ Librosa {librosa.__version__} imported successfully")
        
        import matplotlib.pyplot as plt
        print(f"✓ Matplotlib imported successfully")
        
        import seaborn as sns
        print(f"✓ Seaborn imported successfully")
        
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder, StandardScaler
        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
        print(f"✓ Scikit-learn imported successfully")
        
        print("\nAll imports successful! ✓")
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("Please install missing dependencies using: pip install -r requirements.txt")
        return False

def test_device():
    """Test device configuration"""
    print("\nTesting device configuration...")
    
    try:
        import torch
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"✓ Using device: {device}")
        
        if device.type == "mps":
            print("  - MPS (Metal Performance Shaders) is available for MacBook Air M2")
        else:
            print("  - Using CPU (MPS not available)")
            
        return True
        
    except Exception as e:
        print(f"✗ Device test failed: {e}")
        return False

def test_model_creation():
    """Test if Teacher Model can be created"""
    print("\nTesting Teacher Model creation...")
    
    try:
        from teacher_model import TeacherModel, create_model
        import torch
        
        # Test model creation with dummy parameters
        input_size = 200  # Dummy input size
        num_classes = 8   # Number of emotion classes
        
        model = create_model(input_size, num_classes)
        print(f"✓ Teacher Model created successfully")
        print(f"  - Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  - Model device: {next(model.parameters()).device}")
        
        # Test forward pass with dummy data
        dummy_input = torch.randn(1, input_size).to(next(model.parameters()).device)
        with torch.no_grad():
            output = model(dummy_input)
        print(f"✓ Forward pass successful")
        print(f"  - Output shape: {output.shape}")
        print(f"  - Output device: {output.device}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model creation test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("TEACHER MODEL INSTALLATION TEST")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_device,
        test_model_creation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print("-" * 40)
    
    print(f"\nTest Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your Teacher Model is ready to use.")
        print("\nNext steps:")
        print("1. Update DATA_DIR in teacher_model.py with your dataset path")
        print("2. Run: python teacher_model.py")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")

if __name__ == "__main__":
    main()
