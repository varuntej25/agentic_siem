#!/usr/bin/env python3
"""
GPU Acceleration Test Script
===========================

Quick test to verify GPU acceleration is working properly.
"""

import sys
import os
sys.path.append(os.getcwd())

def test_gpu_utils():
    """Test GPU utilities and device detection"""
    print("🔧 Testing GPU Acceleration Utilities...")
    print("-" * 50)
    
    try:
        from gpu_utils import (
            get_device_info, get_device, get_accelerator,
            to_device, autocast, get_optimal_batch_size,
            optimize_transformers_model
        )
        import torch
        
        # Test device detection
        device = get_device()
        device_info = get_device_info()
        
        print(f"📱 Device: {device}")
        print(f"📊 Device Info: {device_info}")
        
        # Test tensor operations
        print("\n🧮 Testing tensor operations...")
        test_tensor = torch.randn(100, 384)
        gpu_tensor = to_device(test_tensor)
        print(f"✅ Tensor moved to: {gpu_tensor.device}")
        
        # Test autocast
        print("\n⚡ Testing autocast...")
        with autocast():
            result = gpu_tensor @ gpu_tensor.T
        print(f"✅ Autocast operation completed: {result.shape}")
        
        # Test batch size optimization
        optimal_batch = get_optimal_batch_size()
        print(f"\n📦 Optimal batch size: {optimal_batch}")
        
        print("\n🎉 GPU utilities test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ GPU utilities test failed: {e}")
        return False

def test_model_acceleration():
    """Test model acceleration with transformers"""
    print("\n🤖 Testing Model Acceleration...")
    print("-" * 50)
    
    try:
        from transformers import AutoTokenizer, AutoModel
        from gpu_utils import optimize_transformers_model, to_device, autocast
        
        # Load a small model for testing
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        print(f"Loading model: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        # Optimize for GPU
        model, tokenizer = optimize_transformers_model(model, tokenizer)
        print(f"✅ Model optimized and moved to: {next(model.parameters()).device}")
        
        # Test inference
        test_texts = ["This is a test sentence.", "Another test sentence for GPU acceleration."]
        inputs = tokenizer(test_texts, return_tensors='pt', padding=True, truncation=True)
        inputs = to_device(inputs)
        
        with autocast():
            with torch.no_grad():
                outputs = model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :]
        
        print(f"✅ Inference completed: {embeddings.shape}")
        print(f"📍 Embeddings device: {embeddings.device}")
        
        print("\n🎉 Model acceleration test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Model acceleration test failed: {e}")
        return False

def test_agent_integration():
    """Test GPU acceleration with security agents"""
    print("\n🛡️ Testing Security Agent GPU Integration...")
    print("-" * 50)
    
    try:
        from master_agent import MasterAgent
        
        # Initialize agent (should automatically use GPU)
        agent = MasterAgent()
        print(f"✅ MasterAgent initialized with device: {agent.device}")
        
        # Test embedding generation
        test_texts = ["Security alert: Failed login attempt", "Network anomaly detected"]
        embeddings = agent._get_embeddings(test_texts)
        print(f"✅ Embeddings generated: {embeddings.shape}")
        
        print("\n🎉 Security agent GPU integration test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Security agent GPU test failed: {e}")
        return False

def main():
    """Run all GPU acceleration tests"""
    print("🚀 GPU Acceleration Test Suite")
    print("=" * 60)
    
    results = []
    
    # Test GPU utilities
    results.append(test_gpu_utils())
    
    # Test model acceleration
    results.append(test_model_acceleration())
    
    # Test agent integration
    results.append(test_agent_integration())
    
    # Summary
    print("\n📊 Test Results Summary")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"🎉 All tests passed! ({passed}/{total})")
        print("✅ GPU acceleration is working properly")
    else:
        print(f"⚠️  Some tests failed ({passed}/{total})")
        if passed > 0:
            print("🔄 GPU acceleration partially working")
        else:
            print("❌ GPU acceleration not working, falling back to CPU")
    
    return passed == total

if __name__ == "__main__":
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device: {torch.cuda.get_device_name()}")
    print()
    
    success = main()
    sys.exit(0 if success else 1)
