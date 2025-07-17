#!/usr/bin/env python3
"""Validation script to demonstrate the new project structure."""

import subprocess
import sys
from pathlib import Path

def main():
    """Run validation tests for the new project structure."""
    print("🔍 Validating new project structure...")
    
    # Test 1: Check if the package can be imported
    print("\n1. Testing package import...")
    try:
        import online_adaptive_resnet
        print(f"✅ Package imported successfully (version: {online_adaptive_resnet.__version__})")
    except ImportError as e:
        print(f"❌ Failed to import package: {e}")
        return False
    
    # Test 2: Check if individual modules can be imported
    print("\n2. Testing module imports...")
    modules = [
        "online_adaptive_resnet.core.neural_network",
        "online_adaptive_resnet.core.entity", 
        "online_adaptive_resnet.core.dynamics",
        "online_adaptive_resnet.io.config",
        "online_adaptive_resnet.io.data_manager",
        "online_adaptive_resnet.visualization.plotter",
        "online_adaptive_resnet.utils.integrate",
        "online_adaptive_resnet.cli.main"
    ]
    
    for module in modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            return False
    
    # Test 3: Check if CLI commands work
    print("\n3. Testing CLI commands...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "online_adaptive_resnet", "--help"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0 and "Online Adaptive Deep Residual Neural Network" in result.stdout:
            print("✅ `python -m online_adaptive_resnet --help` works")
        else:
            print("❌ CLI module execution failed")
            return False
    except subprocess.TimeoutExpired:
        print("❌ CLI command timed out")
        return False
    except Exception as e:
        print(f"❌ CLI command failed: {e}")
        return False
    
    # Test 4: Check if script entry point works
    print("\n4. Testing script entry point...")
    try:
        result = subprocess.run(
            ["online-adaptive-resnet", "--help"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            print("✅ `online-adaptive-resnet --help` works")
        else:
            print("❌ Script entry point failed")
            return False
    except subprocess.TimeoutExpired:
        print("❌ Script entry point timed out")
        return False
    except Exception as e:
        print(f"❌ Script entry point failed: {e}")
        return False
    
    # Test 5: Run existing tests
    print("\n5. Running existing tests...")
    try:
        result = subprocess.run(
            [sys.executable, "tests/test_resnet_reference.py"],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0 and "test passed" in result.stdout:
            print("✅ Reference test passed")
        else:
            print("❌ Reference test failed")
            print(f"stdout: {result.stdout}")
            print(f"stderr: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("❌ Test timed out")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    
    # Test 6: Check directory structure
    print("\n6. Checking directory structure...")
    expected_structure = [
        "src/online_adaptive_resnet/__init__.py",
        "src/online_adaptive_resnet/__main__.py",
        "src/online_adaptive_resnet/core/__init__.py",
        "src/online_adaptive_resnet/core/neural_network.py",
        "src/online_adaptive_resnet/core/entity.py",
        "src/online_adaptive_resnet/core/dynamics.py",
        "src/online_adaptive_resnet/io/__init__.py",
        "src/online_adaptive_resnet/io/config.py",
        "src/online_adaptive_resnet/io/data_manager.py",
        "src/online_adaptive_resnet/visualization/__init__.py",
        "src/online_adaptive_resnet/visualization/plotter.py",
        "src/online_adaptive_resnet/utils/__init__.py",
        "src/online_adaptive_resnet/utils/integrate.py",
        "src/online_adaptive_resnet/cli/__init__.py",
        "src/online_adaptive_resnet/cli/main.py",
        "tests/__init__.py",
        "tests/test_resnet_reference.py",
        "pyproject.toml",
        "config.json"
    ]
    
    all_exist = True
    for path in expected_structure:
        if Path(path).exists():
            print(f"✅ {path}")
        else:
            print(f"❌ {path} not found")
            all_exist = False
    
    if not all_exist:
        return False
    
    # Test 7: Check that old files are cleaned up (not done yet)
    print("\n7. Checking for old files...")
    old_files = [
        "main.py",
        "neural_network.py",
        "entity.py",
        "dynamics.py",
        "data_manager.py",
        "plotter.py",
        "integrate.py"
    ]
    
    old_files_exist = []
    for file in old_files:
        if Path(file).exists():
            old_files_exist.append(file)
    
    if old_files_exist:
        print(f"⚠️  Old files still exist: {old_files_exist}")
        print("   (This is expected during reorganization)")
    else:
        print("✅ All old files have been cleaned up")
    
    print("\n🎉 All validation tests passed!")
    print("\nThe new project structure is working correctly:")
    print("- Package can be imported and used")
    print("- All modules are properly organized")
    print("- CLI commands work as expected")
    print("- Tests pass with the new structure")
    print("- Directory structure follows src-layout conventions")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)