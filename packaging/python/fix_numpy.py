#!/usr/bin/env python3
"""
NumPy Fix Script for silk.ai

Run this script if you encounter "numpy core multiarray failed to import" errors.
This script will diagnose and fix numpy installation issues using an isolated environment.
"""

import sys
import os
import subprocess
import platform

def setup_isolated_environment():
    """Setup isolated Python environment to avoid system package conflicts."""
    # Get the app cache directory
    if platform.system() == 'Darwin':  # macOS
        app_cache = os.path.join(os.path.expanduser('~'), 'Library', 'Application Support', 'silk.ai')
    elif platform.system() == 'Windows':
        app_cache = os.path.join(os.environ.get('APPDATA', os.path.expanduser('~')), 'silk.ai')
    else:  # Linux
        app_cache = os.path.join(os.path.expanduser('~'), '.config', 'silk.ai')
    
    user_base = os.path.join(app_cache, 'python_packages')
    
    # Set environment variables for isolation
    os.environ['PYTHONUSERBASE'] = user_base
    os.environ['PYTHONNOUSERSITE'] = '0'  # Allow user site packages
    os.environ['PYTHONSAFEPATH'] = '1'    # Don't add current directory to sys.path
    
    print(f"🔧 Using isolated environment: {user_base}")
    return user_base

def print_header():
    print("🧮 silk.ai NumPy Fix Script (Isolated Environment)")
    print("=" * 50)
    print(f"Python: {sys.version}")
    print(f"Platform: {platform.system()} {platform.machine()}")
    print(f"Architecture: {platform.architecture()[0]}")
    print("=" * 50)

def test_numpy():
    """Test if numpy is working correctly."""
    print("\n🔍 Testing NumPy...")
    try:
        import numpy as np
        print(f"✅ NumPy version: {np.__version__}")
        print(f"✅ NumPy location: {np.__file__}")
        
        # Test core multiarray (common failure point)
        from numpy.core.multiarray import _reconstruct
        print("✅ Core multiarray import: OK")
        
        # Test basic operations
        arr = np.array([1, 2, 3, 4, 5])
        result = arr.sum()
        print(f"✅ Array operations: {result} (expected: 15)")
        
        # Test OpenCV compatibility
        try:
            import cv2
            print(f"✅ OpenCV {cv2.__version__} is compatible with NumPy {np.__version__}")
            return True
        except Exception as e:
            print(f"❌ OpenCV compatibility issue: {e}")
            return False
            
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ NumPy error: {e}")
        return False

def fix_numpy():
    """Fix numpy installation using isolated environment."""
    print("\n🔧 Fixing NumPy installation in isolated environment...")
    
    # Uninstall existing numpy
    print("🗑️ Uninstalling existing NumPy...")
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'uninstall', 'numpy', '-y'], 
                      check=False, capture_output=True)
        print("✅ Existing NumPy uninstalled")
    except Exception as e:
        print(f"⚠️ Could not uninstall existing NumPy: {e}")
    
    # Install fresh numpy with platform-specific optimizations
    print("📦 Installing NumPy 1.x with platform optimizations...")
    
    install_args = [sys.executable, '-m', 'pip', 'install', '--user', '--upgrade', '--force-reinstall']
    
    # Platform-specific optimizations
    system = platform.system()
    machine = platform.machine()
    
    if system == 'Darwin':  # macOS
        if machine == 'arm64':
            print("🍎 Detected Apple Silicon - using optimized wheels")
        else:
            print("🍎 Detected Intel Mac - using optimized wheels")
        install_args.extend(['--only-binary=:all:', 'numpy>=1.21.0,<2.0.0'])
    elif system == 'Windows':
        print("🪟 Detected Windows - using pre-compiled wheels")
        install_args.extend(['--only-binary=:all:', 'numpy>=1.21.0,<2.0.0'])
    else:  # Linux
        print("🐧 Detected Linux - using compatible wheels")
        install_args.extend(['numpy>=1.21.0,<2.0.0'])
    
    try:
        result = subprocess.run(install_args, check=True, capture_output=True, text=True)
        print("✅ NumPy 1.x installation completed")
        print(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ NumPy installation failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    print_header()
    
    # Setup isolated environment first
    setup_isolated_environment()
    
    # Test current numpy
    if test_numpy():
        print("\n🎉 NumPy is already working correctly!")
        print("If you're still experiencing issues, try restarting the silk.ai app.")
        return
    
    # Ask user if they want to fix it
    print(f"\n❓ NumPy is not working correctly.")
    response = input("Would you like to try fixing it? (y/n): ").lower()
    
    if response.startswith('y'):
        # Try to fix numpy
        if fix_numpy():
            print("\n🔍 Testing fixed NumPy...")
            if test_numpy():
                print("\n🎉 NumPy has been fixed successfully!")
                print("Please restart the silk.ai app to use the fixed NumPy.")
            else:
                print("\n❌ NumPy is still not working after the fix attempt.")
                print("Manual intervention may be required.")
        else:
            print("\n❌ Failed to fix NumPy automatically.")
            print("Manual intervention may be required.")
    else:
        print("Exiting without making changes.")
    
    print("\n📞 If issues persist, please contact support with the output above.")

if __name__ == "__main__":
    main() 