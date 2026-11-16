"""
Environment and Project Structure Check Script
Verifies that all required packages and project modules are properly configured.
"""
import sys

def check_imports():
    print("--- Checking Required Packages ---")
    packages = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("numpy", "NumPy"),
        ("sklearn", "Scikit-Learn"),
        ("matplotlib", "Matplotlib"),
        ("PIL", "Pillow"),
        ("pytest", "PyTest")
    ]
    
    all_passed = True
    for pkg_name, display_name in packages:
        try:
            mod = __import__(pkg_name)
            version = getattr(mod, "__version__", "Unknown")
            print(f"[OK] {display_name} (v{version}) is successfully installed.")
        except ImportError as e:
            print(f"[ERROR] Failed to import {display_name}: {e}")
            all_passed = False
            
    print("\n--- Checking Project Modules ---")
    modules = ["data", "nn", "search", "csp", "integration", "eval", "tests"]
    for mod_name in modules:
        try:
            __import__(mod_name)
            print(f"[OK] Module '{mod_name}' is accessible.")
        except ImportError as e:
            print(f"[ERROR] Failed to import module '{mod_name}': {e}")
            all_passed = False
            
    if all_passed:
        print("\n🎉 Environment setup complete and fully functional!")
    else:
        print("\n❌ Setup incomplete. Please fix the missing packages above.")

if __name__ == "__main__":
    check_imports()