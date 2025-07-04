#!/usr/bin/env python3
"""
Script to install dependencies in Slicer's Python environment.
This script should be run via: Slicer --python-script install_deps_in_slicer.py
"""

import sys
import os
import slicer

def install_dependencies():
    """Install dependencies in Slicer's Python environment."""
    print(f"Installing dependencies in Slicer's Python: {sys.executable}")
    print(f"Python version: {sys.version}")

    # Essential testing packages (without version constraints for compatibility)
    essential_packages = [
        "pytest",
        "pytest-cov",
        "pytest-mock"
    ]

    print(f"Installing essential packages: {essential_packages}")

    # Install each package using slicer.util.pip_install
    success_count = 0
    for package in essential_packages:
        try:
            print(f"Installing {package}...")
            slicer.util.pip_install(package)
            print(f"✅ Successfully installed {package}")
            success_count += 1
        except Exception as e:
            print(f"❌ Failed to install {package}: {e}")
            print(f"   This might be due to Slicer's Python environment restrictions.")

    print(f"\nInstallation summary: {success_count}/{len(essential_packages)} packages installed successfully")
    return success_count > 0  # Return True if at least one package was installed

def check_installed_packages():
    """Check which packages are installed."""
    print("\nChecking installed packages:")

    packages_to_check = ["pytest", "pytest_cov", "pytest_mock", "PySide2"]

    for package in packages_to_check:
        try:
            __import__(package)
            print(f"✅ {package} is installed and importable")
        except ImportError:
            print(f"❌ {package} is NOT installed or not importable")
        except Exception as e:
            print(f"❌ Error checking {package}: {e}")

def check_slicer_pip_available():
    """Check if slicer.util.pip_install is available."""
    print("\nChecking Slicer pip installation method:")
    try:
        if hasattr(slicer.util, 'pip_install'):
            print("✅ slicer.util.pip_install is available")
            return True
        else:
            print("❌ slicer.util.pip_install is NOT available")
            return False
    except Exception as e:
        print(f"❌ Error checking slicer.util.pip_install: {e}")
        return False

if __name__ == "__main__":
    print("=== Installing Dependencies in Slicer's Python Environment ===")

    # First check if the method is available
    if not check_slicer_pip_available():
        print("\n❌ Cannot install packages - slicer.util.pip_install not available")
        print("   You may need to install packages manually or use a different method.")
        sys.exit(1)

    success = install_dependencies()

    if success:
        check_installed_packages()
        print("\n✅ Installation complete! Quitting Slicer...")
        # Quit Slicer after successful installation
        slicer.util.quit()
    else:
        print("\n❌ Installation failed. Please check the errors above.")
        print("\nAlternative approaches:")
        print("1. Install packages manually in Slicer's Python console:")
        print("   slicer.util.pip_install('pytest')")
        print("   slicer.util.pip_install('pytest-cov')")
        print("   slicer.util.pip_install('pytest-mock')")
        print("2. Use system Python for development: make install-deps-system")
        # Quit Slicer even on failure
        slicer.util.quit()
        sys.exit(1)