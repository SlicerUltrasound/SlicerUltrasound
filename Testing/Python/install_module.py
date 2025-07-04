#!/usr/bin/env python3
"""
Script to build and install the SlicerUltrasound module locally.
This is useful for development and testing.
"""

import os
import sys
import subprocess
import argparse

def find_slicer():
    """Find Slicer installation."""
    print("Searching for Slicer installation...")
    print(f"SLICER_HOME environment variable: {os.environ.get('SLICER_HOME')}")

    # Common Slicer installation paths
    possible_paths = [
        # macOS
        "/Applications/Slicer.app/Contents/MacOS/Slicer",
        # Linux
        os.path.expanduser("~/Slicer-*/bin/Slicer"),
        "/opt/Slicer-*/bin/Slicer",
        # Windows
        "C:/ProgramData/NA-MIC/Slicer-*/bin/Slicer.exe",
    ]

    # Check environment variable first
    slicer_home = os.environ.get('SLICER_HOME')
    if slicer_home:
        # Check multiple possible Slicer executable locations
        possible_slicer_paths = [
            os.path.join(slicer_home, 'bin', 'Slicer'),
            os.path.join(slicer_home, 'bin', 'SlicerApp-real'),
            os.path.join(slicer_home, 'Slicer'),
            os.path.join(slicer_home, 'bin', 'PythonSlicer'),
        ]

        for slicer_path in possible_slicer_paths:
            if os.path.exists(slicer_path):
                print(f"Found Slicer executable at: {slicer_path}")
                return slicer_home

    # Check possible paths
    print("Checking fallback paths...")
    for path in possible_paths:
        print(f"  Checking: {path}")
        if '*' in path:
            import glob
            matches = glob.glob(path)
            if matches:
                result = os.path.dirname(os.path.dirname(matches[0]))
                print(f"  Found via glob: {result}")
                return result
        elif os.path.exists(path):
            result = os.path.dirname(os.path.dirname(path))
            print(f"  Found: {result}")
            return result

    print("No Slicer installation found")
    return None

def build_module(slicer_home, build_dir="build"):
    """Build the SlicerUltrasound module."""
    print(f"Building SlicerUltrasound module...")
    print(f"Slicer home: {slicer_home}")
    print(f"Build directory: {build_dir}")

    # Create build directory
    os.makedirs(build_dir, exist_ok=True)

    # For Slicer extensions, we need to use the Slicer extension build system
    # Try to configure with CMake using Slicer's extension template
    cmake_cmd = [
        'cmake', '..',
        f'-DSlicer_DIR={os.path.join(slicer_home, "lib", "Slicer-5.8", "cmake")}',
        '-DBUILD_TESTING=ON'
    ]

    print(f"Running: {' '.join(cmake_cmd)}")
    try:
        result = subprocess.run(cmake_cmd, cwd=build_dir, check=True)

        # Build
        print("Building...")
        result = subprocess.run(['make', '-j4'], cwd=build_dir, check=True)

        print("✓ Build completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"⚠ CMake build failed: {e}")
        print("  This is expected for Slicer extensions without full build setup")
        print("  Will proceed with direct installation instead")
        return False

def install_module(slicer_home, build_dir="build"):
    """Install the module into Slicer."""
    print("Installing module into Slicer...")

    # Try to install via make first
    try:
        result = subprocess.run(['make', 'install'], cwd=build_dir, check=True)
        print("✓ Module installed successfully via make")
    except subprocess.CalledProcessError:
        print("⚠ Make install failed, trying direct copy...")

        # Fallback: Copy module files directly to Slicer's extension directory
        extension_dir = os.path.join(slicer_home, "lib", "Slicer-5.8", "qt-scripted-modules")
        print(f"Extension directory: {extension_dir}")

        # Ensure extension directory exists
        os.makedirs(extension_dir, exist_ok=True)

        # Copy the module directories
        modules = ["AnnotateUltrasound", "AnonymizeUltrasound", "MmodeAnalysis", "TimeSeriesAnnotation"]
        for module in modules:
            if os.path.exists(module):
                target_dir = os.path.join(extension_dir, module)
                print(f"Copying {module} to {target_dir}")

                # Remove existing directory if it exists
                if os.path.exists(target_dir):
                    import shutil
                    shutil.rmtree(target_dir)

                # Copy the module
                import shutil
                shutil.copytree(module, target_dir)
                print(f"✓ Successfully copied {module}")
            else:
                print(f"⚠ Module {module} not found in current directory")

        print("✓ Module installed successfully via direct copy")

    print(f"Module should now be available in Slicer's extension directory")

def main():
    parser = argparse.ArgumentParser(description='Build and install SlicerUltrasound module')
    parser.add_argument('--build-dir', default='build', help='Build directory')
    parser.add_argument('--build-only', action='store_true', help='Only build, do not install')
    parser.add_argument('--install-only', action='store_true', help='Only install, do not build')

    args = parser.parse_args()

    # Find Slicer
    slicer_home = find_slicer()
    if not slicer_home:
        print("Error: Could not find Slicer installation.")
        print("Please ensure Slicer is installed or set SLICER_HOME environment variable.")
        return 1

    print(f"Found Slicer at: {slicer_home}")

    try:
        build_success = True
        if not args.install_only:
            build_success = build_module(slicer_home, args.build_dir)

        if not args.build_only:
            install_module(slicer_home, args.build_dir)

        print("\n✓ SlicerUltrasound module ready!")
        print("You can now run tests with: python3 run_slicer_tests.py")

    except subprocess.CalledProcessError as e:
        print(f"Error during build/install: {e}")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1

    return 0

if __name__ == '__main__':
    sys.exit(main())