#!/usr/bin/env python3
"""
Ascend 910B Environment Verification Script
Run this inside the container to verify all dependencies are correctly installed.
"""

import sys
import importlib
import subprocess
import platform

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
WARN = "\033[93mWARN\033[0m"


def check_import(name, version_attr=None, expected_version=None):
    """Check if a package can be imported."""
    try:
        mod = importlib.import_module(name)
        version = getattr(mod, "__version__", getattr(mod, "version", "unknown"))
        status = PASS
        if expected_version and version != expected_version:
            status = WARN
        print(f"  [{status}] {name} (version: {version})")
        return True
    except ImportError as e:
        print(f"  [{FAIL}] {name}: {e}")
        return False
    except Exception as e:
        print(f"  [{FAIL}] {name}: {e}")
        return False


def check_npu():
    """Check NPU availability."""
    print("\n[NPU Check]")
    try:
        import torch
        has_npu = hasattr(torch, 'npu') and torch.npu.is_available()
        if has_npu:
            print(f"  [{PASS}] torch.npu is available")
            device_count = torch.npu.device_count()
            print(f"  [{PASS}] NPU device count: {device_count}")
            for i in range(device_count):
                props = torch.npu.get_device_properties(i)
                print(f"  [{PASS}] NPU {i}: {props.name}")
        else:
            print(f"  [{WARN}] torch.npu not available (expected in non-NPU environment)")
            # Try to check if torch_npu is installed
            try:
                import torch_npu
                print(f"  [{PASS}] torch_npu module imported successfully")
                print(f"  [{WARN}] NPU hardware not detected - running on CPU-only host")
            except ImportError:
                print(f"  [{FAIL}] torch_npu not installed")
        return has_npu
    except Exception as e:
        print(f"  [{FAIL}] Error checking NPU: {e}")
        return False


def check_cann():
    """Check CANN installation."""
    print("\n[CANN Check]")
    import os
    cann_paths = [
        "/usr/local/Ascend",
        "/usr/local/Ascend/nnae",
        "/usr/local/Ascend/ascend-toolkit",
    ]
    found = False
    for path in cann_paths:
        if os.path.exists(path):
            print(f"  [{PASS}] CANN path exists: {path}")
            found = True
    if not found:
        print(f"  [{WARN}] CANN not found in standard paths (expected on non-Ascend host)")

    # Check HCCL
    try:
        from hccl import api as hccl
        print(f"  [{PASS}] HCCL available")
    except ImportError:
        print(f"  [{WARN}] HCCL not available (expected on non-Ascend host)")


def main():
    print("=" * 60)
    print("  Ascend 910B Environment Verification")
    print("=" * 60)

    # System info
    print(f"\n[System Info]")
    print(f"  Python: {platform.python_version()}")
    print(f"  Platform: {platform.platform()}")
    print(f"  Machine: {platform.machine()}")

    # Core packages
    print("\n[Core Packages]")
    core_packages = [
        "numpy", "yaml", "typing_extensions", "future", "six",
        "requests", "tqdm", "pytz", "loguru", "easydict",
        "tabulate", "zipp", "dateutil", "packaging", "psutil",
        "PIL", "scipy",
    ]
    for pkg in core_packages:
        check_import(pkg)

    # Image processing
    print("\n[Image Processing]")
    image_pkgs = [
        "cv2", "matplotlib", "skimage", "imageio", "tifffile",
        "pywt", "albumentations", "gdal",
    ]
    for pkg in image_pkgs:
        check_import(pkg)

    # AI Framework
    print("\n[AI Framework]")
    ai_pkgs = [
        "torch", "torchvision", "timm", "thop",
        "mmcv", "mmengine", "mmdet",
        "ultralytics", "onnx",
    ]
    for pkg in ai_pkgs:
        check_import(pkg)

    # Remote sensing
    print("\n[Remote Sensing]")
    rs_pkgs = [
        "xmltodict", "shapely", "openpyxl", "pyproj",
    ]
    for pkg in rs_pkgs:
        check_import(pkg)

    # NPU / CANN
    check_npu()
    check_cann()

    print("\n" + "=" * 60)
    print("  Verification Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
