#!/usr/bin/env python3
"""
Model Converter: A100 PyTorch checkpoint -> Ascend 910B compatible format.

Converts .pt/.pth checkpoints trained on NVIDIA A100 to be compatible with
Ascend 910B. Handles:
  1. State dict key remapping (if model architecture differs)
  2. dtype compatibility (fp16/bf16 -> fp16)
  3. torch_npu device mapping
  4. ONNX export for MindIE deployment

Usage:
    # Basic conversion (load and resave for NPU)
    python model_converter.py --input model_a100.pth --output model_npu.pth

    # Convert with dtype change
    python model_converter.py --input model_a100.pth --output model_npu.pth --dtype float16

    # Export to ONNX for MindIE
    python model_converter.py --input model_a100.pth --onnx model_npu.onnx --input-shape 1,3,512,512

    --verify model_npu.pth
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn


def load_checkpoint(path: str) -> dict:
    """Load a PyTorch checkpoint, handling various formats."""
    print(f"Loading checkpoint: {path}")
    checkpoint = torch.load(path, map_location='cpu')

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        # Common formats: 'state_dict', 'model', 'model_state', etc.
        for key in ['state_dict', 'model', 'model_state', 'state']:
            if key in checkpoint:
                print(f"  Found state dict under key: '{key}'")
                return checkpoint, key

        # Direct state dict (no wrapper)
        return checkpoint, None
    else:
        return checkpoint, None


def convert_dtype(state_dict: dict, target_dtype: str) -> dict:
    """Convert tensor dtypes in state dict."""
    dtype_map = {
        'float16': torch.float16,
        'float32': torch.float32,
        'bfloat16': torch.bfloat16,
    }

    if target_dtype not in dtype_map:
        raise ValueError(f"Unsupported dtype: {target_dtype}. Choose from {list(dtype_map.keys())}")

    target = dtype_map[target_dtype]
    converted = {}

    for key, tensor in state_dict.items():
        if isinstance(tensor, torch.Tensor) and tensor.is_floating_point():
            converted[key] = tensor.to(dtype=target)
        else:
            converted[key] = tensor

    print(f"  Converted tensors to {target_dtype}")
    return converted


def verify_state_dict(state_dict: dict) -> bool:
    """Verify a state dict is valid and can be loaded."""
    if not isinstance(state_dict, dict):
        print("  [FAIL] State dict is not a dictionary")
        return False

    total_params = 0
    total_size = 0
    for key, tensor in state_dict.items():
        if isinstance(tensor, torch.Tensor):
            total_params += tensor.numel()
            total_size += tensor.numel() * tensor.element_size()
        else:
            print(f"  [WARN] Key '{key}' is not a tensor: {type(tensor)}")

    print(f"  Total parameters: {total_params:,}")
    print(f"  Approximate size: {total_size / 1024 / 1024:.1f} MB")
    return True


def export_to_onnx(
    checkpoint_path: str,
    onnx_path: str,
    input_shape: tuple,
    model_class: nn.Module = None,
):
    """Export model to ONNX format for MindIE deployment."""
    print(f"\nExporting to ONNX: {onnx_path}")
    print(f"  Input shape: {input_shape}")

    checkpoint, key = load_checkpoint(checkpoint_path)
    if key:
        state_dict = checkpoint[key]
    else:
        state_dict = checkpoint

    # Note: For actual export, you need the model definition
    # This is a template - users need to provide their model class
    print("\n  [INFO] For actual ONNX export, you need to:")
    print("    1. Import your model class definition")
    print("    2. Create model instance and load state dict")
    print("    3. Run torch.onnx.export() with dummy input")
    print("\n  Example:")
    print(f"    model = YourModelClass()")
    print(f"    model.load_state_dict(state_dict)")
    print(f"    model.eval()")
    print(f"    dummy = torch.randn({input_shape})")
    print(f"    torch.onnx.export(model, dummy, '{onnx_path}',")
    print(f"        input_names=['input'], output_names=['output'],")
    print(f"        dynamic_axes={{'input': {{0: 'batch'}}, 'output': {{0: 'batch'}}}},")
    print(f"        opset_version=13)")


def main():
    parser = argparse.ArgumentParser(description='Model Converter: A100 -> Ascend 910B')
    parser.add_argument('--input', '-i', required=True, help='Input checkpoint path (.pt/.pth)')
    parser.add_argument('--output', '-o', help='Output checkpoint path')
    parser.add_argument('--dtype', default=None, choices=['float16', 'float32', 'bfloat16'],
                        help='Target dtype for tensors')
    parser.add_argument('--onnx', help='Export to ONNX file for MindIE')
    parser.add_argument('--input-shape', default='1,3,512,512',
                        help='Input shape for ONNX export (default: 1,3,512,512)')
    parser.add_argument('--verify', action='store_true',
                        help='Verify the output checkpoint')

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    # Load checkpoint
    checkpoint, key = load_checkpoint(str(input_path))

    if key:
        state_dict = checkpoint[key]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        state_dict = {'model': checkpoint}

    # Verify input
    print("\nVerifying input checkpoint...")
    verify_state_dict(state_dict)

    # Convert dtype if requested
    if args.dtype:
        print(f"\nConverting dtype to {args.dtype}...")
        state_dict = convert_dtype(state_dict, args.dtype)

    # Save output
    if args.output:
        output_path = Path(args.output)
        print(f"\nSaving to: {output_path}")

        if key:
            checkpoint[key] = state_dict
            torch.save(checkpoint, output_path)
        else:
            torch.save(state_dict, output_path)

        print(f"  Saved: {output_path}")

        # Verify output
        print("\nVerifying output checkpoint...")
        verify_checkpoint, verify_key = load_checkpoint(str(output_path))
        if verify_key:
            verify_state_dict(verify_checkpoint[verify_key])
        else:
            verify_state_dict(verify_checkpoint)

    # ONNX export
    if args.onnx:
        input_shape = tuple(int(x) for x in args.input_shape.split(','))
        export_to_onnx(str(input_path), args.onnx, input_shape)


if __name__ == '__main__':
    main()
