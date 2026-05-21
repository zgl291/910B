#!/usr/bin/env python3
"""
Quick NPU Test - Verify torch_npu works with a simple forward pass.
Run inside the container on the 910B server.
"""

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """Simple CNN for testing NPU functionality."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def test_npu():
    """Test basic NPU operations."""
    print("=" * 50)
    print("  Ascend 910B Quick Test")
    print("=" * 50)

    # Check NPU availability
    if not hasattr(torch, 'npu'):
        print("[ERROR] torch.npu not available.")
        print("  Please ensure torch_npu is installed and CANN is available.")
        return False

    if not torch.npu.is_available():
        print("[ERROR] NPU is not available.")
        print("  Check: npu-smi info")
        print("  Check: ls -la /dev/davinci*")
        return False

    device_count = torch.npu.device_count()
    print(f"[OK] NPU device count: {device_count}")

    # Create model and move to NPU
    print("[OK] Creating model...")
    model = SimpleCNN()

    device = torch.device("npu:0")
    print(f"[OK] Moving model to {device}...")
    model = model.to(device)

    # Forward pass
    print("[OK] Running forward pass...")
    x = torch.randn(4, 3, 64, 64, device=device)
    with torch.no_grad():
        y = model(x)

    print(f"[OK] Forward pass output shape: {y.shape}")

    # Backward pass
    print("[OK] Running backward pass...")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    target = torch.randn(4, 10, device=device)
    loss = nn.MSELoss()(y, target)
    loss.backward()
    optimizer.step()

    print(f"[OK] Loss: {loss.item():.4f}")
    print(f"[OK] Backward pass complete!")

    # Multi-card test (if available)
    if device_count > 1:
        print(f"[OK] Testing multi-NPU ({device_count} cards)...")
        print("  (Distributed training requires torchrun, skipped for quick test)")
    else:
        print("[INFO] Single NPU detected. Multi-card test skipped.")

    print("=" * 50)
    print("  All tests passed!")
    print("=" * 50)
    return True


if __name__ == "__main__":
    success = test_npu()
    sys_exit_code = 0 if success else 1
    import sys
    sys.exit(sys_exit_code)
