#!/usr/bin/env python3
"""
A100 -> Ascend 910B Code Adapter
Automatically patches PyTorch code written for NVIDIA GPU to run on Ascend NPU.

Usage:
    # Scan and report changes needed
    python code_adapter.py --scan /path/to/your/project

    # Auto-patch files (creates .bak backups)
    python code_adapter.py --patch /path/to/your/project

    # Dry run - show what would change without modifying
    python code_adapter.py --dry-run /path/to/your/project
"""

import os
import re
import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Dict

# =============================================================================
# Replacement rules for A100 (CUDA) -> Ascend 910B (NPU) migration
# =============================================================================

DEVICE_REPLACEMENTS = [
    # Device placement
    (r'\.to\( *[\'"]cuda[\'"] *\)', '.to("npu")'),
    (r'\.to\( *["\']cuda:\d+["\'] *\)', lambda m: f'.to("npu:{m.group(0).split(":")[1].rstrip("\'\" )")})'),
    (r'\.cuda\(\)', '.npu()'),
    (r'device=[\'"]cuda[\'"]', 'device="npu"'),
    (r'device=[\'"]cuda:\d+[\'"]', lambda m: f'device="npu:{m.group(0).split(":")[1].rstrip("\'\" )")}"'),
    (r'torch\.device\([\'"]cuda[\'"]\)', 'torch.device("npu")'),

    # CUDA-specific API replacements
    (r'torch\.cuda\.is_available\(\)', 'torch.npu.is_available()'),
    (r'torch\.cuda\.device_count\(\)', 'torch.npu.device_count()'),
    (r'torch\.cuda\.current_device\(\)', 'torch.npu.current_device()'),
    (r'torch\.cuda\.set_device\(', 'torch.npu.set_device('),
    (r'torch\.cuda\.empty_cache\(\)', 'torch.npu.empty_cache()'),
    (r'torch\.cuda\.manual_seed\(', 'torch.npu.manual_seed('),
    (r'torch\.cuda\.manual_seed_all\(', 'torch.npu.manual_seed_all('),
    (r'torch\.cuda\.synchronize\(\)', 'torch.npu.synchronize()'),
    (r'torch\.cuda\.Stream\(', 'torch.npu.Stream('),
    (r'torch\.cuda\.Event\(', 'torch.npu.Event('),
    (r'torch\.cuda\.amp', 'torch.npu.amp'),
    (r'torch\.cuda\.distributed', 'torch.distributed'),

    # Tensor device checks
    (r'\.is_cuda', '.is_npu'),

    # Common patterns
    (r'if torch\.cuda\.is_available\(\)', 'if hasattr(torch, "npu") and torch.npu.is_available()'),
]

IMPORT_ADDITIONS = [
    # Add torch_npu import after torch import
    (r'^(import torch\b)', 'import torch\nimport torch_npu'),
]

AMP_REPLACEMENTS = [
    # Apex AMP -> torch.npu.amp
    (r'from apex import amp', '# from apex import amp  # Replaced by torch.npu.amp'),
    (r'torch\.cuda\.amp\.autocast\(\)', 'torch.npu.amp.autocast()'),
    (r'torch\.cuda\.amp\.GradScaler\(\)', 'torch.npu.amp.GradScaler()'),
]

DISTRIBUTED_REPLACEMENTS = [
    # NCCL -> HCCL
    (r'backend=[\'"]nccl[\'"]', 'backend="hccl"'),
    (r'init_method=[\'"]env://[\'"]', 'init_method="env://"'),
    (r'import torch\.cuda\.distributed', 'import torch.distributed'),
]

# Patterns that need manual attention
MANUAL_REVIEW_PATTERNS = [
    (r'cuda', 'Manual review needed: CUDA reference found'),
    (r'GPU', 'Manual review needed: GPU reference found'),
    (r'nvidia', 'Manual review needed: NVIDIA reference found'),
    (r'nccl', 'Manual review needed: NCCL reference found'),
    (r'apex', 'Manual review needed: Apex AMP reference found'),
]


class CodeAdapter:
    """Scans and patches PyTorch code from CUDA to NPU."""

    def __init__(self, target_dir: str, dry_run: bool = False):
        self.target_dir = Path(target_dir)
        self.dry_run = dry_run
        self.scan_results: Dict[str, List[Tuple]] = {}
        self.changes_count = 0

    def find_python_files(self) -> List[Path]:
        """Find all Python files in the target directory."""
        python_files = []
        for root, _, files in os.walk(self.target_dir):
            # Skip virtual environments and common non-project dirs
            skip_dirs = {'venv', '.venv', 'env', '.env', '__pycache__', '.git', 'node_modules'}
            root_parts = set(Path(root).parts)
            if root_parts & skip_dirs:
                continue
            for f in files:
                if f.endswith('.py'):
                    python_files.append(Path(root) / f)
        return python_files

    def scan_file(self, filepath: Path) -> List[Tuple[int, str, str]]:
        """Scan a file for CUDA references that need migration."""
        findings = []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            return findings

        for i, line in enumerate(lines, 1):
            # Check for device references
            for pattern, replacement in DEVICE_REPLACEMENTS:
                if re.search(pattern, line):
                    new_line = re.sub(pattern, replacement, line) if isinstance(replacement, str) else re.sub(pattern, replacement, line)
                    if new_line != line:
                        findings.append((i, line.strip(), new_line.strip(), 'device'))

            # Check for AMP references
            for pattern, replacement in AMP_REPLACEMENTS:
                if re.search(pattern, line):
                    findings.append((i, line.strip(), replacement, 'amp'))

            # Check for distributed references
            for pattern, replacement in DISTRIBUTED_REPLACEMENTS:
                if re.search(pattern, line):
                    findings.append((i, line.strip(), re.sub(pattern, replacement, line).strip(), 'distributed'))

        return findings

    def scan_project(self) -> Dict[str, List[Tuple]]:
        """Scan the entire project for migration needs."""
        python_files = self.find_python_files()
        results = {}

        for filepath in python_files:
            findings = self.scan_file(filepath)
            if findings:
                results[str(filepath)] = findings

        self.scan_results = results
        return results

    def report(self):
        """Print a summary of findings."""
        total_findings = sum(len(v) for v in self.scan_results.values())
        print(f"\n{'=' * 60}")
        print(f"  Migration Scan Report")
        print(f"{'=' * 60}")
        print(f"  Files scanned: {len(self.find_python_files())}")
        print(f"  Files needing changes: {len(self.scan_results)}")
        print(f"  Total references to update: {total_findings}")
        print(f"{'=' * 60}\n")

        for filepath, findings in sorted(self.scan_results.items()):
            print(f"\n--- {filepath} ({len(findings)} changes) ---")
            for line_no, old, new, category in findings:
                print(f"  Line {line_no} [{category}]:")
                print(f"    - {old}")
                print(f"    + {new}")

    def patch_file(self, filepath: Path) -> int:
        """Apply patches to a single file. Returns number of changes made."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
        except Exception:
            return 0

        original_content = content
        changes = 0

        # Apply device replacements
        for pattern, replacement in DEVICE_REPLACEMENTS:
            if isinstance(replacement, str):
                new_content = re.sub(pattern, replacement, content)
            else:
                new_content = re.sub(pattern, replacement, content)
            if new_content != content:
                changes += 1
                content = new_content

        # Apply AMP replacements
        for pattern, replacement in AMP_REPLACEMENTS:
            new_content = re.sub(pattern, replacement, content)
            if new_content != content:
                changes += 1
                content = new_content

        # Apply distributed replacements
        for pattern, replacement in DISTRIBUTED_REPLACEMENTS:
            new_content = re.sub(pattern, replacement, content)
            if new_content != content:
                changes += 1
                content = new_content

        # Add torch_npu import if torch is imported and torch_npu is not
        if 'import torch' in content and 'import torch_npu' not in content:
            content = re.sub(
                r'^(import torch\b)',
                'import torch\nimport torch_npu',
                content,
                count=1,
                flags=re.MULTILINE,
            )
            changes += 1

        if content != original_content:
            if not self.dry_run:
                # Create backup
                backup_path = filepath.with_suffix('.py.bak')
                if not backup_path.exists():
                    with open(backup_path, 'w', encoding='utf-8') as f:
                        f.write(original_content)

                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)

                print(f"  [PATCHED] {filepath} ({changes} changes)")
            else:
                print(f"  [DRY RUN] Would patch {filepath} ({changes} changes)")

        return changes

    def patch_project(self) -> int:
        """Patch all files in the project. Returns total changes."""
        python_files = self.find_python_files()
        total_changes = 0

        print(f"\nPatching {len(python_files)} Python files...")
        print(f"{'=' * 60}\n")

        for filepath in python_files:
            changes = self.patch_file(filepath)
            total_changes += changes

        print(f"\n{'=' * 60}")
        print(f"  Total changes: {total_changes}")
        if self.dry_run:
            print(f"  (Dry run - no files were modified)")
        else:
            print(f"  Backup files created with .bak extension")
        print(f"{'=' * 60}\n")

        return total_changes


def main():
    parser = argparse.ArgumentParser(
        description='A100 (CUDA) -> Ascend 910B (NPU) Code Adapter'
    )
    parser.add_argument(
        'target',
        help='Target directory or file to scan/patch',
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--scan', action='store_true',
        help='Only scan and report changes needed'
    )
    group.add_argument(
        '--patch', action='store_true',
        help='Automatically patch files (creates .bak backups)'
    )
    group.add_argument(
        '--dry-run', action='store_true',
        help='Show what would change without modifying files'
    )

    args = parser.parse_args()

    if not os.path.exists(args.target):
        print(f"Error: {args.target} not found")
        sys.exit(1)

    adapter = CodeAdapter(args.target, dry_run=args.dry_run or args.scan)

    if args.scan or args.dry_run:
        results = adapter.scan_project()
        adapter.report()
    elif args.patch:
        changes = adapter.patch_project()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
