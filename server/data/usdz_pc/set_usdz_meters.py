#!/usr/bin/env python3
"""
USDZ metersPerUnit 設定変更ツール

metersPerUnitを0.001(mm)から1.0(m)に変更します。
座標値は変更しません。

例: 座標値0.005が、0.005mm→0.005m(=5mm)として正しく表示されるようになります

使用例:
    python set_usdz_meters.py input.usdz -o output.usdz
    python set_usdz_meters.py input.usdz --inplace
"""

import os
import sys
import argparse
import tempfile
import shutil
import zipfile
from pathlib import Path

from pxr import Usd, UsdGeom


def extract_usdz(usdz_path: str, extract_dir: str) -> str:
    """USDZファイルを展開"""
    with zipfile.ZipFile(usdz_path, 'r') as zf:
        zf.extractall(extract_dir)
    
    for file in os.listdir(extract_dir):
        if file.endswith('.usdc') or file.endswith('.usda'):
            return os.path.join(extract_dir, file)
    
    raise ValueError(f"No USD file found in {usdz_path}")


def convert_usdz_to_meters(input_path: str, output_path: str, verbose: bool = True):
    """metersPerUnitを1.0に変更"""
    temp_dir = tempfile.mkdtemp()
    
    try:
        if verbose:
            print(f"Processing: {Path(input_path).name}")
        
        extract_dir = os.path.join(temp_dir, "extract")
        os.makedirs(extract_dir, exist_ok=True)
        usd_file = extract_usdz(input_path, extract_dir)
        
        stage = Usd.Stage.Open(usd_file)
        
        current_mpu = UsdGeom.GetStageMetersPerUnit(stage)
        if verbose:
            print(f"  Current metersPerUnit: {current_mpu}")
        
        # m単位に設定
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        
        if verbose:
            print(f"  New metersPerUnit: 1.0")
        
        stage.GetRootLayer().Save()
        
        # USDCに変換
        usdc_path = os.path.join(temp_dir, "converted.usdc")
        stage_usdc = Usd.Stage.Open(usd_file)
        stage_usdc.GetRootLayer().Export(usdc_path)
        
        # USDZ再パッケージ
        with zipfile.ZipFile(output_path, 'w', compression=zipfile.ZIP_STORED) as zf:
            zf.write(usdc_path, arcname=Path(usdc_path).name)
        
        if verbose:
            print(f"  ✓ Saved: {output_path}")
        
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Change USDZ metersPerUnit to 1.0 (meter scale)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("input", nargs="+", help="Input USDZ file(s)")
    parser.add_argument("--output", "-o", help="Output file or directory")
    parser.add_argument("--inplace", action="store_true", help="Overwrite input")
    parser.add_argument("--quiet", "-q", action="store_true", help="Quiet mode")
    
    args = parser.parse_args()
    
    if not args.inplace and not args.output:
        parser.error("Either --output or --inplace required")
    
    if args.inplace and args.output:
        parser.error("Cannot use both --output and --inplace")
    
    for input_file in args.input:
        if not os.path.exists(input_file):
            print(f"Error: {input_file} not found")
            sys.exit(1)
    
    verbose = not args.quiet
    
    for input_file in args.input:
        if args.inplace:
            temp_output = input_file + ".tmp"
            convert_usdz_to_meters(input_file, temp_output, verbose)
            shutil.move(temp_output, input_file)
        else:
            if len(args.input) == 1:
                output_file = args.output if not os.path.isdir(args.output) else os.path.join(args.output, Path(input_file).name)
            else:
                os.makedirs(args.output, exist_ok=True)
                output_file = os.path.join(args.output, Path(input_file).name)
            
            convert_usdz_to_meters(input_file, output_file, verbose)
    
    if verbose:
        print("\n✓ Complete!")


if __name__ == "__main__":
    main()
