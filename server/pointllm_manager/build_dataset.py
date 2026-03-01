"""
build_dataset.py - PointLLM Fine-tuning Dataset Builder
AUTOMATED BATCH VERSION: Processes ALL datasets sequentially

Changes:
1. Iterates through all defined TARGET_DATASETS
2. Dynamically generates input paths based on dataset name
3. Skips missing input directories gracefully

Author: Dataset Pipeline for PointLLM Fine-tuning
"""

import json
import os
import sys
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import argparse

from .modules.anatomical_naming import AnatomicalNaming, get_display_name
from .modules.rule_based_prompt_generator import RuleBasedPromptGenerator
from .modules.teeth_only_prompt_generator import TeethOnlyPromptGenerator
from .modules.simple_pointing_prompt_generator import SimplePointingPromptGenerator

# ============================================================================
# BATCH CONFIGURATION
# ============================================================================
# List of all dataset names to process
TARGET_DATASETS = [
    "dental_model",
    "Upper_Teeth",
    "Lower_Teeth",
    "Skull_and_UTeeth",
    "Mandible",
    "Mandible_and_LTeeth",
    "U_L_and_Mandible",
    "U_and_LTeeth"
]

# Base directory for inputs (The script will append /{NAME}_outputs automatically)
BASE_INPUT_ROOT = "/Volumes/USB-KIM/ConvertPCFrpmARData"

# PointLLM Special Tokens Configuration
POINT_TOKEN_CONFIG = {
    'point_patch_token': '<point_patch>',
    'point_start_token': '<point_start>',
    'point_end_token': '<point_end>',
    'point_token_len': 513,
    'mm_use_point_start_end': True,
}


def format_point_token_sequence() -> str:
    """Format the point token sequence according to PointLLM's specification"""
    point_patch_token = POINT_TOKEN_CONFIG['point_patch_token']
    point_start_token = POINT_TOKEN_CONFIG['point_start_token']
    point_end_token = POINT_TOKEN_CONFIG['point_end_token']
    point_token_len = POINT_TOKEN_CONFIG['point_token_len']
    
    if POINT_TOKEN_CONFIG['mm_use_point_start_end']:
        token_sequence = (
            point_start_token +
            (point_patch_token * point_token_len) +
            point_end_token
        )
    else:
        token_sequence = point_patch_token * point_token_len
    
    return token_sequence


@dataclass
class RegionOfInterest:
    """Represents a region of interest from hand tracking analysis."""
    rank: int
    subclass_id: int
    subclass_name: str
    class_id: int
    class_name: str
    count: int
    ratio: float
    _naming: Optional[AnatomicalNaming] = field(default=None, repr=False)
    
    def set_naming(self, naming: AnatomicalNaming):
        self._naming = naming
    
    @property
    def anatomical_class_name(self) -> str:
        if self._naming:
            return self._naming.get_class_display_name(self.class_id)
        return get_display_name(self.class_name)
    
    @property
    def anatomical_subclass_name(self) -> str:
        if self._naming:
            return self._naming.get_subclass_display_name(self.subclass_id)
        return get_display_name(self.subclass_name)


@dataclass
class HandAnalysisResult:
    """Represents the hand tracking analysis for a single frame."""
    tracking_source: str
    total_hand_points: int
    total_pc_points: int
    closest_distance: float
    regions_of_interest: List[RegionOfInterest]
    
    @classmethod
    def from_json(cls, json_data: dict, naming: Optional[AnatomicalNaming] = None) -> "HandAnalysisResult":
        closest_interaction = json_data.get("closest_interaction", {})
        closest_distance = closest_interaction.get("distance", 0.0)
        
        best_hand_point = None
        min_distance = float('inf')
        
        for hp in json_data.get("hand_point_analysis", []):
            dist = hp.get("distance_to_nearest", float('inf'))
            stats = hp.get("neighbor_statistics", {}).get("statistics", [])
            if stats and dist < min_distance:
                min_distance = dist
                best_hand_point = hp
        
        regions = []
        if best_hand_point:
            for stat in best_hand_point["neighbor_statistics"]["statistics"]:
                roi = RegionOfInterest(
                    rank=stat["rank"],
                    subclass_id=stat["subclass_id"],
                    subclass_name=stat["subclass_name"],
                    class_id=stat["class_id"],
                    class_name=stat["class_name"],
                    count=stat["count"],
                    ratio=stat["ratio"],
                )
                if naming:
                    roi.set_naming(naming)
                regions.append(roi)
        
        return cls(
            tracking_source=json_data.get("tracking_data_source", ""),
            total_hand_points=json_data.get("total_hand_points", 0),
            total_pc_points=json_data.get("total_pc_points", 0),
            closest_distance=closest_distance,
            regions_of_interest=regions,
        )


@dataclass
class DatasetConfig:
    """Configuration for dataset building."""
    data_root: str
    output_dir: str
    class_subclass_info_path: Optional[str] = None
    name: str = "dental_model"
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    num_prompt_variations: int = 6
    num_simple_pointing: int = 2
    seed: int = 42
    num_points: int = 8192


class PointCloudProcessor:
    """Processes point cloud data for PointLLM format."""
    
    @staticmethod
    def load_colorized_pointcloud(npy_path: str) -> np.ndarray:
        data = np.load(npy_path)
        if data.shape[1] != 8:
            raise ValueError(f"Expected 8 columns, got {data.shape[1]}")
        
        xyz = data[:, :3].copy()
        rgb = data[:, 3:6].copy()
        
        centroid = np.mean(xyz, axis=0)
        xyz = xyz - centroid
        max_dist = np.max(np.sqrt(np.sum(xyz ** 2, axis=1)))
        if max_dist > 0:
            xyz = xyz / max_dist
        
        if rgb.max() > 1.0:
            rgb = rgb / 255.0
        rgb = np.clip(rgb, 0.0, 1.0)
        
        result = np.concatenate([xyz, rgb], axis=1).astype(np.float32)
        return result
    
    @staticmethod
    def save_pointcloud(data: np.ndarray, output_path: str):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, data.astype(np.float32))


class DatasetBuilder:
    """Main class for building the PointLLM fine-tuning dataset."""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.prompt_generator = RuleBasedPromptGenerator(seed=config.seed)
        self.teeth_only_generator = TeethOnlyPromptGenerator(seed=config.seed)
        self.simple_pointing_generator = SimplePointingPromptGenerator(seed=config.seed)
        self.pc_processor = PointCloudProcessor()
        self.naming: Optional[AnatomicalNaming] = None
        
        random.seed(config.seed)
        np.random.seed(config.seed)
        
        self._load_anatomical_naming()
    
    def _is_teeth_only_dataset(self) -> bool:
        teeth_only_names = ["Upper_Teeth", "Lower_Teeth", "U_and_L_Teeth"]
        return self.config.name in teeth_only_names
    
    def _load_anatomical_naming(self):
        if self.config.class_subclass_info_path:
            if Path(self.config.class_subclass_info_path).exists():
                self.naming = AnatomicalNaming.from_json(self.config.class_subclass_info_path)
                print(f"Loaded naming from: {self.config.class_subclass_info_path}")
                return
        
        tracking_folders = self._find_tracking_folders()
        if tracking_folders:
            for folder in tracking_folders:
                json_path = folder / "class_subclass_info.json"
                if json_path.exists():
                    self.naming = AnatomicalNaming.from_json(str(json_path))
                    print(f"Loaded naming from: {json_path}")
                    return
        
        try:
            self.naming = AnatomicalNaming.load_default()
            print("Loaded naming from default search paths")
        except FileNotFoundError:
            print("⚠️ Warning: Could not load class_subclass_info.json")
    
    def _find_tracking_folders(self) -> List[Path]:
        data_root = Path(self.config.data_root)
        if not data_root.exists():
            return []
        folders = sorted(data_root.glob("TrackingData_*"))
        return [f for f in folders if f.is_dir()]
    
    def _get_colorization_folders(self, tracking_folder: Path) -> Dict[str, Path]:
        colorizations = {}
        for name in ["colorized_hover", "colorized_contrast", 
                     "colorized_fixed_blue", "colorized_fixed_red", "colorized_fixed_black"]:
            folder = tracking_folder / name
            if folder.exists():
                effect_type = name.replace("colorized_", "").replace("fixed_", "")
                colorizations[effect_type] = folder
        return colorizations
    
    def _load_hand_analysis(self, tracking_folder: Path) -> Optional[HandAnalysisResult]:
        json_path = tracking_folder / "hand_analysis.json"
        if not json_path.exists():
            return None
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return HandAnalysisResult.from_json(data, naming=self.naming)
    
    def _generate_sample_id(self, tracking_name: str, effect_type: str, variation: Any) -> str:
        base = f"{tracking_name}_{effect_type}_v{variation}"
        return hashlib.md5(base.encode()).hexdigest()[:12]
    
    def build(self) -> Dict[str, Any]:
        print(f"\nNAME Configuration: {self.config.name}")
        
        if self.naming:
            class_ids = self.naming.get_class_ids_for_name(self.config.name)
            subclass_ids = self.naming.get_subclass_ids_for_name(self.config.name)
            print(f"  NAME Filtering: Included {len(subclass_ids)} subclasses")
        
        tracking_folders = self._find_tracking_folders()
        if not tracking_folders:
            print(f"  ⚠️ No TrackingData folders found in {self.config.data_root}")
            return {"total_samples": 0}

        print(f"  Found {len(tracking_folders)} tracking folders")
        
        if self.naming:
            all_candidate_names = self.naming.get_filtered_subclass_display_names(self.config.name)
        else:
            all_candidate_names = ["Unknown Region"]
        
        all_samples = []
        
        for tracking_folder in tracking_folders:
            tracking_name = tracking_folder.name
            # print(f"  Processing: {tracking_name}") # Reduced verbosity for batch
            
            hand_analysis = self._load_hand_analysis(tracking_folder)
            if not hand_analysis or not hand_analysis.regions_of_interest:
                continue
            
            primary_region = hand_analysis.regions_of_interest[0]
            secondary_region = (hand_analysis.regions_of_interest[1] 
                              if len(hand_analysis.regions_of_interest) > 1 else None)
            closest_distance = hand_analysis.closest_distance
            
            colorizations = self._get_colorization_folders(tracking_folder)
            
            for effect_type, effect_folder in colorizations.items():
                npy_path = effect_folder / "pointcloud_colorized.npy"
                if not npy_path.exists():
                    continue
                
                try:
                    pc_data = self.pc_processor.load_colorized_pointcloud(str(npy_path))
                except Exception as e:
                    print(f"    ❌ Error loading point cloud: {e}")
                    continue
                
                # --- Standard/Teeth Prompt Generation ---
                for var_idx in range(self.config.num_prompt_variations):
                    sample_id = self._generate_sample_id(tracking_name, effect_type, var_idx)
                    distance_mm = closest_distance * 1000
                    
                    if self._is_teeth_only_dataset():
                        if self.config.name == "U_and_L_Teeth":
                            if var_idx < 3:
                                question, answer = self.teeth_only_generator.generate_version1_ambiguous(
                                    effect_type, primary_region, all_candidate_names, distance_mm, self.config.name
                                )
                            else:
                                question, answer = self.teeth_only_generator.generate_ul_teeth_protrusion_variant(
                                    effect_type, primary_region, all_candidate_names, distance_mm
                                )
                        else:
                            if var_idx < 3:
                                question, answer = self.teeth_only_generator.generate_version1_ambiguous(
                                    effect_type, primary_region, all_candidate_names, distance_mm, self.config.name
                                )
                            else:
                                question, answer = self.teeth_only_generator.generate_version2_class_specified(
                                    effect_type, primary_region, all_candidate_names, distance_mm, self.config.name
                                )
                    else:
                        question, answer = self.prompt_generator.generate(
                            effect_type, primary_region, secondary_region,
                            all_candidate_names, distance_mm, var_idx
                        )
                    
                    point_token_sequence = format_point_token_sequence()
                    question = f"{point_token_sequence}\n{question}"
                    
                    prompt_version = "standard"
                    if self._is_teeth_only_dataset():
                        if self.config.name == "U_and_L_Teeth":
                            prompt_version = "v1_ambiguous" if var_idx < 3 else "ul_protrusion"
                        else:
                            prompt_version = "v1_ambiguous" if var_idx < 3 else "v2_class_specified"
                    
                    pc_output_path = f"point_clouds/{tracking_name}/{self.config.name}_{effect_type}_v{var_idx}.npy"
                    full_pc_path = Path(self.config.output_dir) / pc_output_path
                    self.pc_processor.save_pointcloud(pc_data, str(full_pc_path))
                    
                    sample = {
                        "id": sample_id,
                        "point_cloud": pc_output_path,
                        "conversations": [{"role": "user", "content": question}, {"role": "assistant", "content": answer}],
                        "metadata": {
                            "tracking_source": tracking_name,
                            "colorization_effect": effect_type,
                            "variation_index": var_idx,
                            "prompt_version": prompt_version,
                            "hand_distance_mm": float(closest_distance * 1000),
                            "primary_region": {
                                "id": primary_region.subclass_id,
                                "raw_name": primary_region.subclass_name,
                                "display_name": primary_region.anatomical_subclass_name,
                                "confidence": primary_region.ratio
                            },
                        }
                    }
                    all_samples.append(sample)
                
                # --- Simple Pointing Generation ---
                for simple_idx in range(self.config.num_simple_pointing):
                    sample_id = self._generate_sample_id(tracking_name, effect_type, f"simple_{simple_idx}")
                    include_class_info = (simple_idx % 2 == 1)
                    
                    question, answer = self.simple_pointing_generator.generate(
                        primary_region=primary_region,
                        include_class_info=include_class_info,
                        variation_index=simple_idx
                    )
                    
                    point_token_sequence = format_point_token_sequence()
                    question = f"{point_token_sequence}\n{question}"
                    
                    pc_output_path = f"point_clouds/{tracking_name}/{self.config.name}_{effect_type}_simple_{simple_idx}.npy"
                    full_pc_path = Path(self.config.output_dir) / pc_output_path
                    self.pc_processor.save_pointcloud(pc_data, str(full_pc_path))
                    
                    sample = {
                        "id": sample_id,
                        "point_cloud": pc_output_path,
                        "conversations": [{"role": "user", "content": question}, {"role": "assistant", "content": answer}],
                        "metadata": {
                            "tracking_source": tracking_name,
                            "colorization_effect": effect_type,
                            "variation_index": simple_idx,
                            "prompt_version": "simple_pointing",
                            "primary_region": {"id": primary_region.subclass_id}
                        }
                    }
                    all_samples.append(sample)
        
        if not all_samples:
            print("  ⚠️ No samples generated.")
            return {"total_samples": 0}

        # Split and Save
        random.shuffle(all_samples)
        n_total = len(all_samples)
        n_train = int(n_total * self.config.train_ratio)
        n_val = int(n_total * self.config.val_ratio)
        
        train_samples = all_samples[:n_train]
        val_samples = all_samples[n_train:n_train + n_val]
        test_samples = all_samples[n_train + n_val:]
        
        output_dir = Path(self.config.output_dir)
        annotations_dir = output_dir / "annotations"
        annotations_dir.mkdir(parents=True, exist_ok=True)
        
        dataset_info = {
            "version": "2.2-batch",
            "name_configuration": self.config.name,
            "created_at": datetime.now().isoformat(),
            "config": {"num_prompt_variations": self.config.num_prompt_variations}
        }
        
        for split_name, split_data in [("train", train_samples), ("val", val_samples), ("test", test_samples)]:
            output_data = {**dataset_info, "split": split_name, "data": split_data}
            output_path = annotations_dir / f"{self.config.name}_{split_name}.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            print(f"  ✅ Saved: {output_path} ({len(split_data)} samples)")
        
        return {
            "total_samples": len(all_samples),
            "output_dir": str(output_dir),
        }


def main():
    """Main entry point for AUTOMATED BATCH PROCESSING."""
    
    parser = argparse.ArgumentParser(description="Automated Batch Build for PointLLM Dataset")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./ar_usage_dataset",
        help="Output directory for the dataset"
    )
    parser.add_argument(
        "--base_input_root",
        type=str,
        default=BASE_INPUT_ROOT,
        help="Base directory containing {NAME}_outputs folders"
    )
    
    # Optional arguments to override defaults
    parser.add_argument("--num_variations", type=int, default=1)
    parser.add_argument("--num_simple_pointing", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("STARTING BATCH DATASET GENERATION")
    print(f"Target Datasets: {len(TARGET_DATASETS)}")
    print(f"Base Input Root: {args.base_input_root}")
    print("=" * 60)
    
    total_samples_all = 0
    
    for name in TARGET_DATASETS:
        print(f"\n>>> Processing Dataset Type: {name}")
        
        # Dynamically construct the specific data root for this NAME
        # e.g., /Volumes/.../dental_model_outputs
        specific_data_root = os.path.join(args.base_input_root, f"{name}_outputs")
        
        # Check if directory exists
        if not os.path.exists(specific_data_root):
            print(f"❌ SKIPPING: Directory not found -> {specific_data_root}")
            continue
            
        print(f"  Input Source: {specific_data_root}")
        
        # Create config for this specific dataset
        config = DatasetConfig(
            data_root=specific_data_root,
            output_dir=args.output_dir,
            name=name,
            num_prompt_variations=args.num_variations,
            num_simple_pointing=args.num_simple_pointing,
            seed=args.seed,
        )
        
        # Build dataset
        builder = DatasetBuilder(config)
        results = builder.build()
        total_samples_all += results.get("total_samples", 0)
        
    print("\n" + "=" * 60)
    print("BATCH PROCESSING COMPLETE!")
    print("=" * 60)
    print(f"Grand Total Samples Generated: {total_samples_all}")
    print(f"Output Directory: {args.output_dir}")


if __name__ == "__main__":
    main()