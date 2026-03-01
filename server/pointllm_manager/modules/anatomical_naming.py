"""
anatomical_naming.py - Dynamic Anatomical Naming Loader with NAME-based Filtering

This module:
1. Loads raw label names from class_subclass_info.json
2. Provides human-readable anatomical names for prompts and answers
3. Filters subclasses based on NAME configuration (e.g., "Upper_Teeth", "Mandible")
4. Replaces hardcoded SUBCLASS_NAME_MAP and CLASS_NAME_MAP

Usage:
    from anatomical_naming import AnatomicalNaming
    
    # Load from a specific class_subclass_info.json
    naming = AnatomicalNaming.from_json("/path/to/class_subclass_info.json")
    
    # Get all subclasses for a specific NAME
    upper_teeth_subclasses = naming.get_subclasses_for_name("Upper_Teeth")
    
    # Get display names for filtered subclasses
    display_names = naming.get_filtered_subclass_display_names("Upper_Teeth")
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Set

# ==============================================================================
# NAME Configuration Mappings
# ==============================================================================

# NAME -> List of Class Names that should be included
NAME_TO_CLASSES = {
    "dental_model": ["Maxilla & Upper Skull", "Mandible", "Upper Teeth", "Lower Teeth"],
    "Upper_Teeth": ["Upper Teeth"],
    "Lower_Teeth": ["Lower Teeth"],
    "Skull_and_Upper_Teeth": ["Maxilla & Upper Skull", "Upper Teeth"],
    "Skull_and_UTeeth": ["Maxilla & Upper Skull", "Upper Teeth"],  # Alias
    "Mandible": ["Mandible"],
    "Mandible_and_LTeeth": ["Mandible", "Lower Teeth"],
    "U_L_and_Mandible": ["Upper Teeth", "Lower Teeth", "Mandible"],
    "U_and_LTeeth": ["Upper Teeth", "Lower Teeth"],
}

# ==============================================================================
# Human-readable Name Mappings (Raw Label -> Display Name)
# ==============================================================================

# Standard class name display mappings (for pretty formatting)
# These are ONLY used if a custom display name is desired
# Otherwise, the raw name from JSON is used directly
STANDARD_CLASS_DISPLAY_NAMES = {
    "Maxilla & Upper Skull": "Maxilla & Upper Skull",
    "Maxilla_and_Upper_Skull": "Maxilla & Upper Skull",
    "Mandible": "Mandible",
    "Upper Teeth": "Upper Teeth",
    "Upper_Teeth": "Upper Teeth",
    "Lower Teeth": "Lower Teeth",
    "Lower_Teeth": "Lower Teeth",
}

# Raw Label Name -> Human-Readable Display Name Mapping
SUBCLASS_DISPLAY_NAME_MAP = {
    # =========================================================================
    # Maxilla & Upper Skull Regions (Class 1)
    # =========================================================================
    "Maxilla_and_Upper_Skull_RPosteriorSuperior": 
        "Right Upper Posterior of Maxilla and Upper Skull",
    "Maxilla_and_Upper_Skull_LPosteriorSuperior": 
        "Left Upper Posterior of Maxilla and Upper Skull",
    "Maxilla_and_Upper_Skull_RAnteriorSuperior": 
        "Right Upper Anterior of Maxilla and Upper Skull",
    "Maxilla_and_Upper_Skull_LAnteriorSuperior": 
        "Left Upper Anterior of Maxilla and Upper Skull",
    "Maxilla_and_Upper_Skull_RPosteriorInferior": 
        "Right Lower Posterior of Maxilla and Upper Skull",
    "Maxilla_and_Upper_Skull_LPosteriorInferior": 
        "Left Lower Posterior of Maxilla and Upper Skull",
    "Maxilla_and_Upper_Skull_RAnteriorInferior": 
        "Right Cheekbone (Zygomatic) Region",
    "Maxilla_and_Upper_Skull_LAnteriorInferior": 
        "Left Cheekbone (Zygomatic) Region",
    
    # =========================================================================
    # Mandible Regions (Class 2)
    # =========================================================================
    "Mandible_Right_Anterior_Inferior_Lingual": 
        "Right Anterior Inferior Lingual Surface of Mandible",
    "Mandible_Right_Anterior_Inferior_Buccal": 
        "Right Anterior Inferior Buccal Surface of Mandible",
    "Mandible_Right_Posterior_Inferior_Lingual": 
        "Right Posterior Inferior Lingual Surface of Mandible",
    "Mandible_Right_Posterior_Inferior_Buccal": 
        "Right Posterior Inferior Buccal Surface of Mandible",
    "Mandible_Right_Posterior_Superior_Lingual": 
        "Right Posterior Superior Lingual Surface of Mandible",
    "Mandible_Right_Posterior_Superior_Buccal": 
        "Right Posterior Superior Buccal Surface of Mandible",
    "Mandible_Left_Anterior_Inferior_Lingual": 
        "Left Anterior Inferior Lingual Surface of Mandible",
    "Mandible_Left_Anterior_Inferior_Buccal": 
        "Left Anterior Inferior Buccal Surface of Mandible",
    "Mandible_Left_Posterior_Inferior_Buccal": 
        "Left Posterior Inferior Buccal Surface of Mandible",
    "Mandible_Left_Posterior_Inferior_Lingual": 
        "Left Posterior Inferior Lingual Surface of Mandible",
    "Mandible_Left_Posterior_Superior_Buccal": 
        "Left Posterior Superior Buccal Surface of Mandible",
    "Mandible_Left_Posterior_Superior_Lingual": 
        "Left Posterior Superior Lingual Surface of Mandible",
    
    # =========================================================================
    # Upper Teeth (Class 3) - FDI Quadrants 1 (UR) and 2 (UL)
    # =========================================================================
    # Upper Right (FDI 11-18)
    "FDI_11_UR1_central_incisor": 
        "FDI 11: Upper right 1st tooth from midline (Central incisor)",
    "FDI_12_UR2_lateral_incisor": 
        "FDI 12: Upper right 2nd tooth from midline (Lateral incisor)",
    "FDI_13_UR3_canine": 
        "FDI 13: Upper right 3rd tooth from midline (Canine)",
    "FDI_14_UR4_first_premolar": 
        "FDI 14: Upper right 4th tooth from midline (1st premolar)",
    "FDI_15_UR5_second_premolar": 
        "FDI 15: Upper right 5th tooth from midline (2nd premolar)",
    "FDI_16_UR6_first_molar": 
        "FDI 16: Upper right 6th tooth from midline (1st molar)",
    "FDI_17_UR7_second_molar": 
        "FDI 17: Upper right 7th tooth from midline (2nd molar)",
    "FDI_18_UR8_third_molar": 
        "FDI 18: Upper right 8th tooth from midline (3rd molar / Wisdom tooth)",
    
    # Upper Left (FDI 21-28)
    "FDI_21_UL1_central_incisor": 
        "FDI 21: Upper left 1st tooth from midline (Central incisor)",
    "FDI_22_UL2_lateral_incisor": 
        "FDI 22: Upper left 2nd tooth from midline (Lateral incisor)",
    "FDI_23_UL3_canine": 
        "FDI 23: Upper left 3rd tooth from midline (Canine)",
    "FDI_24_UL4_first_premolar": 
        "FDI 24: Upper left 4th tooth from midline (1st premolar)",
    "FDI_25_UL5_second_premolar": 
        "FDI 25: Upper left 5th tooth from midline (2nd premolar)",
    "FDI_26_UL6_first_molar": 
        "FDI 26: Upper left 6th tooth from midline (1st molar)",
    "FDI_27_UL7_second_molar": 
        "FDI 27: Upper left 7th tooth from midline (2nd molar)",
    "FDI_28_UL8_third_molar": 
        "FDI 28: Upper left 8th tooth from midline (3rd molar / Wisdom tooth)",
    
    # =========================================================================
    # Lower Teeth (Class 4) - FDI Quadrants 3 (LL) and 4 (LR)
    # =========================================================================
    # Lower Left (FDI 31-38)
    "FDI_31_LL1_central_incisor": 
        "FDI 31: Lower left 1st tooth from midline (Central incisor)",
    "FDI_32_LL2_lateral_incisor": 
        "FDI 32: Lower left 2nd tooth from midline (Lateral incisor)",
    "FDI_33_LL3_canine": 
        "FDI 33: Lower left 3rd tooth from midline (Canine)",
    "FDI_34_LL4_first_premolar": 
        "FDI 34: Lower left 4th tooth from midline (1st premolar)",
    "FDI_35_LL5_second_premolar": 
        "FDI 35: Lower left 5th tooth from midline (2nd premolar)",
    "FDI_36_LL6_first_molar": 
        "FDI 36: Lower left 6th tooth from midline (1st molar)",
    "FDI_37_LL7_second_molar": 
        "FDI 37: Lower left 7th tooth from midline (2nd molar)",
    "FDI_38_LL8_third_molar": 
        "FDI 38: Lower left 8th tooth from midline (3rd molar / Wisdom tooth)",
    
    # Lower Right (FDI 41-48)
    "FDI_41_LR1_central_incisor": 
        "FDI 41: Lower right 1st tooth from midline (Central incisor)",
    "FDI_42_LR2_lateral_incisor": 
        "FDI 42: Lower right 2nd tooth from midline (Lateral incisor)",
    "FDI_43_LR3_canine": 
        "FDI 43: Lower right 3rd tooth from midline (Canine)",
    "FDI_44_LR4_first_premolar": 
        "FDI 44: Lower right 4th tooth from midline (1st premolar)",
    "FDI_45_LR5_second_premolar": 
        "FDI 45: Lower right 5th tooth from midline (2nd premolar)",
    "FDI_46_LR6_first_molar": 
        "FDI 46: Lower right 6th tooth from midline (1st molar)",
    "FDI_47_LR7_second_molar": 
        "FDI 47: Lower right 7th tooth from midline (2nd molar)",
    "FDI_48_LR8_third_molar": 
        "FDI 48: Lower right 8th tooth from midline (3rd molar / Wisdom tooth)",
}


# ==============================================================================
# AnatomicalNaming Class with NAME-based Filtering
# ==============================================================================

class AnatomicalNaming:
    """
    Dynamic anatomical naming loader with NAME-based filtering.
    
    Loads raw label names from class_subclass_info.json and provides
    human-readable display names for prompts and answers. Can filter
    subclasses based on NAME configuration (e.g., "Upper_Teeth").
    """
    
    def __init__(
        self,
        classes: Dict[int, Dict[str, Any]],
        subclasses: Dict[int, Dict[str, Any]],
        source_path: Optional[str] = None
    ):
        """
        Initialize AnatomicalNaming.
        
        Args:
            classes: Dictionary mapping class_id (int) to class info
            subclasses: Dictionary mapping subclass_id (int) to subclass info
            source_path: Path to the source JSON file (for reference)
        """
        self.classes = classes
        self.subclasses = subclasses
        self.source_path = source_path
        
        # Build lookup tables
        self._build_lookup_tables()
    
    def _build_lookup_tables(self):
        """Build internal lookup tables for fast access."""
        # Raw name -> subclass_id
        self._raw_name_to_id: Dict[str, int] = {}
        # subclass_id -> raw name
        self._id_to_raw_name: Dict[int, str] = {}
        # subclass_id -> display name
        self._id_to_display_name: Dict[int, str] = {}
        # class_id -> class display name
        self._class_id_to_display_name: Dict[int, str] = {}
        # class_id -> class raw name (from JSON)
        self._class_id_to_raw_name: Dict[int, str] = {}
        # class raw name -> class_id
        self._class_raw_name_to_id: Dict[str, int] = {}
        # subclass_id -> class_id
        self._subclass_to_class: Dict[int, int] = {}
        
        # Build class lookup FROM JSON DATA (not hardcoded)
        for class_id, class_info in self.classes.items():
            raw_name = class_info.get("name", f"Class {class_id}")
            self._class_id_to_raw_name[class_id] = raw_name
            self._class_raw_name_to_id[raw_name] = class_id
            
            # Use standard display name if available, otherwise use raw name
            display_name = STANDARD_CLASS_DISPLAY_NAMES.get(raw_name, raw_name)
            self._class_id_to_display_name[class_id] = display_name
        
        # Build subclass lookup
        for subclass_id, subclass_info in self.subclasses.items():
            raw_name = subclass_info.get("name", f"Subclass {subclass_id}")
            
            self._raw_name_to_id[raw_name] = subclass_id
            self._id_to_raw_name[subclass_id] = raw_name
            
            # Get display name from mapping, or generate one
            display_name = SUBCLASS_DISPLAY_NAME_MAP.get(
                raw_name, 
                self._generate_display_name(raw_name)
            )
            self._id_to_display_name[subclass_id] = display_name
            
            # Infer class from raw name
            class_id = self._infer_class_from_name(raw_name)
            self._subclass_to_class[subclass_id] = class_id
    
    def _infer_class_from_name(self, raw_name: str) -> int:
        """
        Infer class ID from raw subclass name.
        Uses actual class names from JSON instead of hardcoded assumptions.
        """
        name_lower = raw_name.lower()
        
        # Try to match with actual class names from JSON
        for class_id, class_raw_name in self._class_id_to_raw_name.items():
            class_name_lower = class_raw_name.lower()
            
            # Check if subclass name contains the class name
            # e.g., "Mandible_Right_..." contains "Mandible"
            if class_name_lower in name_lower:
                return class_id
        
        # Fallback: FDI notation inference
        if raw_name.startswith("FDI_"):
            # Extract FDI number to determine upper/lower
            import re
            match = re.search(r'FDI_(\d{2})', raw_name)
            if match:
                fdi_num = int(match.group(1))
                # FDI 11-28 are upper, 31-48 are lower
                if 11 <= fdi_num <= 28:
                    # Look for "Upper" or "Upper_Teeth" in class names
                    for class_id, class_raw_name in self._class_id_to_raw_name.items():
                        if "upper" in class_raw_name.lower() and "teeth" in class_raw_name.lower():
                            return class_id
                elif 31 <= fdi_num <= 48:
                    # Look for "Lower" or "Lower_Teeth" in class names
                    for class_id, class_raw_name in self._class_id_to_raw_name.items():
                        if "lower" in class_raw_name.lower() and "teeth" in class_raw_name.lower():
                            return class_id
        
        return 0  # Unknown
    
    def _generate_display_name(self, raw_name: str) -> str:
        """
        Generate a display name from raw name if not in mapping.
        This is a fallback for unknown labels.
        """
        # Try FDI notation first
        if raw_name.startswith("FDI_"):
            return self._generate_fdi_display_name(raw_name)
        
        # Otherwise, just clean up the name
        display = raw_name.replace("_", " ")
        return display
    
    def _generate_fdi_display_name(self, raw_name: str) -> str:
        """Generate display name for FDI tooth notation."""
        # Pattern: FDI_XX_YYYN_description
        match = re.match(r'FDI_(\d{2})_([UL][LR])(\d)_(.+)', raw_name)
        if not match:
            return raw_name.replace("_", " ")
        
        fdi_num = match.group(1)
        quadrant = match.group(2)  # UL, UR, LL, LR
        tooth_num = match.group(3)
        tooth_type = match.group(4).replace("_", " ")
        
        # Determine position
        quadrant_map = {
            "UR": "Upper right",
            "UL": "Upper left",
            "LR": "Lower right",
            "LL": "Lower left"
        }
        position = quadrant_map.get(quadrant, quadrant)
        
        return f"FDI {fdi_num}: {position} {tooth_num}th tooth from midline ({tooth_type})"
    
    @classmethod
    def from_json(cls, json_path: str) -> "AnatomicalNaming":
        """
        Load anatomical naming from a class_subclass_info.json file.
        
        Args:
            json_path: Path to the JSON file
            
        Returns:
            AnatomicalNaming instance
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Parse classes (keys are strings like "1", "2", ...)
        classes = {}
        for key, value in data.get("classes", {}).items():
            classes[int(key)] = value
        
        # Parse subclasses (keys are strings like "0", "1", ...)
        subclasses = {}
        for key, value in data.get("subclasses", {}).items():
            subclasses[int(key)] = value
        
        return cls(classes, subclasses, source_path=json_path)
    
    @classmethod
    def load_default(cls, search_paths: Optional[List[str]] = None) -> "AnatomicalNaming":
        """
        Load from a default/reference class_subclass_info.json.
        
        Args:
            search_paths: List of paths to search for the JSON file
            
        Returns:
            AnatomicalNaming instance
        """
        if search_paths is None:
            # Default search paths
            search_paths = [
                "./class_subclass_info.json",
                "../class_subclass_info.json",
                "../../class_subclass_info.json",
            ]
        
        for path in search_paths:
            if Path(path).exists():
                return cls.from_json(path)
        
        raise FileNotFoundError(
            f"Could not find class_subclass_info.json in: {search_paths}"
        )
    
    # =========================================================================
    # NAME-based Filtering Methods (NEW)
    # =========================================================================
    
    def get_class_ids_for_name(self, name: str) -> List[int]:
        """
        Get the class IDs that should be included for a given NAME.
        
        Args:
            name: NAME value (e.g., "Upper_Teeth", "Mandible")
            
        Returns:
            List of class IDs to include
        """
        if name not in NAME_TO_CLASSES:
            # If NAME is not defined, include all classes
            return sorted(self.classes.keys())
        
        class_names = NAME_TO_CLASSES[name]
        class_ids = []
        
        for standard_class_name in class_names:
            # Try to find matching class ID in the actual JSON data
            found = False
            
            # Method 1: Direct match with raw name from JSON
            for class_id, raw_name in self._class_id_to_raw_name.items():
                # Normalize names for comparison (handle variations like "Lower_Teeth" vs "Lower Teeth")
                normalized_raw = raw_name.replace("_", " ").replace("&", "and").strip().lower()
                normalized_standard = standard_class_name.replace("_", " ").replace("&", "and").strip().lower()
                
                if normalized_raw == normalized_standard:
                    class_ids.append(class_id)
                    found = True
                    break
            
            # Method 2: If not found, try partial match (e.g., "Mandible" matches "Mandible")
            if not found:
                for class_id, raw_name in self._class_id_to_raw_name.items():
                    if standard_class_name.lower() in raw_name.lower() or \
                       raw_name.lower() in standard_class_name.lower():
                        class_ids.append(class_id)
                        found = True
                        break
        
        return sorted(class_ids)
    
    def get_subclass_ids_for_name(self, name: str) -> List[int]:
        """
        Get the subclass IDs that should be included for a given NAME.
        
        Args:
            name: NAME value (e.g., "Upper_Teeth", "Mandible")
            
        Returns:
            List of subclass IDs to include, sorted
        """
        # Get allowed class IDs for this NAME
        allowed_class_ids = set(self.get_class_ids_for_name(name))
        
        # Filter subclasses by their class ID
        filtered_ids = []
        for subclass_id in self.subclasses.keys():
            class_id = self._subclass_to_class.get(subclass_id, 0)
            if class_id in allowed_class_ids:
                filtered_ids.append(subclass_id)
        
        return sorted(filtered_ids)
    
    def get_filtered_subclass_display_names(self, name: str) -> List[str]:
        """
        Get display names for subclasses filtered by NAME.
        
        Args:
            name: NAME value (e.g., "Upper_Teeth", "Mandible")
            
        Returns:
            List of human-readable display names for the filtered subclasses
        """
        filtered_ids = self.get_subclass_ids_for_name(name)
        return [self.get_subclass_display_name(sid) for sid in filtered_ids]
    
    def get_filtered_subclass_raw_names(self, name: str) -> List[str]:
        """
        Get raw names for subclasses filtered by NAME.
        
        Args:
            name: NAME value (e.g., "Upper_Teeth", "Mandible")
            
        Returns:
            List of raw names for the filtered subclasses
        """
        filtered_ids = self.get_subclass_ids_for_name(name)
        return [self.get_subclass_raw_name(sid) for sid in filtered_ids]
    
    def get_available_names(self) -> List[str]:
        """Get list of all available NAME values."""
        return sorted(NAME_TO_CLASSES.keys())
    
    # =========================================================================
    # Public API (Original)
    # =========================================================================
    
    def get_subclass_count(self) -> int:
        """Get the total number of subclasses."""
        return len(self.subclasses)
    
    def get_class_count(self) -> int:
        """Get the total number of classes."""
        return len(self.classes)
    
    def get_subclass_raw_name(self, subclass_id: int) -> str:
        """Get the raw label name for a subclass ID."""
        return self._id_to_raw_name.get(subclass_id, f"Unknown_{subclass_id}")
    
    def get_subclass_display_name(self, subclass_id: int) -> str:
        """Get the human-readable display name for a subclass ID."""
        return self._id_to_display_name.get(subclass_id, f"Unknown Region {subclass_id}")
    
    def get_subclass_id_from_raw_name(self, raw_name: str) -> Optional[int]:
        """Get the subclass ID from a raw label name."""
        return self._raw_name_to_id.get(raw_name)
    
    def get_class_display_name(self, class_id: int) -> str:
        """Get the human-readable display name for a class ID."""
        return self._class_id_to_display_name.get(class_id, f"Unknown Class {class_id}")
    
    def get_class_raw_name(self, class_id: int) -> str:
        """Get the raw name for a class ID from JSON."""
        return self._class_id_to_raw_name.get(class_id, f"Unknown Class {class_id}")
    
    def get_class_id_from_raw_name(self, raw_name: str) -> Optional[int]:
        """Get the class ID from a raw class name."""
        return self._class_raw_name_to_id.get(raw_name)
    
    def get_class_id_for_subclass(self, subclass_id: int) -> int:
        """Get the parent class ID for a subclass ID."""
        return self._subclass_to_class.get(subclass_id, 0)
    
    def get_all_subclass_ids(self) -> List[int]:
        """Get all subclass IDs in sorted order."""
        return sorted(self.subclasses.keys())
    
    def get_all_subclass_display_names(self) -> List[str]:
        """Get all subclass display names in ID order."""
        return [
            self.get_subclass_display_name(sid)
            for sid in self.get_all_subclass_ids()
        ]
    
    def get_all_subclass_raw_names(self) -> List[str]:
        """Get all subclass raw names in ID order."""
        return [
            self.get_subclass_raw_name(sid)
            for sid in self.get_all_subclass_ids()
        ]
    
    def get_subclass_name_map(self) -> Dict[int, str]:
        """
        Get a dictionary mapping subclass_id -> display_name.
        This is a drop-in replacement for the old SUBCLASS_NAME_MAP.
        """
        return dict(self._id_to_display_name)
    
    def get_class_name_map(self) -> Dict[int, str]:
        """
        Get a dictionary mapping class_id -> display_name.
        This is a drop-in replacement for the old CLASS_NAME_MAP.
        """
        return dict(self._class_id_to_display_name)
    
    def display_name_to_raw_name(self, display_name: str) -> Optional[str]:
        """Convert a display name back to raw name (if possible)."""
        for subclass_id, disp in self._id_to_display_name.items():
            if disp == display_name:
                return self._id_to_raw_name.get(subclass_id)
        return None
    
    def __repr__(self) -> str:
        return (
            f"AnatomicalNaming("
            f"classes={self.get_class_count()}, "
            f"subclasses={self.get_subclass_count()}, "
            f"source={self.source_path})"
        )


# ==============================================================================
# Convenience Functions
# ==============================================================================

def load_anatomical_naming(json_path: str) -> AnatomicalNaming:
    """
    Convenience function to load anatomical naming from JSON.
    
    Args:
        json_path: Path to class_subclass_info.json
        
    Returns:
        AnatomicalNaming instance
    """
    return AnatomicalNaming.from_json(json_path)


def get_display_name(raw_name: str) -> str:
    """
    Get display name for a raw label name without loading full JSON.
    Uses the predefined SUBCLASS_DISPLAY_NAME_MAP.
    
    Args:
        raw_name: Raw label name from class_subclass_info.json
        
    Returns:
        Human-readable display name
    """
    if raw_name in SUBCLASS_DISPLAY_NAME_MAP:
        return SUBCLASS_DISPLAY_NAME_MAP[raw_name]
    
    # Fallback: clean up the name
    return raw_name.replace("_", " ")


# ==============================================================================
# Test/Demo
# ==============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("AnatomicalNaming Module Test with NAME Filtering")
    print("=" * 80)
    
    # Test with sample data
    sample_json = {
        "classes": {
            "1": {"name": "Maxilla & Upper Skull", "num_points": 2048},
            "2": {"name": "Mandible", "num_points": 2048},
            "3": {"name": "Upper Teeth", "num_points": 2040},
            "4": {"name": "Lower Teeth", "num_points": 2056}
        },
        "subclasses": {
            "0": {"name": "Maxilla_and_Upper_Skull_RPosteriorSuperior", "num_points": 244},
            "8": {"name": "Mandible_Right_Anterior_Inferior_Lingual", "num_points": 196},
            "20": {"name": "FDI_48_LR8_third_molar", "num_points": 130},
            "39": {"name": "FDI_11_UR1_central_incisor", "num_points": 123},
            "38": {"name": "FDI_21_UL1_central_incisor", "num_points": 125},
        }
    }
    
    # Write temp file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(sample_json, f)
        temp_path = f.name
    
    try:
        # Load and test
        naming = AnatomicalNaming.from_json(temp_path)
        print(f"\nLoaded: {naming}")
        
        print("\n--- Available NAMEs ---")
        for name in naming.get_available_names():
            print(f"  {name}")
        
        print("\n--- Test NAME Filtering ---")
        test_names = ["Upper_Teeth", "Mandible", "dental_model"]
        for name in test_names:
            print(f"\nNAME: {name}")
            class_ids = naming.get_class_ids_for_name(name)
            print(f"  Class IDs: {class_ids}")
            
            subclass_ids = naming.get_subclass_ids_for_name(name)
            print(f"  Subclass IDs: {subclass_ids}")
            
            display_names = naming.get_filtered_subclass_display_names(name)
            print(f"  Display Names: {display_names}")
        
    finally:
        import os
        os.unlink(temp_path)
    
    print("\n" + "=" * 80)
    print("Test Complete!")
    print("=" * 80)