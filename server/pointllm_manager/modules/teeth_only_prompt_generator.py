"""
teeth_only_prompt_generator.py - Specialized prompt generator for teeth-only datasets

For datasets containing only teeth (no Mandible or Maxilla reference),
special prompts are needed because vertical orientation cannot be determined.
"""

import random
from typing import Tuple, Optional, List
from dataclasses import dataclass

from modules.fdi_mirror_utils import (
    get_fdi_mirror_vertically, 
    extract_fdi_from_subclass_name,
    get_fdi_quadrant_info
)


COLORIZATION_EFFECTS = {
    "hover": {
        # 黄色く光るのではなく、純粋に「明るく」する表現へ修正
        "verb": "brightened", 
        "adjective": "brightened with increased luminosity", 
        "noun": "brightness enhancement"
    },
    "contrast": {
        # 明るい色ではなく、「補色」であることを明記
        "verb": "colored",
        "adjective": "shown in complementary colors",
        "noun": "complementary color highlighting"
    },
    "fixed_blue": {
        "verb": "colored",
        "adjective": "colored in blue",
        "noun": "blue coloring"
    },
    "fixed_red": {
        "verb": "colored",
        "adjective": "colored in red",
        "noun": "red coloring"
    },
    "fixed_black": {
        "verb": "darkened",
        "adjective": "darkened",
        "noun": "dark effect"
    },
}


class TeethOnlyPromptGenerator:
    """
    Prompt generator for teeth-only datasets (no Mandible/Maxilla reference).
    
    Generates two types of prompts:
    1. Version 1: Ambiguous (both FDI possibilities due to unknown orientation)
    2. Version 2: Class-specified (for Upper_Teeth or Lower_Teeth only)
    """
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
    
    def can_use_version2(self, dataset_name: str) -> bool:
        """
        Check if Version 2 (class-specified) can be used.
        
        Version 2 requires knowing if teeth are upper or lower,
        so it only works for Upper_Teeth or Lower_Teeth datasets.
        """
        return dataset_name in ["Upper_Teeth", "Lower_Teeth"]
    
    def generate_version1_ambiguous(
        self,
        colorization_type: str,
        primary_region,
        all_candidates: List[str],
        distance_mm: float,
        dataset_name: str
    ) -> Tuple[str, str]:
        """
        Generate Version 1: Ambiguous orientation prompt.
        
        Since there's no Mandible/Maxilla, we can't determine up/down.
        Answer provides both FDI possibilities.
        
        Args:
            dataset_name: "Upper_Teeth", "Lower_Teeth", or "U_and_L_Teeth"
        """
        color_info = COLORIZATION_EFFECTS.get(
            colorization_type, 
            COLORIZATION_EFFECTS["contrast"]
        )
        
        num_candidates = len(all_candidates)
        all_candidates_str = ", ".join(all_candidates)
        
        # Extract FDI number from primary region
        fdi_num = extract_fdi_from_subclass_name(primary_region.subclass_name)
        
        if fdi_num is None:
            # Fallback for non-tooth regions
            return self._generate_fallback(
                colorization_type, primary_region, all_candidates, distance_mm
            )
        
        # Get mirror FDI
        mirror_fdi = get_fdi_mirror_vertically(fdi_num)
        original_info = get_fdi_quadrant_info(fdi_num)
        mirror_info = get_fdi_quadrant_info(mirror_fdi)
        
        # Question templates
        if dataset_name == "U_and_L_Teeth":
            # Special handling for mixed upper/lower
            question = (
                f"This dental point cloud contains {num_candidates} teeth: [{all_candidates_str}]. "
                f"The visualization shows both upper and lower teeth, but without mandible or maxilla "
                f"references, vertical orientation cannot be determined. "
                f"One region is {color_info['adjective']}, with the user's hand at {distance_mm:.1f}mm distance. "
                f"Considering that the model may be vertically flipped, identify the possible FDI numbers."
            )
        else:
            # Upper_Teeth or Lower_Teeth only
            teeth_type = "upper" if dataset_name == "Upper_Teeth" else "lower"
            question = (
                f"This point cloud shows {num_candidates} {teeth_type} teeth: [{all_candidates_str}]. "
                f"Without mandible or maxilla reference structures, the vertical orientation is ambiguous. "
                f"One tooth is {color_info['adjective']} (hand distance: {distance_mm:.1f}mm). "
                f"What are the possible FDI numbers considering potential vertical flipping?"
            )
        
        # Answer with both possibilities
        if dataset_name == "U_and_L_Teeth":
            answer = (
                f"Since this model contains both upper and lower teeth without jaw bone references, "
                f"vertical orientation cannot be determined. The {color_info['verb']} region appears to be "
                f"on the {original_info['side'].lower()} side. "
                f"If the model is in standard orientation, this is FDI {fdi_num} "
                f"({original_info['abbr']}{original_info['tooth_number']}: "
                f"{original_info['arch']} {original_info['side']}). "
                f"If vertically flipped, this would be FDI {mirror_fdi} "
                f"({mirror_info['abbr']}{mirror_info['tooth_number']}: "
                f"{mirror_info['arch']} {mirror_info['side']}). "
                f"Therefore: FDI {fdi_num} or FDI {mirror_fdi}."
            )
        else:
            answer = (
                f"Without mandible or maxilla landmarks, we cannot definitively determine "
                f"if these are upper or lower teeth. The {color_info['verb']} tooth is positioned "
                f"on the {original_info['side'].lower()} side. "
                f"This could be FDI {fdi_num} ({original_info['abbr']}{original_info['tooth_number']}) "
                f"or, if the model is vertically flipped, FDI {mirror_fdi} "
                f"({mirror_info['abbr']}{mirror_info['tooth_number']}). "
                f"Answer: FDI {fdi_num} or FDI {mirror_fdi}."
            )
        
        return question, answer
    
    def generate_version2_class_specified(
        self,
        colorization_type: str,
        primary_region,
        all_candidates: List[str],
        distance_mm: float,
        dataset_name: str
    ) -> Tuple[str, str]:
        """
        Generate Version 2: Class-specified prompt (Upper_Teeth or Lower_Teeth only).
        
        Prompt explicitly states "This is Upper Teeth" or "This is Lower Teeth",
        allowing for a definitive FDI number answer.
        """
        if not self.can_use_version2(dataset_name):
            raise ValueError(f"Version 2 not available for {dataset_name}")
        
        color_info = COLORIZATION_EFFECTS.get(
            colorization_type, 
            COLORIZATION_EFFECTS["contrast"]
        )
        
        num_candidates = len(all_candidates)
        all_candidates_str = ", ".join(all_candidates)
        
        # Extract FDI
        fdi_num = extract_fdi_from_subclass_name(primary_region.subclass_name)
        if fdi_num is None:
            return self._generate_fallback(
                colorization_type, primary_region, all_candidates, distance_mm
            )
        
        fdi_info = get_fdi_quadrant_info(fdi_num)
        
        # Explicit class specification
        class_statement = f"This is {dataset_name.replace('_', ' ')}"
        
        # Question
        question = (
            f"{class_statement}. "
            f"The point cloud contains {num_candidates} teeth: [{all_candidates_str}]. "
            f"One tooth is {color_info['adjective']} (hand distance: {distance_mm:.1f}mm). "
            f"What is the FDI number?"
        )
        
        # Answer with definitive FDI
        answer = (
            f"Given that this is {dataset_name.replace('_', ' ')}, we can determine the exact position. "
            f"The {color_info['verb']} tooth is on the {fdi_info['side'].lower()} side. "
            f"This corresponds to FDI {fdi_num} "
            f"({fdi_info['abbr']}{fdi_info['tooth_number']}: "
            f"{fdi_info['arch']} {fdi_info['side']} tooth #{fdi_info['tooth_number']})."
        )
        
        return question, answer
    
    def generate_ul_teeth_protrusion_variant(
        self,
        colorization_type: str,
        primary_region,
        all_candidates: List[str],
        distance_mm: float
    ) -> Tuple[str, str]:
        """
        Generate U_and_L_Teeth variant using anterior protrusion as a clue.
        
        Upper anterior teeth typically protrude forward relative to lower teeth.
        """
        color_info = COLORIZATION_EFFECTS.get(
            colorization_type, 
            COLORIZATION_EFFECTS["contrast"]
        )
        
        num_candidates = len(all_candidates)
        all_candidates_str = ", ".join(all_candidates)
        
        fdi_num = extract_fdi_from_subclass_name(primary_region.subclass_name)
        if fdi_num is None:
            return self._generate_fallback(
                colorization_type, primary_region, all_candidates, distance_mm
            )
        
        mirror_fdi = get_fdi_mirror_vertically(fdi_num)
        original_info = get_fdi_quadrant_info(fdi_num)
        mirror_info = get_fdi_quadrant_info(mirror_fdi)
        
        # Question
        question = (
            f"This point cloud contains {num_candidates} teeth from both upper and lower arches: "
            f"[{all_candidates_str}]. "
            f"One set of anterior teeth protrudes forward. "
            f"A tooth is {color_info['adjective']} (hand distance: {distance_mm:.1f}mm). "
            f"If the protruding teeth are upper, what is the FDI number? "
            f"If the protruding teeth are lower, what is the FDI number?"
        )
        
        # Answer
        # Determine which is upper and which is lower
        if "upper" in original_info['arch'].lower():
            upper_fdi = fdi_num
            lower_fdi = mirror_fdi
        else:
            lower_fdi = fdi_num
            upper_fdi = mirror_fdi
        
        answer = (
            f"The {color_info['verb']} tooth is positioned on the "
            f"{original_info['side'].lower()} side. "
            f"Considering anterior protrusion: "
            f"If the protruding teeth are upper (typical anatomy), this is FDI {upper_fdi}. "
            f"If the protruding teeth are lower (inverted), this is FDI {lower_fdi}."
        )
        
        return question, answer
    
    def _generate_fallback(
        self,
        colorization_type: str,
        primary_region,
        all_candidates: List[str],
        distance_mm: float
    ) -> Tuple[str, str]:
        """Fallback for non-tooth regions"""
        color_info = COLORIZATION_EFFECTS.get(
            colorization_type, 
            COLORIZATION_EFFECTS["contrast"]
        )
        
        num_candidates = len(all_candidates)
        all_candidates_str = ", ".join(all_candidates)
        
        question = (
            f"This point cloud contains {num_candidates} structures: [{all_candidates_str}]. "
            f"One region is {color_info['adjective']} (hand distance: {distance_mm:.1f}mm). "
            f"Which structure is indicated?"
        )
        
        answer = (
            f"The {color_info['verb']} region corresponds to "
            f"{primary_region.anatomical_subclass_name}."
        )
        
        return question, answer


# Test
if __name__ == "__main__":
    from dataclasses import dataclass
    
    @dataclass
    class MockRegion:
        subclass_name: str
        anatomical_subclass_name: str
    
    generator = TeethOnlyPromptGenerator(seed=42)
    
    # Mock region
    region = MockRegion(
        subclass_name="FDI_48_LR8_third_molar",
        anatomical_subclass_name="FDI 48: Lower right 8th tooth from midline (3rd molar / Wisdom tooth)"
    )
    
    candidates = [
        "FDI 38: Lower left 8th tooth",
        "FDI 48: Lower right 8th tooth",
        # ... more
    ]
    
    print("=" * 80)
    print("Version 1: Ambiguous (Lower_Teeth)")
    print("=" * 80)
    q, a = generator.generate_version1_ambiguous(
        "hover", region, candidates, 10.5, "Lower_Teeth"
    )
    print(f"Question:\n{q}\n")
    print(f"Answer:\n{a}\n")
    
    print("\n" + "=" * 80)
    print("Version 2: Class-specified (Lower_Teeth)")
    print("=" * 80)
    q, a = generator.generate_version2_class_specified(
        "contrast", region, candidates, 10.5, "Lower_Teeth"
    )
    print(f"Question:\n{q}\n")
    print(f"Answer:\n{a}\n")
    
    print("\n" + "=" * 80)
    print("U_and_L_Teeth: Protrusion variant")
    print("=" * 80)
    q, a = generator.generate_ul_teeth_protrusion_variant(
        "blue", region, candidates, 10.5
    )
    print(f"Question:\n{q}\n")
    print(f"Answer:\n{a}\n")
