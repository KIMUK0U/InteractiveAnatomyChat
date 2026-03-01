"""
fdi_mirror_utils.py - FDI Tooth Numbering Mirror/Flip Utilities

When upper/lower jaw distinction is not available (teeth-only datasets),
we need to consider mirror possibilities when teeth are flipped vertically.
"""

from typing import Optional, Tuple


def get_fdi_mirror_vertically(fdi_number: int) -> Optional[int]:
    """
    Get the corresponding FDI number when teeth are flipped vertically.
    
    Vertical flip mapping (upper ⟷ lower, opposite side):
    - Upper Right (11-18) ⟷ Lower Left (31-38)
    - Upper Left (21-28) ⟷ Lower Right (41-48)
    
    Args:
        fdi_number: Original FDI number (11-48)
        
    Returns:
        Mirrored FDI number, or None if invalid
        
    Examples:
        18 (UR8) → 38 (LL8)
        28 (UL8) → 48 (LR8)
        48 (LR8) → 28 (UL8)
        31 (LL1) → 11 (UR1)
    """
    if not (11 <= fdi_number <= 48):
        return None
    
    quadrant = fdi_number // 10
    tooth_num = fdi_number % 10
    
    # Quadrant mapping for vertical flip
    quadrant_map = {
        1: 3,  # UR → LL
        2: 4,  # UL → LR
        3: 1,  # LL → UR
        4: 2,  # LR → UL
    }
    
    mirrored_quadrant = quadrant_map.get(quadrant)
    if mirrored_quadrant is None:
        return None
    
    return mirrored_quadrant * 10 + tooth_num


def get_fdi_both_possibilities(fdi_number: int) -> Tuple[int, int]:
    """
    Get both FDI possibilities when orientation is unknown.
    
    Returns:
        (original, mirrored) tuple
        
    Example:
        48 → (48, 28)  # Could be LR8 or (flipped) UL8
    """
    mirrored = get_fdi_mirror_vertically(fdi_number)
    if mirrored is None:
        return (fdi_number, fdi_number)
    return (fdi_number, mirrored)


def get_fdi_quadrant_info(fdi_number: int) -> dict:
    """
    Get detailed information about an FDI tooth.
    
    Returns:
        Dictionary with quadrant info, side, arch, etc.
    """
    if not (11 <= fdi_number <= 48):
        return {"valid": False}
    
    quadrant = fdi_number // 10
    tooth_num = fdi_number % 10
    
    quadrant_info = {
        1: {"arch": "Upper", "side": "Right", "abbr": "UR"},
        2: {"arch": "Upper", "side": "Left", "abbr": "UL"},
        3: {"arch": "Lower", "side": "Left", "abbr": "LL"},
        4: {"arch": "Lower", "side": "Right", "abbr": "LR"},
    }
    
    info = quadrant_info.get(quadrant, {})
    
    return {
        "valid": True,
        "fdi": fdi_number,
        "quadrant": quadrant,
        "tooth_number": tooth_num,
        "arch": info.get("arch", "Unknown"),
        "side": info.get("side", "Unknown"),
        "abbr": info.get("abbr", "Unknown"),
    }


def extract_fdi_from_subclass_name(subclass_name: str) -> Optional[int]:
    """
    Extract FDI number from subclass name.
    
    Examples:
        "FDI_48_LR8_third_molar" → 48
        "FDI_11_UR1_central_incisor" → 11
    """
    import re
    match = re.search(r'FDI_(\d{2})', subclass_name)
    if match:
        return int(match.group(1))
    return None


# Test
if __name__ == "__main__":
    print("Testing FDI Mirror Utilities")
    print("=" * 60)
    
    test_cases = [18, 28, 48, 38, 11, 21, 31, 41]
    
    for fdi in test_cases:
        info = get_fdi_quadrant_info(fdi)
        mirror = get_fdi_mirror_vertically(fdi)
        mirror_info = get_fdi_quadrant_info(mirror)
        
        print(f"\nFDI {fdi} ({info['abbr']}{info['tooth_number']})")
        print(f"  Original: {info['arch']} {info['side']}")
        print(f"  Mirrored: {mirror} ({mirror_info['abbr']}{mirror_info['tooth_number']})")
        print(f"  → {mirror_info['arch']} {mirror_info['side']}")
