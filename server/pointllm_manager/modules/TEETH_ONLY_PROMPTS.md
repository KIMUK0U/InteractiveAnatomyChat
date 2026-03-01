# Teeth-Only Prompt Generation Guide

## Overview

When datasets contain **only teeth** (no Mandible or Maxilla reference structures), vertical orientation cannot be determined. This requires special prompt generation strategies.

## Problem

Without mandible or maxilla bones:
- We cannot tell if teeth are upper or lower
- Vertical flipping makes FDI 48 (Lower Right 8th) look like FDI 28 (Upper Left 8th)
- Standard anatomical descriptions fail

## Solution: Two Prompt Versions

### Version 1: Ambiguous Orientation (All teeth datasets)

**Applicable to**: `Upper_Teeth`, `Lower_Teeth`, `U_and_L_Teeth`

**Question Example**:
```
This point cloud shows 16 lower teeth: [FDI 38, FDI 48, ...]. 
Without mandible or maxilla reference structures, the vertical 
orientation is ambiguous. One tooth is glowing with increased 
brightness (hand distance: 12.5mm). What are the possible FDI 
numbers considering potential vertical flipping?
```

**Answer Example**:
```
Without mandible or maxilla landmarks, we cannot definitively 
determine if these are upper or lower teeth. The glowing tooth 
is positioned on the left side. This could be FDI 38 (LL8) or, 
if the model is vertically flipped, FDI 18 (UR8). 
Answer: FDI 38 or FDI 18.
```

**FDI Mirror Mapping**:
- Upper Right (11-18) ⟷ Lower Left (31-38) when flipped
- Upper Left (21-28) ⟷ Lower Right (41-48) when flipped

### Version 2: Class-Specified (Upper_Teeth or Lower_Teeth only)

**NOT applicable to**: `U_and_L_Teeth` (mixed upper/lower)

**Question Example**:
```
This is Lower Teeth. The point cloud contains 16 teeth: 
[FDI 38, FDI 48, ...]. One tooth is shown in contrasting 
bright colors (hand distance: 12.5mm). What is the FDI number?
```

**Answer Example**:
```
Given that this is Lower Teeth, we can determine the exact position. 
The highlighted tooth is on the left side. This corresponds to 
FDI 38 (LL8: Lower Left tooth #8).
```

### Special Case: U_and_L_Teeth Protrusion Variant

**Applicable to**: `U_and_L_Teeth` only

Uses anterior tooth protrusion as a clue (upper anterior teeth typically protrude forward).

**Question Example**:
```
This point cloud contains 32 teeth from both upper and lower arches. 
One set of anterior teeth protrudes forward. A tooth is colored in 
blue (hand distance: 8.3mm). If the protruding teeth are upper, 
what is the FDI number? If the protruding teeth are lower, what 
is the FDI number?
```

**Answer Example**:
```
The colored tooth is positioned on the left side. Considering 
anterior protrusion: If the protruding teeth are upper (typical 
anatomy), this is FDI 18. If the protruding teeth are lower 
(inverted), this is FDI 38.
```

## Dataset Configuration

### Variation Distribution

For `num_prompt_variations=6`:

**Upper_Teeth / Lower_Teeth**:
- Variations 0-2: Version 1 (Ambiguous)
- Variations 3-5: Version 2 (Class-specified)

**U_and_L_Teeth**:
- Variations 0-2: Version 1 (Ambiguous)
- Variations 3-5: Protrusion variant

## File Structure

```
project/
├── fdi_mirror_utils.py              # FDI number mirroring logic
├── teeth_only_prompt_generator.py   # Teeth-only prompt generator
├── build_dataset_improved.py        # Main builder (with integration)
└── anatomical_naming_improved.py    # Dynamic naming system
```

## Usage Example

```python
from build_dataset_improved import DatasetConfig, DatasetBuilder

# Build Lower_Teeth dataset
config = DatasetConfig(
    data_root="/path/to/data",
    output_dir="./output",
    name="Lower_Teeth",  # Triggers teeth-only prompts
    num_prompt_variations=6,
    seed=42,
)

builder = DatasetBuilder(config)
results = builder.build()
```

## Metadata

Each sample includes `prompt_version` in metadata:

```json
{
  "metadata": {
    "prompt_version": "v1_ambiguous",  // or "v2_class_specified" or "ul_protrusion"
    "primary_region": {
      "id": 0,
      "display_name": "FDI 38: Lower left 8th tooth..."
    }
  }
}
```

## FDI Number Correspondence Table

| Original | Quadrant | Mirror (Flipped) | Quadrant |
|----------|----------|------------------|----------|
| FDI 18 | UR (Upper Right) | FDI 38 | LL (Lower Left) |
| FDI 17 | UR | FDI 37 | LL |
| FDI 11 | UR | FDI 31 | LL |
| FDI 28 | UL (Upper Left) | FDI 48 | LR (Lower Right) |
| FDI 27 | UL | FDI 47 | LR |
| FDI 21 | UL | FDI 41 | LR |

## Testing

```bash
# Test FDI mirroring logic
python fdi_mirror_utils.py

# Test prompt generation
python teeth_only_prompt_generator.py

# Test integration
python test_teeth_only_integration.py
```

## Important Notes

1. **Version 2 is NOT available for U_and_L_Teeth** because we can't definitively say "This is Upper Teeth" when the dataset contains both.

2. **Protrusion variant is ONLY for U_and_L_Teeth** because it requires comparing upper and lower teeth.

3. **All answers maintain proper FDI notation**: FDI XX format with clear quadrant specification.

4. **Metadata tracking**: Each sample records which prompt version was used for analysis.

## Benefits

✅ **Scientifically accurate**: Acknowledges orientation ambiguity  
✅ **Flexible training**: Two different answer formats  
✅ **Clear documentation**: Metadata tracks prompt type  
✅ **Anatomically sound**: Uses proper FDI mirroring logic  
✅ **Well-tested**: Comprehensive test coverage  

## References

- FDI World Dental Federation notation
- Vertical flipping in dental imaging
- Anatomical orientation determination
