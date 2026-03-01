# PointLLM Custom Object Evaluation System

This evaluation system allows you to assess PointLLM's performance on custom point cloud objects with comprehensive visualization and reporting.

## Overview

The system evaluates PointLLM on your custom point cloud data (.npy files) using multiple prompts and generates:
- Multi-view 3D visualizations of each point cloud
- Inference results for multiple prompts
- Comprehensive Excel report with embedded images
- CSV data for further analysis
- HTML report for easy viewing

## Setup

### Prerequisites

1. **PointLLM Repository**: Clone the PointLLM repository
```bash
git clone https://github.com/OpenRobotLab/PointLLM.git
cd PointLLM
```

2. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

3. **Model Download**: The model will be automatically downloaded from HuggingFace on first run:
   - Model: `RunsenXu/PointLLM_7B_v1.2`

### Data Preparation

Place your point cloud data in `.npy` format in the directory:
```
PointLLM/pointllm/data/npy/
```

**Data Format Requirements**:
- Each `.npy` file should contain a point cloud array of shape `(N, 6)` or `(N, 3)`
- Columns: `[x, y, z, r, g, b]` or just `[x, y, z]`
- RGB values should be normalized to [0, 1] range
- XYZ coordinates will be automatically normalized by the script

## Usage

### Basic Usage

Run the evaluation with default settings:
```bash
python pointllm_evaluate_enhanced.py
```

### Advanced Usage

```bash
python pointllm_evaluate_enhanced.py \
    --model_path RunsenXu/PointLLM_7B_v1.2 \
    --data_path PointLLM/pointllm/data/npy \
    --prompt_file prompts.json \
    --output_dir evaluation_results \
    --device cuda \
    --max_objects 10  # Process only first 10 objects for testing
```

### Parameters

- `--model_path`: Path to PointLLM model (default: RunsenXu/PointLLM_7B_v1.2)
- `--data_path`: Directory containing .npy files (default: PointLLM/pointllm/data/npy)
- `--prompt_file`: JSON file with prompt definitions (default: prompts.json)
- `--output_dir`: Output directory for results (default: evaluation_results)
- `--device`: Device for inference - cuda or cpu (default: cuda)
- `--max_objects`: Maximum objects to process, -1 for all (default: -1)

## Output Files

The evaluation generates the following files in the output directory:

1. **Excel Report** (`evaluation_results_TIMESTAMP.xlsx`):
   - Contains embedded visualizations
   - All prompt responses in organized columns
   - Prompt definitions in header row

2. **CSV File** (`evaluation_results_TIMESTAMP.csv`):
   - Raw data for further analysis
   - Easy to import into analysis tools

3. **HTML Report** (`evaluation_results_TIMESTAMP.html`):
   - Web-viewable report with all results
   - Includes visualizations and responses

4. **Visualizations** (`visualizations/`):
   - Multi-view PNG images for each object
   - 4 views: 3D perspective, top, front, side

5. **Summary** (`evaluation_summary_TIMESTAMP.json`):
   - Metadata about the evaluation run

## Prompt Configuration

Edit `prompts.json` to customize evaluation prompts:

```json
{
  "prompts": {
    "Prompt_A": {
      "definition": "Short description for table header",
      "full_prompt": "Complete prompt with context for inference"
    },
    ...
  }
}
```

### Default Prompts

- **Prompt_A**: Describe this point cloud object
- **Prompt_B**: Main geometric features analysis
- **Prompt_C**: Function/purpose inference
- **Prompt_D**: Object category classification
- **Prompt_E**: Color/material properties analysis

## Excel Output Format

The Excel file has the following structure:

| Row | Content |
|-----|---------|
| 1 | Headers: Object ID, Visualization, Prompt_A, Prompt_B, ... |
| 2 | Prompt definitions |
| 3+ | Object data with embedded visualizations and responses |

## Troubleshooting

### Memory Issues
- Reduce `--max_objects` to process fewer objects at once
- Use CPU mode with `--device cpu` (slower but uses less memory)

### Model Loading Issues
- Ensure you have sufficient disk space for model download (~15GB)
- Check internet connection for HuggingFace model download

### Point Cloud Issues
- Verify .npy files are in correct format
- Check that RGB values are normalized to [0, 1]
- Ensure point clouds have reasonable number of points (1000-100000)

## Performance Tips

1. **GPU Usage**: Use CUDA for 10x faster inference
2. **Batch Processing**: Process in batches using `--max_objects` if memory limited
3. **Point Cloud Size**: Keep point clouds around 8192 points for optimal performance
4. **Multi-processing**: Run multiple instances with different data subsets

## Example Workflow

1. Prepare your point cloud data:
```python
import numpy as np

# Example: Save a point cloud with color
points = np.random.randn(8192, 3)  # xyz
colors = np.random.rand(8192, 3)   # rgb in [0,1]
point_cloud = np.concatenate([points, colors], axis=1)
np.save('PointLLM/pointllm/data/npy/my_object.npy', point_cloud)
```

2. Run evaluation:
```bash
python pointllm_evaluate_enhanced.py --max_objects 5
```

3. View results:
   - Open `evaluation_results/evaluation_results_*.xlsx` in Excel
   - Or open `evaluation_results/evaluation_results_*.html` in browser

## Notes

- The script automatically normalizes XYZ coordinates to unit sphere
- RGB values should be pre-normalized to [0, 1] range
- Point clouds are resampled to exactly 8192 points
- Generation parameters are optimized to reduce repetitive outputs
- Images in Excel are embedded, not linked (portable file)

## Citation

If you use this evaluation system, please cite the original PointLLM paper:
```
@article{xu2023pointllm,
  title={PointLLM: Empowering Large Language Models to Understand Point Clouds},
  author={Xu, Runsen and Wang, Xiaolong and Wang, Tai and Chen, Yilun and Pang, Jiangmiao and Lin, Dahua},
  journal={arXiv preprint arXiv:2308.16911},
  year={2023}
}
```
