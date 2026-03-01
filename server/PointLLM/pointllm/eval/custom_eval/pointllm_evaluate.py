"""
PointLLM Evaluation Program for Custom Point Cloud Objects
This program evaluates PointLLM on custom point cloud objects, generating:
1. 3D visualizations of the point clouds
2. Inference results for multiple prompts
3. A comprehensive results table (Excel format)
"""

import argparse
import os
import sys
import glob
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import PointLLM modules
sys.path.append('PointLLM')
from pointllm.conversation import conv_templates, SeparatorStyle
from pointllm.utils import disable_torch_init
from pointllm.model import PointLLMLlamaForCausalLM
from pointllm.model.utils import KeywordsStoppingCriteria
from transformers import AutoTokenizer

def pc_normalize(pc):
    """
    Normalize a point cloud to fit within a unit sphere
    pc: Nx6 or Nx3 array (xyz + optional rgb)
    """
    xyz = pc[:, :3]
    other_features = pc[:, 3:] if pc.shape[1] > 3 else None
    
    # Center the point cloud
    centroid = np.mean(xyz, axis=0)
    xyz = xyz - centroid
    
    # Scale to unit sphere
    m = np.max(np.sqrt(np.sum(xyz**2, axis=1)))
    if m > 0:
        xyz = xyz / m
    
    # Combine normalized xyz with other features
    if other_features is not None:
        pc = np.concatenate((xyz, other_features), axis=1)
    else:
        pc = xyz
    
    return pc

def load_custom_point_cloud(npy_path, use_color=True):
    """
    Load and preprocess a custom point cloud from .npy file
    """
    print(f"Loading point cloud from: {npy_path}")
    
    # Load the point cloud
    point_cloud = np.load(npy_path)
    print(f"  Original shape: {point_cloud.shape}")
    
    # Ensure we have the right format (N, 6) for xyz+rgb
    if point_cloud.shape[1] < 6 and use_color:
        # If no color, add zeros
        zeros_rgb = np.zeros((point_cloud.shape[0], 3))
        point_cloud = np.concatenate([point_cloud[:, :3], zeros_rgb], axis=1)
        print("  No RGB values found, adding zeros")
    elif point_cloud.shape[1] >= 6:
        # Take only xyz and rgb
        point_cloud = point_cloud[:, :6]
    
    # Normalize xyz coordinates
    point_cloud = pc_normalize(point_cloud)
    
    # Ensure RGB values are in [0, 1]
    if use_color and point_cloud.shape[1] >= 6:
        rgb = point_cloud[:, 3:6]
        if np.max(rgb) > 1.0:
            point_cloud[:, 3:6] = rgb / 255.0
            print("  Normalized RGB from [0-255] to [0-1]")
    
    # Sample to 8192 points if necessary
    if point_cloud.shape[0] > 8192:
        indices = np.random.choice(point_cloud.shape[0], 8192, replace=False)
        point_cloud = point_cloud[indices]
        print(f"  Downsampled to 8192 points")
    elif point_cloud.shape[0] < 8192:
        # Repeat points if we have too few
        repeats = 8192 // point_cloud.shape[0] + 1
        point_cloud = np.tile(point_cloud, (repeats, 1))[:8192]
        print(f"  Upsampled to 8192 points")
    
    print(f"  Final shape: {point_cloud.shape}")
    
    return point_cloud

def visualize_point_cloud(point_cloud, save_path, title="Point Cloud"):
    """
    Create and save a 3D visualization of the point cloud
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract xyz and rgb
    xyz = point_cloud[:, :3]
    
    # Use RGB if available, otherwise use default color
    if point_cloud.shape[1] >= 6:
        colors = point_cloud[:, 3:6]
    else:
        colors = np.ones((point_cloud.shape[0], 3)) * 0.5
    
    # Plot points
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], 
              c=colors, s=1, alpha=0.8)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Set equal aspect ratio
    max_range = np.array([xyz[:, 0].max()-xyz[:, 0].min(),
                          xyz[:, 1].max()-xyz[:, 1].min(),
                          xyz[:, 2].max()-xyz[:, 2].min()]).max() / 2.0
    
    mid_x = (xyz[:, 0].max()+xyz[:, 0].min()) * 0.5
    mid_y = (xyz[:, 1].max()+xyz[:, 1].min()) * 0.5
    mid_z = (xyz[:, 2].max()+xyz[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Save the figure
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Visualization saved to: {save_path}")

def load_prompts(prompt_file):
    """
    Load prompts from a JSON file
    """
    with open(prompt_file, 'r') as f:
        prompts_data = json.load(f)
    return prompts_data

def init_model(model_path, device='cuda'):
    """
    Initialize PointLLM model
    """
    print(f"Initializing model from: {model_path}")
    
    disable_torch_init()
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Determine torch dtype and device
    if device == 'cuda' and torch.cuda.is_available():
        torch_dtype = torch.float16
        device = 'cuda'
    else:
        torch_dtype = torch.float32
        device = 'cpu'
    
    model = PointLLMLlamaForCausalLM.from_pretrained(
        model_path, 
        low_cpu_mem_usage=True, 
        use_cache=True, 
        torch_dtype=torch_dtype
    )
    
    model = model.to(device)
    model.initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer)
    model.eval()
    
    # Get point configuration
    mm_use_point_start_end = getattr(model.config, "mm_use_point_start_end", False)
    point_backbone_config = model.get_model().point_backbone_config
    
    # Setup conversation template
    conv_mode = "vicuna_v1_1"
    conv = conv_templates[conv_mode].copy()
    
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    
    return model, tokenizer, point_backbone_config, keywords, mm_use_point_start_end, conv, device

def run_inference(model, tokenizer, point_cloud, prompt_text, 
                 point_backbone_config, keywords, mm_use_point_start_end, conv, device):
    """
    Run inference on a single point cloud with a given prompt
    """
    # Reset conversation
    conv.reset()
    
    # Prepare point cloud tokens
    point_token_len = point_backbone_config['point_token_len']
    default_point_patch_token = point_backbone_config['default_point_patch_token']
    default_point_start_token = point_backbone_config['default_point_start_token']
    default_point_end_token = point_backbone_config['default_point_end_token']
    
    # Prepare the question with point tokens
    if mm_use_point_start_end:
        qs = default_point_start_token + default_point_patch_token * point_token_len + \
             default_point_end_token + '\n' + prompt_text
    else:
        qs = default_point_patch_token * point_token_len + '\n' + prompt_text
    
    # Add to conversation
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    # Tokenize
    inputs = tokenizer([prompt])
    input_ids = torch.as_tensor(inputs.input_ids).to(device)
    
    # Prepare point cloud tensor
    point_cloud_tensor = torch.from_numpy(point_cloud).unsqueeze(0).to(device).to(model.dtype)
    
    # Setup stopping criteria
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    stop_str = keywords[0]
    
    # Generate response
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            point_clouds=point_cloud_tensor,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            max_length=2048,
            top_p=0.95,
            repetition_penalty=1.1,
            stopping_criteria=[stopping_criteria]
        )
    
    # Decode output
    input_token_len = input_ids.shape[1]
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    
    outputs = outputs.strip()
    
    return outputs

def create_excel_with_images(df, image_paths, output_path):
    """
    Create an Excel file with embedded images
    """
    # Create Excel writer
    writer = pd.ExcelWriter(output_path, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Results', index=False)
    
    # Get workbook and worksheet
    workbook = writer.book
    worksheet = writer.sheets['Results']
    
    # Set column widths
    worksheet.set_column('A:A', 15)  # num_id
    worksheet.set_column('B:B', 30)  # image column
    
    # Set wider width for prompt result columns
    num_prompts = len([col for col in df.columns if col.startswith('Prompt')])
    for i in range(num_prompts):
        worksheet.set_column(i+2, i+2, 50)
    
    # Set row heights for images
    for i in range(2, len(df) + 2):  # Starting from row 2 (after headers)
        worksheet.set_row(i, 150)  # Height for image rows
    
    # Insert images
    for idx, img_path in enumerate(image_paths):
        if os.path.exists(img_path):
            # Insert image at row idx+2 (accounting for header and prompt definition rows)
            worksheet.insert_image(idx + 2, 1, img_path, 
                                 {'x_scale': 0.3, 'y_scale': 0.3, 'positioning': 1})
    
    # Format headers
    header_format = workbook.add_format({
        'bold': True,
        'text_wrap': True,
        'valign': 'top',
        'fg_color': '#D7E4BD',
        'border': 1
    })
    
    # Format cells
    cell_format = workbook.add_format({
        'text_wrap': True,
        'valign': 'top',
        'border': 1
    })
    
    # Apply formatting to prompt definition row
    for col in range(len(df.columns)):
        worksheet.write(1, col, df.iloc[0, col] if col > 1 else '', cell_format)
    
    # Close the writer
    writer.close()
    
    print(f"Excel file with images saved to: {output_path}")

def main(args):
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load prompts
    prompts_data = load_prompts(args.prompt_file)
    prompt_keys = list(prompts_data['prompts'].keys())
    
    # Initialize model
    model, tokenizer, point_backbone_config, keywords, mm_use_point_start_end, conv, device = \
        init_model(args.model_path, args.device)
    
    # Find all .npy files
    npy_files = glob.glob(os.path.join(args.data_path, '*.npy'))
    print(f"\nFound {len(npy_files)} .npy files")
    
    # Prepare data for DataFrame
    results = []
    image_paths = []
    
    # Create visualization directory
    viz_dir = os.path.join(args.output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Process each point cloud
    for idx, npy_path in enumerate(npy_files):
        filename = os.path.basename(npy_path)
        object_name = os.path.splitext(filename)[0]
        
        print(f"\n[{idx+1}/{len(npy_files)}] Processing: {object_name}")
        
        # Load point cloud
        try:
            point_cloud = load_custom_point_cloud(npy_path, use_color=True)
        except Exception as e:
            print(f"  Error loading point cloud: {e}")
            continue
        
        # Create visualization
        viz_path = os.path.join(viz_dir, f"{object_name}.png")
        visualize_point_cloud(point_cloud, viz_path, title=object_name)
        image_paths.append(viz_path)
        
        # Run inference for each prompt
        row_data = {
            'num_id': idx + 1,
            'object_image': ''  # Placeholder for image
        }
        
        for prompt_key in prompt_keys:
            prompt_text = prompts_data['prompts'][prompt_key]['full_prompt']
            print(f"  Running inference for {prompt_key}...")
            
            try:
                response = run_inference(
                    model, tokenizer, point_cloud, prompt_text,
                    point_backbone_config, keywords, mm_use_point_start_end, conv, device
                )
                row_data[prompt_key] = response
                print(f"    Response: {response[:100]}...")
            except Exception as e:
                print(f"    Error during inference: {e}")
                row_data[prompt_key] = f"Error: {str(e)}"
        
        results.append(row_data)
        
        # Save intermediate results
        if (idx + 1) % 5 == 0:
            temp_df = pd.DataFrame(results)
            temp_df.to_csv(os.path.join(args.output_dir, 'temp_results.csv'), index=False)
    
    # Create final DataFrame
    df = pd.DataFrame(results)
    
    # Add prompt definition row at the top
    prompt_definitions = {'num_id': '', 'object_image': ''}
    for prompt_key in prompt_keys:
        prompt_definitions[prompt_key] = prompts_data['prompts'][prompt_key]['definition']
    
    # Insert prompt definitions as first row
    df = pd.concat([pd.DataFrame([prompt_definitions]), df], ignore_index=True)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save CSV version
    csv_path = os.path.join(args.output_dir, f'evaluation_results_{timestamp}.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nCSV results saved to: {csv_path}")
    
    # Save Excel version with images
    excel_path = os.path.join(args.output_dir, f'evaluation_results_{timestamp}.xlsx')
    create_excel_with_images(df, image_paths, excel_path)
    
    # Save summary statistics
    summary = {
        'total_objects': len(results),
        'prompts_used': prompt_keys,
        'model_path': args.model_path,
        'timestamp': timestamp
    }
    
    with open(os.path.join(args.output_dir, f'evaluation_summary_{timestamp}.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*50)
    print("Evaluation completed successfully!")
    print(f"Results saved to: {args.output_dir}")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate PointLLM on custom point clouds")
    
    parser.add_argument("--model_path", type=str, 
                       default="RunsenXu/PointLLM_7B_v1.2",
                       help="Path to PointLLM model")
    
    parser.add_argument("--data_path", type=str,
                       default="PointLLM/pointllm/data/npy",
                       help="Path to directory containing .npy files")
    
    parser.add_argument("--prompt_file", type=str,
                       default="prompts.json",
                       help="Path to JSON file containing prompts")
    
    parser.add_argument("--output_dir", type=str,
                       default="evaluation_results",
                       help="Directory to save results")
    
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "cpu"],
                       help="Device to use for inference")
    
    args = parser.parse_args()
    
    main(args)
