"""
Geometric Dataset PointLLM Inference with Multi-Stage Chain-of-Thought Reasoning
Original import structure preserved - uses pointllm.model.PointLLMLlamaForCausalLM
"""

import argparse
import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import re
import torch

# PointLLM imports will be added at runtime
POINTLLM_AVAILABLE = False


def setup_pointllm_path(pointllm_path: str = "PointLLM"):
    """Setup PointLLM module path"""
    global POINTLLM_AVAILABLE
    
    if pointllm_path not in sys.path:
        sys.path.insert(0, pointllm_path)
    
    try:
        from pointllm.conversation import conv_templates, SeparatorStyle
        from pointllm.utils import disable_torch_init
        from pointllm.model import PointLLMLlamaForCausalLM
        from pointllm.model.utils import KeywordsStoppingCriteria
        from transformers import AutoTokenizer
        POINTLLM_AVAILABLE = True
        print("[INFO] PointLLM modules loaded successfully")
        return True
    except ImportError as e:
        print(f"[Error] Failed to import PointLLM modules: {e}")
        print(f"[Info] Please ensure PointLLM is installed at: {pointllm_path}")
        return False


def pc_normalize(pc: np.ndarray) -> np.ndarray:
    """Normalize point cloud to unit sphere"""
    xyz = pc[:, :3]
    other_features = pc[:, 3:] if pc.shape[1] > 3 else None
    
    centroid = np.mean(xyz, axis=0)
    xyz = xyz - centroid
    
    m = np.max(np.sqrt(np.sum(xyz**2, axis=1)))
    if m > 0:
        xyz = xyz / m
    
    if other_features is not None:
        pc = np.concatenate((xyz, other_features), axis=1)
    else:
        pc = xyz
    
    return pc


def load_point_cloud_npy(npy_path: str, use_color: bool = True, target_points: int = 8192) -> np.ndarray:
    """Load and preprocess point cloud from .npy file"""
    point_cloud = np.load(npy_path)
    
    if point_cloud.shape[1] < 6 and use_color:
        zeros_rgb = np.zeros((point_cloud.shape[0], 3))
        point_cloud = np.concatenate([point_cloud[:, :3], zeros_rgb], axis=1)
    elif point_cloud.shape[1] >= 6:
        point_cloud = point_cloud[:, :6]
    
    # Sampling
    if len(point_cloud) > target_points:
        indices = np.random.choice(len(point_cloud), target_points, replace=False)
        point_cloud = point_cloud[indices]
    
    # Normalize
    point_cloud = pc_normalize(point_cloud)
    
    return point_cloud


# ============================================================================
# Prompt Templates (All in English)
# ============================================================================

def generate_cot_prompt(question: str) -> str:
    """
    Generate prompt for CoT step generation.
    Model should output ONLY numbered reasoning steps, NO direct answers.
    """
    return f"""You are a highly analytical assistant specialized in 3D point cloud processing and **Chain of Thought (CoT) generation**. Your SOLE purpose is to generate the detailed, step-by-step reasoning process required to answer the user's question, without providing the final answer.

## 🎯 Primary Goal: Strict CoT Generation

You are analyzing a 3D point cloud. Each object in the scene is defined by its distinct color.

Question: {question}

TASK: Break down this question into a step-by-step Chain of Thought (CoT) reasoning process.

## 🚫 CRITICAL INSTRUCTIONS (STRICT ADHERENCE REQUIRED)

1.  **PENALTY WARNING:** IF you provide any part of the final numerical or factual answer, or any conclusion, the entire output will be rejected. **YOUR ONLY JOB IS TO REASON.**
2.  **OUTPUT FORMAT:** Output **ONLY** the CoT steps in the numbered list format: (1., 2., 3., etc.).
3.  **DO NOT** include the final answer, any analysis (e.g., "The result is...", "There are two ...-colored objects in the given point cloud"), or any concluding statements.
4.  **FOCUS:** Each step must be a specific, actionable observation or computational step based on the point cloud data (e.g., "Extract all points...", "Calculate the centroid...").

Example CoT format:
CoT:
1. 
2. 
3. 
4. 

Now provide your CoT steps for the question above (numbered list only, NO answers):
"""


def generate_cot_step_prompt(cot_steps: List[str], 
                             previous_qa: List[Tuple[str, str]], 
                             current_step: str,
                             current_step_num: int) -> str:
    """Generate prompt for executing a specific CoT step"""
    context = "CONTEXT: Each object in this 3D point cloud scene is defined by its distinct color."
    
    previous_context = ""
    if previous_qa:
        previous_context = "\n\nPrevious Chain of Thought steps:\n"
        for i, (step, answer) in enumerate(previous_qa, 1):
            previous_context += f"Step {i}: {step}\nObservation: {answer}\n\n"
    
    prompt = f"""{context}
{previous_context}
Current step to analyze (Step {current_step_num}):
{current_step}

TASK: Provide a detailed observation for this specific step only. Focus on what you see in the point cloud.

Your observation:
"""
    return prompt


def generate_final_answer_prompt(question: str, 
                                 cot_qa_pairs: List[Tuple[str, str]]) -> str:
    """Generate prompt for final answer generation"""
    context = "CONTEXT: Each object in this 3D point cloud scene is defined by its distinct color."
    
    cot_context = "\n\nChain of Thought Analysis (completed steps):\n"
    for i, (step, answer) in enumerate(cot_qa_pairs, 1):
        cot_context += f"Step {i}: {step}\nObservation: {answer}\n\n"
    
    prompt = f"""{context}
{cot_context}
Original Question: {question}

TASK: Based on the Chain of Thought analysis above, provide your final answer to the question. YOU MUST ANSWER BASED SOLELY ON THE OBSERVATIONS FROM THE CoT STEPS.
Answer with a focus on text rather than point clouds.
ANSWER:

Your final answer:
"""
    return prompt


# ============================================================================
# CoT Parser
# ============================================================================

def parse_cot_steps(cot_output: str) -> List[str]:
    """Parse CoT output to extract numbered steps"""
    steps = []
    pattern = r'^\s*(\d+)\.\s*(.+)$'
    
    for line in cot_output.strip().split('\n'):
        match = re.match(pattern, line.strip())
        if match:
            step_text = match.group(2).strip()
            if step_text:
                steps.append(step_text)
    
    # Fallback: split by newlines if no numbered format found
    if not steps:
        steps = [line.strip() for line in cot_output.strip().split('\n') 
                if line.strip() and not line.strip().startswith('CoT:')]
    
    return steps


# ============================================================================
# Multi-Stage CoT Inference Engine
# ============================================================================

class MultiStageCoTInference:
    """Multi-stage Chain-of-Thought inference pipeline for PointLLM"""
    
    def __init__(self, model_path: str, device: str = "mps", pointllm_path: str = "PointLLM"):
        """Initialize PointLLM model with original structure"""
        
        # Setup PointLLM path
        if not setup_pointllm_path(pointllm_path):
            raise ImportError("Failed to setup PointLLM")
        
        # Import PointLLM modules (EXACTLY as in original code)
        from pointllm.conversation import conv_templates, SeparatorStyle
        from pointllm.utils import disable_torch_init
        from pointllm.model import PointLLMLlamaForCausalLM
        from pointllm.model.utils import KeywordsStoppingCriteria
        from transformers import AutoTokenizer
        
        self.conv_templates = conv_templates
        self.SeparatorStyle = SeparatorStyle
        self.KeywordsStoppingCriteria = KeywordsStoppingCriteria
        
        print(f"[INFO] Initializing model from: {model_path}")
        
        disable_torch_init()
        
        # Load tokenizer and model (EXACTLY as in original code)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Determine torch dtype and device
        if device == 'cuda' and torch.cuda.is_available():
            torch_dtype = torch.float16
            self.device = 'cuda'
        elif device == 'mps' and torch.backends.mps.is_available():
            torch_dtype = torch.float16
            self.device = 'mps'
        else:
            torch_dtype = torch.float32
            self.device = 'cpu'
        
        print(f"[INFO] Using device: {self.device}, dtype: {torch_dtype}")
        
        self.model = PointLLMLlamaForCausalLM.from_pretrained(
            model_path, 
            low_cpu_mem_usage=True, 
            use_cache=True, 
            torch_dtype=torch_dtype
        )
        
        self.model = self.model.to(self.device)
        self.model.initialize_tokenizer_point_backbone_config_wo_embedding(self.tokenizer)
        self.model.eval()
        
        # Get point configuration
        self.mm_use_point_start_end = getattr(self.model.config, "mm_use_point_start_end", False)
        self.point_backbone_config = self.model.get_model().point_backbone_config
        
        # Setup conversation template
        self.conv_mode = "vicuna_v1_1"
        self.conv = self.conv_templates[self.conv_mode].copy()
        
        self.stop_str = self.conv.sep if self.conv.sep_style != self.SeparatorStyle.TWO else self.conv.sep2
        self.keywords = [self.stop_str]
        
        print("[INFO] Model initialized successfully")
    
    def run_single_inference(self, point_cloud: np.ndarray, prompt: str) -> str:
        """
        Run a single inference with PointLLM.
        (Copied EXACTLY from original code)
        """
        conv = self.conv_templates[self.conv_mode].copy()
        
        # Prepare point cloud tokens
        point_token_len = self.point_backbone_config['point_token_len']
        default_point_patch_token = self.point_backbone_config['default_point_patch_token']
        default_point_start_token = self.point_backbone_config['default_point_start_token']
        default_point_end_token = self.point_backbone_config['default_point_end_token']
        
        # Prepare the question with point tokens
        if self.mm_use_point_start_end:
            qs = default_point_start_token + default_point_patch_token * point_token_len + \
                 default_point_end_token + '\n' + prompt
        else:
            qs = default_point_patch_token * point_token_len + '\n' + prompt
        
        # Add to conversation
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt_text = conv.get_prompt()
        
        # Tokenize
        inputs = self.tokenizer([prompt_text])
        input_ids = torch.as_tensor(inputs.input_ids).to(self.device)
        
        # Prepare point cloud tensor
        point_cloud_tensor = torch.from_numpy(point_cloud).unsqueeze(0).to(self.device).to(self.model.dtype)
        
        # Generate
        stopping_criteria = self.KeywordsStoppingCriteria(self.keywords, self.tokenizer, input_ids)
        
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                point_clouds=point_cloud_tensor,
                do_sample=True,
                temperature=0.3,
                top_k=50,
                max_new_tokens=512,
                top_p=0.95,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                stopping_criteria=[stopping_criteria]
            )
        
        input_token_len = input_ids.shape[1]
        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        
        if outputs.endswith(self.stop_str):
            outputs = outputs[:-len(self.stop_str)]
        
        return outputs.strip()
    
    def run_multi_stage_inference(self, 
                                  point_cloud: np.ndarray, 
                                  question: str,
                                  answer_format: str = "[Your answer here]") -> Dict:
        """Run complete multi-stage CoT inference"""
        results = {
            "cot_steps": [],
            "cot_answers": [],
            "final_answer": "",
            "all_outputs": {}
        }
        
        # ===== Stage 1: Generate CoT Steps =====
        print("\n[Stage 1] Generating Chain of Thought steps...")
        cot_prompt = generate_cot_prompt(question)
        cot_output = self.run_single_inference(point_cloud, cot_prompt)
        results["all_outputs"]["cot_generation"] = cot_output
        print(f"CoT Output:\n{cot_output}\n")
        # Parse CoT steps
        cot_steps = parse_cot_steps(cot_output)
        results["cot_steps"] = cot_steps
        
        if not cot_steps:
            print("[Warning] No CoT steps parsed. Using default reasoning.")
            cot_steps = [
                "Identify all colors in the point cloud",
                "Analyze the geometry of each object",
                "Determine spatial relationships"
            ]
            results["cot_steps"] = cot_steps
        
        # print(f"Generated {len(cot_steps)} CoT steps:")
        # for i, step in enumerate(cot_steps, 1):
        #     print(f"  {i}. {step}")
        
        # ===== Stage 2: Execute Each CoT Step =====
        print(f"\n[Stage 2] Executing {len(cot_steps)} CoT steps...")
        cot_qa_pairs = []
        
        for i, step in enumerate(cot_steps, 1):
            print(f"  Executing Step {i}/{len(cot_steps)}...")
            
            step_prompt = generate_cot_step_prompt(
                cot_steps=cot_steps,
                previous_qa=cot_qa_pairs,
                current_step=step,
                current_step_num=i
            )
            
            step_answer = self.run_single_inference(point_cloud, step_prompt)
            results["all_outputs"][f"cot_step_{i}"] = step_answer
            
            cot_qa_pairs.append((step, step_answer))
            results["cot_answers"].append(step_answer)
            
            print(f"    Answer: {step_answer[:100]}..." if len(step_answer) > 100 else f"    Answer: {step_answer}")
        
        # ===== Stage 3: Generate Final Answer =====
        print("\n[Stage 3] Generating final answer...")
        final_prompt = generate_final_answer_prompt(
            question=question,
            cot_qa_pairs=cot_qa_pairs
        )
        
        final_answer = self.run_single_inference(point_cloud, final_prompt)
        results["final_answer"] = final_answer
        results["all_outputs"]["final_answer"] = final_answer
        
        print(f"Final Answer: {final_answer}")
        
        return results


# ============================================================================
# Dataset Processing
# ============================================================================

def find_all_scene_folders(base_dir: str) -> List[Dict[str, str]]:
    """Find all scene folders containing point_clouds_npy and metadata directories"""
    base_path = Path(base_dir)
    scene_folders = []
    
    for folder in base_path.rglob("*"):
        if folder.is_dir():
            npy_dir = folder / "point_clouds_npy"
            metadata_dir = folder / "metadata"
            
            if npy_dir.exists() and metadata_dir.exists():
                scene_folders.append({
                    "folder_path": str(folder),
                    "npy_dir": str(npy_dir),
                    "metadata_dir": str(metadata_dir)
                })
    
    return scene_folders


def process_single_scene(scene_info: Dict[str, str],
                        scene_id: int,
                        inference_engine: MultiStageCoTInference,
                        max_scenes: int = -1) -> Optional[List[Dict]]:
    """Process a single scene with all its QA pairs"""
    npy_path = Path(scene_info["npy_dir"]) / f"scene_{scene_id:05d}.npy"
    metadata_path = Path(scene_info["metadata_dir"]) / f"scene_{scene_id:05d}.json"
    
    if not npy_path.exists() or not metadata_path.exists():
        return None
    
    # Load point cloud
    point_cloud = load_point_cloud_npy(str(npy_path))
    
    # Load metadata
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    qa_pairs = metadata.get("qa_pairs", [])
    if not qa_pairs:
        return None
    
    results = []
    
    for qa_idx, qa in enumerate(qa_pairs):
        question = qa.get("question", "")
        ground_truth = qa.get("answer", "")
        qa_type = qa.get("type", "unknown")
        
        # Run multi-stage inference
        print(f"\n{'='*60}")
        print(f"Scene {scene_id}, QA {qa_idx+1}/{len(qa_pairs)}")
        print(f"Question: {question[:100]}...")
        
        inference_results = inference_engine.run_multi_stage_inference(
            point_cloud=point_cloud,
            question=question
        )
        
        results.append({
            "scene_id": scene_id,
            "qa_index": qa_idx,
            "qa_type": qa_type,
            "question": question,
            "ground_truth": ground_truth,
            "cot_steps": "; ".join(inference_results["cot_steps"]),
            "cot_answers": "; ".join(inference_results["cot_answers"]),
            "predicted_answer": inference_results["final_answer"]
        })
    
    return results


def process_folder(scene_info: Dict[str, str],
                   inference_engine: MultiStageCoTInference,
                   max_scenes: int = -1) -> pd.DataFrame:
    """Process all scenes in a folder"""
    npy_files = sorted(Path(scene_info["npy_dir"]).glob("scene_*.npy"))
    
    if max_scenes > 0:
        npy_files = npy_files[:max_scenes]
    
    all_results = []
    
    for npy_file in tqdm(npy_files, desc=f"Processing {Path(scene_info['folder_path']).name}"):
        scene_id = int(npy_file.stem.split('_')[-1])
        
        scene_results = process_single_scene(
            scene_info=scene_info,
            scene_id=scene_id,
            inference_engine=inference_engine,
            max_scenes=max_scenes
        )
        
        if scene_results:
            all_results.extend(scene_results)
    
    return pd.DataFrame(all_results)


# ============================================================================
# Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PointLLM Geometric Dataset Inference with Multi-Stage CoT"
    )
    
    parser.add_argument("--base_dir", type=str, required=True,
                       help="Base directory containing geometric_datasets")
    parser.add_argument("--model_path", type=str, default="RunsenXu/PointLLM_7B_v1.2",
                       help="Path to PointLLM model")
    parser.add_argument("--pointllm_path", type=str, default="PointLLM",
                       help="Path to PointLLM repository")
    parser.add_argument("--device", type=str, default="mps", choices=["cuda", "cpu", "mps"],
                       help="Device to use for inference")
    parser.add_argument("--max_scenes", type=int, default=-1,
                       help="Maximum scenes per folder (-1 for all)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("PointLLM Multi-Stage Chain-of-Thought Inference")
    print("=" * 60)
    print(f"Base directory: {args.base_dir}")
    print(f"Model: {args.model_path}")
    print(f"Device: {args.device}")
    print(f"PointLLM path: {args.pointllm_path}")
    print(f"Max scenes per folder: {args.max_scenes if args.max_scenes > 0 else 'All'}")
    
    # Initialize PointLLM
    print("\n[1] Loading PointLLM model...")
    inference_engine = MultiStageCoTInference(
        model_path=args.model_path,
        device=args.device,
        pointllm_path=args.pointllm_path
    )
    
    # Find all scene folders
    print("\n[2] Scanning for scene folders...")
    scene_folders = find_all_scene_folders(args.base_dir)
    print(f"Found {len(scene_folders)} scene folders")
    
    # Process each folder
    print("\n[3] Running multi-stage inference...")
    
    for scene_info in scene_folders:
        folder_name = Path(scene_info["folder_path"]).name
        print(f"\n--- Processing: {folder_name} ---")
        
        results_df = process_folder(
            scene_info=scene_info,
            inference_engine=inference_engine,
            max_scenes=args.max_scenes
        )
        
        # Save results
        results_dir = Path(scene_info["folder_path"]) / "results"
        results_dir.mkdir(exist_ok=True)
        
        output_csv = results_dir / "inference_results_cot.csv"
        results_df.to_csv(output_csv, index=False)
        
        print(f"Saved results to: {output_csv}")
        print(f"Processed {len(results_df)} QA pairs")
    
    print("\n" + "=" * 60)
    print("Inference Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()