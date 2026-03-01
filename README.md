# PointLLM × Apple Vision Pro — Dental 3D Analysis System

An end-to-end system that captures hand-tracking and 3D point cloud data from **Apple Vision Pro**, sends it to a **Mac mini inference server** running PointLLM, and returns natural-language analysis of dental anatomical structures.

---

## Repository Structure

```
.
├── app/                    # VisionOS application (Apple Vision Pro)
├── server/                 # FastAPI inference server (Mac mini)
└── PointLLMFinetuning/    # Fine-tuning scripts for PointLLM
```

---

## 1. VisionOS App (Main)

The Apple Vision Pro app captures hand-tracking data and 3D point cloud interactions in real time, then sends them to the inference server for PointLLM analysis.

### Requirements

- Apple Vision Pro with visionOS 2.0+
- Xcode 16+
- Mac on the same local network as the inference server

### Setup

1. Open `app/DataProvider.xcodeproj` in Xcode.
2. Set your development team in **Signing & Capabilities**.
3. Update the server IP address in `DataProvider/NetworkManager.swift`:

```swift
private let serverBaseURL = "http://<your-mac-mini-ip>:8000"
```

4. Build and run on Apple Vision Pro.

### Usage

1. Launch the app on Apple Vision Pro.
2. In the main window, enter the question you want to ask (e.g., *"What structure is being indicated?"*).
3. Tap **Record** to capture a single frame snapshot.
   - The app waits up to 2 seconds for hand detection before capturing.
   - Hand tracking samples at **10 Hz** with 27 joints per hand (wrist, thumb, index, middle, ring, little fingers, forearm).
4. The captured frame is sent to the server and the PointLLM response is displayed.

### Network Configuration

| Item | Value |
|------|-------|
| Server IP | `10.1.199.116` (update to your Mac mini's IP) |
| Port | `8000` |
| Timeout | 60 seconds |

### API Called by the App

```
POST http://<server-ip>:8000/api/process_tracking
Content-Type: application/json

{
  "session": {
    "sessionStartTime": "<ISO8601>",
    "frames": [ ... ],
    "metadata": {
      "deviceModel": "Apple Vision Pro",
      "markerColor": "blue"
    }
  },
  "question": "What anatomical structure is being indicated?"
}
```

### Captured Data Format

Each frame includes:
- **Device transform** (6-DoF pose)
- **Left / Right hand** joint positions (27 joints each)
- **3D object transforms** (dental model pose in AR world space)

---

## 2. Inference Server

A FastAPI server that runs on Mac mini and performs PointLLM inference on incoming point cloud + hand tracking data.

### Requirements

- macOS with Apple Silicon (MPS backend)
- Python 3.10+
- Conda or venv

### Installation

```bash
cd server

# Install dependencies
pip install -r requirements.txt

# Install the bundled PointLLM package
pip install -e PointLLM/
```

### Model Checkpoints

Place trained checkpoints under `server/PointLLM/pointllm/outputs/`.
The default paths configured in `server.py` are:

```
server/PointLLM/pointllm/outputs/
├── encoder_proj_tune_ddp_demo_ar_410_v1/
│   └── checkpoints/checkpoint-epoch-20/          # Stage 1 checkpoint
└── demo_410_ar_intration_encoder_proj_llm_ddp_finetune_.../
    └── checkpoints/best_model_epoch14_mashi/     # Stage 2 checkpoint
```

> Checkpoints are excluded from the repository (`.gitignore`). Download or train them separately.

### Running the Server

```bash
cd server
python server.py
```

The server starts on `0.0.0.0:8000`.

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Health check with memory info |
| `POST` | `/api/process_tracking` | Main inference endpoint |
| `GET` | `/api/model_info` | Model configuration |
| `POST` | `/api/clear_cache` | Manual MPS cache clear |

#### Health Check

```bash
curl http://localhost:8000/api/health
```

```json
{
  "status": "ok",
  "model_initialized": true,
  "memory": {
    "cpu_used_mb": 4200,
    "mps_allocated_mb": 9800
  }
}
```

### Point Cloud Processing Pipeline

```
Input (Vision Pro tracking data)
    ↓ Load .npy file [8192 pts × 6 channels]
    ↓ AR Space Conversion  (DICOM → AR World)
    ↓ Hand Interaction Colorization
    ↓ Head-Space Conversion (World → Head-relative)
    ↓ Normalization (center + unit sphere)
    ↓ PointLLM Inference (PointBERT Encoder → Projector → LLaMA 7B)
    ↓ Response
```

### Project Layout

```
server/
├── server.py               # FastAPI app entry point
├── pointllm_manager/       # Model loading & inference
│   ├── model_manager.py
│   ├── generate_response.py
│   └── modules/
├── pc_process/             # Point cloud preprocessing
│   ├── loader.py
│   └── pre_process/
├── data/
│   ├── usdz_pc/            # Point cloud files (.npy) — not in git
│   ├── class_info/         # Anatomical class metadata (.json)
│   └── results/            # Visualization outputs — not in git
└── PointLLM/               # PointLLM Python package (custom inference)
```

---

## 3. PointLLM Fine-tuning

Fine-tuning is performed in **two stages**.
All scripts are in `PointLLMFinetuning/`.

### Requirements

- CUDA-capable GPU(s) (tested on A100 40 GB)
- Python 3.10+
- Conda environment `pointllm_finetune`

```bash
cd PointLLMFinetuning
pip install -r PointLLM/pyproject.toml   # or install PointLLM package manually
```

### Dataset Format

```
your_dataset/
├── point_clouds/
│   ├── sample_001.npy     # shape: (N, 6), columns: [X, Y, Z, R, G, B]
│   └── ...
└── annotations/
    ├── train.json
    └── val.json
```

**annotation JSON structure:**

```json
{
  "data": [
    {
      "id": "sample_001",
      "point_cloud": "point_clouds/sample_001.npy",
      "conversations": [
        { "role": "user",      "content": "<point>\nWhat is this?" },
        { "role": "assistant", "content": "This is the upper left first molar." }
      ]
    }
  ]
}
```

> **Important:** User messages must include the `<point>` token.

Point cloud normalization:
- XYZ: centered at origin, scaled to unit sphere
- RGB: normalized to `[0.0, 1.0]`

---

### Stage 1 — Encoder + Projection Layer Fine-tuning (DDP)

Trains the **Point Encoder** and **Projection layer** while keeping LLaMA frozen.

**Script:** `encoder_proj_tune_ddp.py`
**Shell:** `run_encoder_proj_tune_dpp_training.sh`

```bash
bash run_encoder_proj_tune_dpp_training.sh [NUM_GPUS] [DATA_ROOT] [RESUME_PATH]

# Example (2 GPUs)
bash run_encoder_proj_tune_dpp_training.sh 2 /path/to/dataset
```

Or run directly with `torchrun`:

```bash
torchrun --nproc_per_node=2 encoder_proj_tune_ddp.py \
    --pointllm-path ./PointLLM \
    --data-root    /path/to/dataset
```

**Key hyperparameters (defaults):**

| Parameter | Value |
|-----------|-------|
| Base model | `RunsenXu/PointLLM_7B_v1.2` |
| Frozen modules | LLM |
| Trainable modules | Point Encoder + Projector |
| Epochs | 20 |
| Batch size (per GPU) | 32 |
| Gradient accumulation | 8 |
| Learning rate | `5e-4` |
| Scheduler | cosine |

Checkpoints are saved to:
```
PointLLM/pointllm/outputs/<experiment_name>/checkpoints/
├── checkpoint-epoch-X/
│   ├── point_proj.pt
│   └── trainer_state.pt
└── best_model/
```

---

### Stage 2 — Projection + LLM Fine-tuning (DDP)

Uses the Stage 1 checkpoint to further fine-tune the **Projection layer** and **LLaMA 7B** together.

**Script:** `encoder_proj_llm_finetune_ddp.py`
**Shell:** `run_proj_llm_ddp.sh`

```bash
bash run_proj_llm_ddp.sh [NUM_GPUS] [PRETRAINED_PATH] [DATA_ROOT] [RESUME_PATH]

# Example
bash run_proj_llm_ddp.sh 2 \
    PointLLM/pointllm/outputs/encoder_proj_tune_ddp_.../checkpoints/best_model \
    /path/to/dataset
```

Or with `torchrun`:

```bash
torchrun --nproc_per_node=2 encoder_proj_llm_finetune_ddp.py \
    --pointllm-path    ./PointLLM \
    --pretrained-path  PointLLM/pointllm/outputs/<stage1>/checkpoints/best_model \
    --data-root        /path/to/dataset
```

**Key hyperparameters (defaults):**

| Parameter | Value |
|-----------|-------|
| Frozen modules | Point Encoder |
| Trainable modules | Projector + LLaMA 7B |
| Epochs | 100 |
| Batch size | 16 |
| Gradient accumulation | 2 |
| Learning rate | `1e-5` |
| Scheduler | cosine |

---

### Training Monitoring

```bash
# TensorBoard
tensorboard --logdir PointLLM/pointllm/outputs/

# Loss curves are also saved as PNG
PointLLM/pointllm/outputs/<experiment_name>/logs/loss_curves.png
```

W&B logging can be enabled in `config.py` (`use_wandb: True`).

---

### Other Scripts (Legacy)

The following scripts are retained for reference but are **not part of the main training pipeline**:

| Script | Description |
|--------|-------------|
| `encoder_proj_tune.py` | Single-GPU encoder+proj tuning (non-DDP) |
| `encoder_proj_llm_finetune.py` | Single-GPU LLM finetune (non-DDP) |
| `encoder_proj_LoRA_finetune.py` | LoRA-based fine-tuning |
| `proj_tune_and_LoRA.py` | Projection + LoRA combined |
| `main_finetune.py` | General fine-tuning entry point |
| `run_encoder_proj_lora_finetune.sh` | LoRA run script |
| `run_main_training*.sh` | Old main training scripts |

---

## System Overview

```
┌─────────────────────────┐         ┌──────────────────────────────┐
│   Apple Vision Pro       │  JSON   │        Mac mini               │
│                          │ ──────► │  FastAPI Server (:8000)       │
│  - Hand tracking (27 jt) │         │  PointLLM 7B (MPS)           │
│  - 3D dental model pose  │ ◄────── │  Point Encoder + Projector   │
│  - Natural language query│  Text   │  + Fine-tuned LLaMA           │
└─────────────────────────┘         └──────────────────────────────┘
```
