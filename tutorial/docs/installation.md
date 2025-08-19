# Running DeepSeek Models on Alliance Canada's Narval Cluster

- [Running DeepSeek Models on Alliance Canada's Narval Cluster](#running-deepseek-models-on-alliance-canadas-narval-cluster)
  - [Phase 1: Initial Setup and Environment Preparation](#phase-1-initial-setup-and-environment-preparation)
    - [Step 1: Setup environment](#step-1-setup-environment)
    - [Step 2: Set up Your Working Directory](#step-2-set-up-your-working-directory)
    - [Step 3: Create Python Virtual Environment](#step-3-create-python-virtual-environment)
    - [Step 4: Download DeepSeek Model](#step-4-download-deepseek-model)
  - [Phase 2: Create SLURM Job Scripts](#phase-2-create-slurm-job-scripts)
    - [Step 5: Main Job Script](#step-5-main-job-script)
    - [Step 6: Python Inference Script](#step-6-python-inference-script)
  - [Phase 3: Job Submission and Management](#phase-3-job-submission-and-management)
    - [Step 7: Submit and Monitor Job](#step-7-submit-and-monitor-job)
  - [Troubleshooting Guide](#troubleshooting-guide)
    - [Common Issues and Verified Solutions](#common-issues-and-verified-solutions)
    - [Performance Expectations](#performance-expectations)


## Phase 1: Initial Setup and Environment Preparation

### Step 1: Setup environment

**Load modules in this order**
```bash
# Always start clean
module purge

# Load standard environment
module load StdEnv/2023

# Load dependencies in this order
module load opencv/4.11 python/3.12
module load cuda/12.2
```

### Step 2: Set up Your Working Directory

**Set up your workspace (REPLACE def-yourgroup with actual group name):**
```bash
# In your project directory
cd ~/projects/def-yourgroup

# Create organized directory structure
mkdir -p deepseek-project/{models,scripts,logs,cache}
cd deepseek-project

# Configure HuggingFace cache in scratch space (for large models)
export HF_HOME="/scratch/$USER/.cache/huggingface"
mkdir -p $HF_HOME

# Orient yourself
pwd
ls -la
```

### Step 3: Create Python Virtual Environment

```bash
# Create virtual environment in current project directory
virtualenv --no-download ./vllm-env

# Activate the environment
source ./vllm-env/bin/activate

# The prompt should show (vllm-env) at this point
```

**Install required packages using Alliance wheels:**
```bash
# Install in order
pip install --no-index --upgrade pip
pip install --no-index huggingface_hub
pip install --no-index vllm
pip install --no-index accelerate safetensors transformers


pip install --no-index pillow
pip install --no-index opencv-python-headless
pip install --no-index pyzmq
pip install --no-index scipy
pip install --no-index six

```

### Step 4: Download DeepSeek Model

**Download model on login node:**
```bash
# Ensure virtual environment is active
source ./vllm-env/bin/activate

# Download DeepSeek-R1-Distill-Qwen-14B (~26GB)
huggingface-cli download \
    deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
    --cache-dir $HF_HOME \
    --resume-download

# SafeTensors files should be several GB each, NOT just bytes
# If files show only ~135 bytes, Git LFS didn't download properly
```

## Phase 2: Create SLURM Job Scripts

### Step 5: Main Job Script

**Create the job submission script (make sure to fill your group into the project directory path in this script):**
```bash
cd scripts
nano deepseek_inference.sh
```

**SLURM script:**
```bash
#!/bin/bash
#SBATCH --job-name=deepseek-14b
#SBATCH --account=def-yourgroup               # REPLACE with your actual group
#SBATCH --gpus-per-task=2                     
#SBATCH --cpus-per-task=12                    # Max 12 CPUs per GPU on Narval
#SBATCH --mem=248G                            # 124GB per GPU × 2 GPUs
#SBATCH --time=06:00:00
#SBATCH --export=ALL,DISABLE_DCGM=1          
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "SLURM Job ID: $SLURM_JOB_ID"

# Load environment in order
module purge
module load StdEnv/2023
module load opencv/4.11 python/3.12
module load cuda/12

# Set offline mode for compute nodes
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="/scratch/$USER/.cache/huggingface"

# Get absolute path to project directory
PROJECT_DIR="$(cd ~/projects/def-yourgroup/deepseek-project && pwd)"
echo "Project directory: $PROJECT_DIR"

# Activate virtual environment
source "$PROJECT_DIR/vllm-env/bin/activate"

# Simple GPU verification instead of DCGM check
echo "Verifying GPU accessibility..."
nvidia-smi > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "GPUs accessible - proceeding"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "ERROR: Cannot access GPUs - check SLURM allocation"
    exit 1
fi

# Copy model to node-attached storage for faster performance, scratch has slow I/O request speed, and copying then accessing is many times faster.
echo "Copying model to local storage..."
MODEL_CACHE_DIR="$HF_HOME/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-14B"
if [ -d "$MODEL_CACHE_DIR" ]; then
    cp -r "$MODEL_CACHE_DIR" "$SLURM_TMPDIR/"
    MODEL_PATH="$SLURM_TMPDIR/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-14B"
    echo "Model copied to: $MODEL_PATH"
else
    echo "ERROR: Model not found at $MODEL_CACHE_DIR"
    exit 1
fi

# Run inference
echo "Starting DeepSeek inference..."
python "$PROJECT_DIR/scripts/inference_vllm.py" --model_path "$MODEL_PATH"

echo "Job completed at: $(date)"
```

**Make script executable:**
```bash
chmod +x deepseek_inference.sh
```

### Step 6: Python Inference Script

**Create the inference script:**
```bash
nano inference_vllm.py
```

```python
#!/usr/bin/env python3
"""
DeepSeek-R1-Distill-Qwen-14B Inference using vLLM
"""

import argparse
import time
import os
import sys
from pathlib import Path

try:
    from vllm import LLM, SamplingParams
except ImportError:
    print("ERROR: vLLM not installed")
    sys.exit(1)

def find_model_directory(model_path):
    """
    Find the actual model directory, handling various download structures.
    """
    model_path = Path(model_path)
    
    # Direct model path (already contains model files)
    if (model_path / "config.json").exists():
        return str(model_path)
    
    # HuggingFace cache structure with snapshots
    snapshots_dir = model_path / "snapshots"
    if snapshots_dir.exists():
        snapshot_dirs = [d for d in snapshots_dir.iterdir() if d.is_dir()]
        if snapshot_dirs:
            # Use the first (usually only) snapshot
            actual_path = snapshot_dirs[0]
            if (actual_path / "config.json").exists():
                return str(actual_path)
    
    raise ValueError(
        f"Could not find valid model directory in {model_path}. "
        f"Expected either config.json directly or in snapshots subdirectory."
    )

def load_model(model_path):
    """Load DeepSeek model with vLLMs"""
    print(f"Loading model from: {model_path}")
    
    actual_model_path = find_model_directory(model_path)
    print(f"Using model files from: {actual_model_path}")
    
    # vLLM configuration optimized for Narval A100 GPUs
    llm = LLM(
        model=actual_model_path,
        tensor_parallel_size=2,          # Use both allocated GPUs
        gpu_memory_utilization=0.9,      # Use 90% of GPU memory
        max_model_len=4096,              # Context length
        dtype="bfloat16",                # Efficient precision for A100
        trust_remote_code=True,          # Required for DeepSeek
        enforce_eager=True,              # Better compatibility on HPC
        disable_log_stats=True           # Reduce log verbosity
    )
    
    print("Model loaded successfully")
    return llm

def run_inference(llm, prompts):
    """Run inference with DeepSeek-optimized parameters"""
    # DeepSeek-specific optimal parameters from research
    sampling_params = SamplingParams(
        temperature=0.6,                 # Optimal for DeepSeek reasoning
        top_p=0.95,                     # Optimal for DeepSeek quality
        max_tokens=512,
        stop=["<|User|>", "<|Assistant|>", "\n\nUser:", "\n\nAssistant:"]
    )
    
    print(f"Processing {len(prompts)} prompts...")
    start_time = time.time()
    
    outputs = llm.generate(prompts, sampling_params)
    
    inference_time = time.time() - start_time
    tokens_per_second = sum(len(output.outputs[0].text.split()) for output in outputs) / inference_time
    print(f"Inference completed in {inference_time:.2f} seconds")
    print(f"Throughput: {tokens_per_second:.1f} tokens/second")
    
    return outputs

def main():
    parser = argparse.ArgumentParser(description="DeepSeek inference on Alliance Narval")
    parser.add_argument("--model_path", required=True, help="Path to model directory")
    parser.add_argument("--prompts_file", help="File containing prompts (one per line)")
    args = parser.parse_args()
    
    # Load the model
    try:
        llm = load_model(args.model_path)
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return 1
    
    # Get prompts
    if args.prompts_file and os.path.exists(args.prompts_file):
        with open(args.prompts_file, 'r') as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        # Default test prompts with proper DeepSeek formatting
        prompts = [
            "<|User|>Explain machine learning in simple terms:<|Assistant|>",
            "<|User|>Write a Python function to calculate factorial:<|Assistant|>",
            "<|User|>What are the benefits of using HPC for AI research?<|Assistant|>"
        ]
    
    # Run inference
    try:
        outputs = run_inference(llm, prompts)
    except Exception as e:
        print(f"ERROR during inference: {e}")
        return 1
    
    # Display results
    print("\n" + "="*60)
    print("INFERENCE RESULTS")
    print("="*60)
    
    for i, output in enumerate(outputs):
        print(f"\n--- Result {i+1} ---")
        print(f"Prompt: {output.prompt}")
        print(f"Response: {output.outputs[0].text}")
        print("-" * 50)
    
    print(f"\nSuccessfully processed {len(outputs)} prompts")
    return 0

if __name__ == "__main__":
    sys.exit(main())
```

## Phase 3: Job Submission and Management

### Step 7: Submit and Monitor Job

**Submit the job:**
```bash
cd scripts
sbatch deepseek_inference.sh
```

**Monitor job progress:**
```bash
# Check job status
squeue -u $USER

# Follow output in real-time (replace JOBID with actual ID)
tail -f deepseek-14b-JOBID.out

# Check for errors
tail -f deepseek-14b-JOBID.err
```

## Troubleshooting Guide

### Common Issues and Verified Solutions

**1. Module Loading Failures**
```bash
# Symptom: "Module not found" errors
# Solution: Verify correct order and availability
module avail python
module avail opencv
# Always load: opencv/4.11 python/3.12 (in that order)
```

**2. Account Name Errors**
```bash
# Symptom: "Invalid account" in SLURM
# Solution: Find your exact account name
sacctmgr show user $USER -s | grep def-
# Use the exact string shown (e.g., def-smithj, not def-smith)
```

**3. Virtual Environment Issues**
```bash
# Symptom: Package import errors
# Solution: Recreate environment
rm -rf ./vllm-env
module load opencv/4.11 python/3.12
virtualenv --no-download ./vllm-env
source ./vllm-env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index vllm huggingface_hub
```

**4. Model Loading Failures**
```bash
# Symptom: "No such file or directory" for model
# Solution: Verify Git LFS completion
find $HF_HOME -name "*.safetensors" -size -1M
# If any files are tiny, re-download:
cd $(find $HF_HOME -name "*.git" -type d | head -1 | xargs dirname)
git lfs pull
```

**5. GPU Allocation Issues**
```bash
# Symptom: Job doesn't start or GPU errors
# Solution: Check resource availability
sinfo -p gpu
# Verify DCGM disable worked:
dcgmi -v 2>/dev/null || echo "DCGM properly disabled"
```

### Performance Expectations

**On Narval with 2×A100 GPUs, expect:**
- Model loading: 2-5 minutes
- Inference speed: 50-100 tokens/second
- Memory usage: ~80% of 40GB per GPU
- Total job time: 10-15 minutes for test prompts