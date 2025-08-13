# Running DeepSeek Models on Alliance Canada's Narval Cluster
- [Running DeepSeek Models on Alliance Canada's Narval Cluster](#running-deepseek-models-on-alliance-canadas-narval-cluster)
  - [Phase 1. Initial Setup and Environment Preparation](#phase-1-initial-setup-and-environment-preparation)
    - [Step 1: Connect and Configure Environment](#step-1-connect-and-configure-environment)
    - [Step 2: Set Up Directory Structure](#step-2-set-up-directory-structure)
    - [Step 3: Create Python Virtual Environment](#step-3-create-python-virtual-environment)
    - [Step 4: Chose, and download a model:](#step-4-chose-and-download-a-model)
  - [Phase 2: Creating SLURM Job Scripts](#phase-2-creating-slurm-job-scripts)
    - [Step 5: Basic Inference Job Script](#step-5-basic-inference-job-script)
    - [Step 6: Create your inference script:](#step-6-create-your-inference-script)
    - [Step 7:](#step-7)
  - [Phase 3: Job Submission \& Management](#phase-3-job-submission--management)
    - [Step 8: Submit your job](#step-8-submit-your-job)
  - [Troubleshooting \& Optimization](#troubleshooting--optimization)
    - [Common Issues and Solutions](#common-issues-and-solutions)

## Phase 1. Initial Setup and Environment Preparation
### Step 1: Connect and Configure Environment
*On Narval:*
```bash
# Clear any existing modules, and load standard environment
module purge
module load StdEnv/2023


# Load python & Cuda
module load python/3.11
module load cuda/12.3

```
### Step 2: Set Up Directory Structure
```bash
# In your project directory, create a workspace
mkdir deepseek-project/{models,scripts,logs,cache}
cd deepseek-project


# Also set the cache directory to avoid home quota overflow
export HF_HOME"/scratch/$USER/.cache/huggingface"
mkdr -p $HF_HOME
```
### Step 3: Create Python Virtual Environment
*Setup a persistant virtual environment, with all the required packages*
```bash

# Create a virtual environment for vLLM
cd ~/projects/def-yourgroup/deepseek-project

# Use virtualenv
virtualenv --no-download vllm-env
source vllm-env/bin/activate

# Your prompt should show (vllm-env), meaning you are in the isolated environment

# Install packages from Alliance wheelhouse
pip install --no-index --upgrade pip

# Install packages for DeepSeek
pip install --no-index torch torchvision torchaudio
pip install --no-index vllm
pip install --no-index accelerate safetensors
pip install --no-index huggingface-hub # For downloading the models

# Deactivate for normal terminal work
deactivate

```

### Step 4: Chose, and download a model:
| Model    | Repository      | Allocation Guidelines |
| ------------- | ------------- | ------------- |
| 1.5B Param, for smaller resource requirements | git clone https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | 1 GPU, 6 CPUs, 40GB RAM |
| 14B model (balanced capability and resources) | git clone https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B| 2 GPUs, 12 CPUs, 80GB RAM |
| Instruct, for coding tasks | git clone https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Instruct |  |
|32B model, for larger capabilities | git clone https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B| 4 GPUs, 24 CPUs, 160GB RAM |

```bash
# Setup the cache directory: 
export HF_HOME="/scratch/$USER/.cache/huggingface"

# Download the model using HuggingFace CLI, around ~26GB total
huggingface-cli download \
    deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
    --cache-dir $HF_HOME \
    --resume-download

# Create a link to the model:
MODEL_DIR=$(find $HF_HOME -path "*/snapshots/*" -type d | head -1)
ln -s "$MODEL_DIR" ~/projects/def-yourgroup/deepseek-project/mode


# Take a look to make sure they downloaded
ls -la ~/projects/def-yourgroup/deepseek-project/model/
```

## Phase 2: Creating SLURM Job Scripts
### Step 5: Basic Inference Job Script
*Create a job script for running inference with your model*

*In your scripts folder*
```bash
cd ~/projects/def-yourgroup/deepseek-project/scripts
```

```bash
nano deepseek_inference.sh
```

```bash
#!/bin/bash
#SBATCH --job-name=deepseek-14b
#SBATCH --account=def-yourgroup           # Update this to your account
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --export=ALL,DISABLE_DCGM=1       # Required for Narval
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# Load environment
module load StdEnv/2023 python/3.11 cuda/12.2
source /project/def-yourgroup/deepseek-project/vllm-env/bin/activate

# Set offline mode
export HF_HUB_OFFLINE=1
export HF_HOME="/scratch/$USER/.cache/huggingface"

# Copy model to local storage for faster loading
cp -r /project/def-yourgroup/deepseek-project/model $SLURM_TMPDIR/deepseek-model

# Run inference
python inference_vllm.py --model_path $SLURM_TMPDIR/deepseek-model
```

*Make the script executable*
```bash
chmod +x deepseek_inference.sh
```
*Remember to update --account=def-yourgroup with your actual account name*


### Step 6: Create your inference script:
*Create a python script that will run your specific tasks:*

*In your scripts directory*
```bash
cd ~/projects/def-yourgroup/deepseek-project/scripts
```

```bash
nano inference_vllm.py
```

```bash
#!/usr/bin/env python3
"""
DeepSeek-R1-Distill-Qwen-14B Inference using vLLM
Framework script for running DeepSeek models on Narval
"""

import argparse
import time
from vllm import LLM, SamplingParams

def load_model(model_path):
    """Load DeepSeek model with vLLM"""
    print(f"Loading model from: {model_path}")
    
    llm = LLM(
        model=model_path,
        tensor_parallel_size=2,      # Use both GPUs  
        gpu_memory_utilization=0.85,
        max_model_len=8192,         # Adjust based on your needs
        dtype="bfloat16",
        trust_remote_code=True,
        enforce_eager=True
    )
    
    print("Model loaded successfully")
    return llm

def run_inference(llm, prompts):
    """Run inference on provided prompts"""
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=512,
        stop=["Human:", "Assistant:"]
    )
    
    print(f"Processing {len(prompts)} prompts...")
    start_time = time.time()
    
    outputs = llm.generate(prompts, sampling_params)
    
    inference_time = time.time() - start_time
    print(f"Inference completed in {inference_time:.2f} seconds")
    
    return outputs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to model directory")
    args = parser.parse_args()
    
    # Load the model
    llm = load_model(args.model_path)
    
    # Example prompts - modify these for your use case
    prompts = [
        "Explain machine learning in simple terms:",
        "Write a Python function to calculate factorial:",
        "What are the benefits of using HPC for AI research?"
    ]
    
    # Run inference
    outputs = run_inference(llm, prompts)
    
    # Display results
    for i, output in enumerate(outputs):
        print(f"\n--- Result {i+1} ---")
        print(f"Prompt: {output.prompt}")
        print(f"Response: {output.outputs[0].text}")

if __name__ == "__main__":
    main()
EOF
```


### Step 7: 
## Phase 3: Job Submission & Management
### Step 8: Submit your job
*Make sure your scripts are properly configured*

*In your scripts dir:*
`cd ~/projects/def-yourgroup/deepseek-project/scripts`

*Submit:*
`sbatch deepseek_inference.sh`

*Check job status:*
`squeue -u $USER`

## Troubleshooting & Optimization
### Common Issues and Solutions
1. Module Loading Errors
   ```bash
    # Always start with a clean environment
    module purge
    module load StdEnv/2023  # Load this first
    module load python/3.11 gcc/12.3 opencv/4.11
    ```
2. Model Loading Issues\
    *Verify model files are complete, '.safetensors' files should be GB, not bytes*
    ```bash
    # If files are incomplete, re-download
    git lfs fetch -all
    git reset --hard HEAD
    git lfs pull
    ```
3. GPU Memory Issues\
    *Reduce the memory usage percentage in your python script*
    ```bash
    llm = LLM(
        model=model_path,
        gpu_memory_utilization=0.75,  # Reduce from 0.85
        max_model_len=16384,          # Reduce context length if acceptable
        tensor_parallel_size=2
    )
    ```
4. Job Timeout Issues:

    *For longer jobs, increase time limit and use checkpointing*\
    `#SBATCH --time=12:00:00`


    




















