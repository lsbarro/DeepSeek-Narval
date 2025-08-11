# Running DeepSeek Models on Alliance Canada's Narval Cluster
- [Running DeepSeek Models on Alliance Canada's Narval Cluster](#running-deepseek-models-on-alliance-canadas-narval-cluster)
  - [Phase 1. Initial Setup and Environment Preparation](#phase-1-initial-setup-and-environment-preparation)
    - [Step 1: Connect and Configure Environment](#step-1-connect-and-configure-environment)
    - [Step 2: Set Up Directory Structure](#step-2-set-up-directory-structure)
    - [Step 3: Chose, and download a model:](#step-3-chose-and-download-a-model)
    - [Step 4: Create Python Virtual Environment](#step-4-create-python-virtual-environment)
  - [Phase 2: Creating SLURM Job Scripts](#phase-2-creating-slurm-job-scripts)
    - [Step 5: Basic Inference Job Script](#step-5-basic-inference-job-script)
    - [Step 6: Create your inference script:](#step-6-create-your-inference-script)
    - [Step 7: API Server Job Script (Optional)](#step-7-api-server-job-script-optional)
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
```
### Step 2: Set Up Directory Structure
```bash
# In your project directory, create a workspace
mkdir deepseek-workspace
cd deepseek-workspace

# Create a subdirectory for organization
mkdir logs jobs
```
### Step 3: Chose, and download a model:
| Model    | Repository      | Allocation Guidelines |
| ------------- | ------------- | ------------- |
| 1.5B Param, for smaller resource requirements | git clone https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | 1 GPU, 6 CPUs, 40GB RAM |
| 14B model (balanced capability and resources) | git clone https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B| 2 GPUs, 12 CPUs, 80GB RAM |
| Instruct, for coding tasks | git clone https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Instruct |  |
|32B model, for larger capabilities | git clone https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B| 4 GPUs, 24 CPUs, 160GB RAM |

```bash
# Load Git LFS module for downloading large model files
module load git-lfs/3.4.0
git lfs install

# Clone the model repository (using 14B model as example)
git clone https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B

# Enter the model directory, and download the actual model
cd DeepSeek-R1-Distill-Qwen-14B
git lfs pull
```
*Make sure to verify the size of your downloaded models, there should be four '.safetensors' files totaling ~26GB, not bytes.*

### Step 4: Create Python Virtual Environment
*Setup a persistant virtual environment, with all the required packages*
```bash
# Back in your workspace (/project/def-yourgroup/deepseek-workspace)

# Load OpenCV module before creating virtual environment
module load gcc/12.3 opencv/4.11

# Create a virtual env
virtualenv --no-download deepseek_env
source deepseek_env/bin/activate

# Install required packages from wheelhouse
pip install --no-index --upgrade pip
pip install --no-index torch torchvision torchaudio
pip install --no-index transformers safetensors tokenizers

# Optional: Install additional packages for specific use cases
pip install --no-index fastapi uvicorn  # For API servers
pip install --no-index datasets trl     # For fine-tuning workflows

# Deactivate for now
deactivate
```

## Phase 2: Creating SLURM Job Scripts
### Step 5: Basic Inference Job Script
*Create a job script for running inference with your model*
```bash
# In your jobs directory (/project/def-yourgroup/deepseek-workspace/jobs)
nano deepseek_inference.sh
```
*Write contets `deepseek_inference.sh`:*
```bash
#!/bin/bash
#SBATCH --job-name=deepseek-inference
#SBATCH --account=def-yourgroup          # Use your account
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=a100:2             # Request appropriate # of GPU's, 2 for 14B model
#SBATCH --cpus-per-task=12               # Request appropriate # of CPU's, 12 for 14B model
#SBATCH --mem=80G                        # Adjust based on model size
#SBATCH --time=3:00:00                   # Adjust based on your needs
#SBATCH --output=../logs/inference_%j.out
#SBATCH --error=../logs/inference_%j.err

echo "DeepSeek Inference Started"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"

# Load Software Environment in Sequence
module load StdEnv/2023
module load python/3.11 gcc/12.3 opencv/4.11

echo "Setting up job-specific environment"
# Create job-specific venv in a temporary directory
virtualenv --no-download $SLURM_TMPDIR/deepseek_env
source $SLURM_TMPDIR/deepseek_env/bin/activate

# Install required packages in the job environment
pip install --no-index --upgrade pip
pip install --no-index torch vllm>=0.10.0 transformers safetensors

echo "Setting up model path for direct access"
# Use model directly from shared storage
export MODEL_PATH="/project/def-yourgroup/deepseek-workspace/DeepSeek-R1-Distill-Qwen-14B"

# Set cache directories BEFORE offline mode
export HF_HOME="/scratch/$USER/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"

# Set variables to indicate offline mode, this is required as narval compute nodes are offline
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# GPU optimization settings
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "GPU Info:"
nvidia-smi

echo "Starting Inference"
# Run your inference script (create this in the next step)
python $SLURM_SUBMIT_DIR/inference_script.py

echo "Job completed at $(date)"
```

### Step 6: Create your inference script:
*Create a python script that will run your specific tasks:*
```bash
#in your jobs directory (/project/def-yourgroup/deepseek-workspace/jobs)
nano inference_script.py
```
*Write contents of `inference_script.py`:*
```python
#!/usr/bin/env python3
"""
Deepseek Inference Script, modify for usecase
"""

import os
from vllm import LLM, SamplingParams

def main():
    # Use model directly from shared storage (set via MODEL_PATH environment variable)
    model_path = os.environ.get('MODEL_PATH', '/project/def-yourgroup/deepseek-workspace/DeepSeek-R1-Distill-Qwen-14B')
    
    print(f"Loading model from: {model_path}")
    
    # Initialize vLLM with configuration optimized for your model
    llm = LLM(
        model=model_path,
        tensor_parallel_size=2,           # Match the number of GPUs in your job
        gpu_memory_utilization=0.85,      # Use 85% of GPU memory
        max_model_len=32768,              # Full context length for DeepSeek models
        trust_remote_code=True,           # Required for DeepSeek models
        load_format='safetensors',        # Explicit format specification
        dtype='half'                      # Use FP16 for memory efficiency
    )
    
    print("Model loaded!")
    
    # Configure sampling parameters for your use case
    sampling_params = SamplingParams(
        temperature=0.7,                  # Adjust for creativity vs consistency
        top_p=0.95,                       # Nucleus sampling parameter
        max_tokens=1024,                  # Maximum response length
        stop=["Human:", "Assistant:", "<|im_end|>"]  # Stop sequences
    )
    
    # Define your custom prompts or load from file
    prompts = [
        "Explain the concept of machine learning in simple terms.",
        "Write a Python function to calculate the factorial of a number.",
        "What are the main differences between supervised and unsupervised learning?",
        # Add your own prompts here
    ]
    
    print(f"Processing {len(prompts)} prompts...")
    
    # Generate responses
    outputs = llm.generate(prompts, sampling_params)
    
    # Process and display results
    for i, output in enumerate(outputs, 1):
        print(f"\n{'='*60}")
        print(f"PROMPT {i}")
        print(f"{'='*60}")
        print(f"Input: {output.prompt}")
        print(f"\nResponse: {output.outputs[0].text}")
        print(f"Finish reason: {output.outputs[0].finish_reason}")
        
        # Optional: Save results to file
        # You can modify this section to save results in your preferred format
    
    print(f"\nCompleted processing all {len(prompts)} prompts")

if __name__ == "__main__":
    main()
```

### Step 7: API Server Job Script (Optional)
*For interactive applications, you will need to create a server that provides an API interface:*
```bash
nano deepseek_server.sh
```
*Write contents of `deepseek_server.sh`, replacing necessary values as before*
```bash
#!/bin/bash
#SBATCH --job-name=deepseek-server
#SBATCH --account=def-yourgroup          # Replace with your account
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=a100:2
#SBATCH --cpus-per-task=12               # Request appropriate # of CPU's, 12 for 14B model
#SBATCH --mem=80G
#SBATCH --time=8:00:00                   # Longer duration for server applications
#SBATCH --output=../logs/server_%j.out
#SBATCH --error=../logs/server_%j.err

echo "DeepSeek API server setup"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"

# Set up env
module load StdEnv/2023
module load python/3.11 gcc/12.3 opencv/4.11

virtualenv --no-download $SLURM_TMPDIR/deepseek_env
source $SLURM_TMPDIR/deepseek_env/bin/activate
pip install --no-index vllm>=0.10.0 fastapi uvicorn transformers

# Use model directly from shared storage & set offline mode
export MODEL_PATH="/project/def-yourgroup/deepseek-workspace/DeepSeek-R1-Distill-Qwen-14B"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# Set cache directories for consistency
export HF_HOME="/scratch/$USER/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"

# Display server connection information
NODE_IP=$(hostname -I | awk '{print $1}')
echo "   Server Access Information:"
echo "   Base URL: http://$NODE_IP:8000"
echo "   API Documentation: http://$NODE_IP:8000/docs"
echo "   Health Check: http://$NODE_IP:8000/health"

# Start OpenAI-compatible API server
vllm serve $MODEL_PATH \
    --tensor-parallel-size 2 \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.85 \
    --trust-remote-code \
    --load-format safetensors \
    --dtype half

echo "Server stopped at $(date)"
```

## Phase 3: Job Submission & Management
### Step 8: Submit your job
*Make sure your scripts are properly configured*
```bash
# Make your job scripts executable
chmod +x deepseek_inference.sh deepseek_server.sh

# In your jobs dir (/project/def-yourgroup/deepseek-workspace/jobs) submit your inference job
sbatch deepseek_inference.sh

# Check job status
squeue -u $USER
```

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


    




















