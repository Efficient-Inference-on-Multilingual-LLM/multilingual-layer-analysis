#!/bin/bash

set -e
export BASE_CACHE="/mnt/disks/vm2/.cache"
mkdir -p "$BASE_CACHE/huggingface" "$BASE_CACHE/tmp" "$BASE_CACHE/vllm"

# 2. FORCE ENVIRONMENT OVERRIDES AT THE SYSTEM LEVEL
# This covers HF, vLLM, and General Linux Cache defaults
export HF_HOME="$BASE_CACHE/huggingface"
export XDG_CACHE_HOME="$BASE_CACHE"
export TMPDIR="$BASE_CACHE/tmp"
export VLLM_CACHE="$BASE_CACHE/vllm"
export VLLM_USE_V1=0
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export TORCHINDUCTOR_CACHE_DIR="$BASE_CACHE/torchinductor"
export PYTHONPYCACHEPREFIX="$BASE_CACHE/pycache"

# 1. SETUP WORKING DIRECTORY
WORKING_DIR="$(cd "$(dirname "$0")/.." && pwd)"
echo "🚀 Working directory: $WORKING_DIR"
cd "$WORKING_DIR"

if [[ -d "./risqi" ]]; then
    echo "✅ Activating uv venv 'risqi'..."
    source "./risqi/bin/activate"
elif [[ -n "$VIRTUAL_ENV" ]]; then
    echo "✅ Using existing virtual environment: $VIRTUAL_ENV"
    source "$VIRTUAL_ENV/bin/activate"
elif [[ -f "./venv/bin/activate" ]]; then   
    echo "✅ Activating standard local venv..."
    source "./venv/bin/activate"
else
    echo "❌ Error: Could not find 'risqi' venv. Please run 'uv venv risqi' first."
    exit 1
fi

# 3. INITIALIZE VARIABLES
entrypoint=""
extra_args=()
model_list=""
selection_mode=""

usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -e, --extract           Run neuron activation extraction"
    echo "  -p, --plot              Run visualization/plotting mode"
    echo "  -i, --intervene         Run neuron ablation intervention"
    echo "  -m, --mode MODE         Selection algorithm: 'entropy' (default) or 'threshold'"
    echo "  --models LIST           Comma-separated list of model names"
    echo "  --top_rate VAL          Top K entropy rate (default 0.01)"
    echo "  --component TYPE        Component type: 'proj_up' or 'mlp_out'"
    echo "  --data_dir PATH         Path to input data"
    echo "  --save_path PATH        Path to save results"
    exit 1
}

# 4. PARSE ARGUMENTS
SHORT_OPTS="eiph"
LONG_OPTS="extract,plot,intervene,mode:,models:,top_rate:,component:,data_dir:,save_path:,help"

PARSED_OPTS=$(getopt --options $SHORT_OPTS --longoptions $LONG_OPTS --name "$0" -- "$@")
eval set -- "$PARSED_OPTS"

while true; do
    case "$1" in
        -e|--extract)   entrypoint="src.main.run_extraction"; shift ;;
        -p|--plot)      entrypoint="src.main.run_plotting"; shift ;;
        -i|--intervene) entrypoint="src.main.run_intervention"; shift ;;
        -m|--mode)      selection_mode="$2"; shift 2 ;;
        --models)       model_list="$2"; shift 2 ;;
        --top_rate)     extra_args+=("processing.top_rate=$2"); shift 2 ;;
        --component)    extra_args+=("processing.component_type=$2"); shift 2 ;;
        --data_dir)     extra_args+=("paths.data_dir=$2"); shift 2 ;;
        --save_path)    extra_args+=("paths.output_dir=$2"); shift 2 ;;
        --help)         usage ;;
        --) shift; break ;;
        *) usage ;;
    esac
done

if [[ -z "$selection_mode" ]]; then
    selection_mode="entropy"
fi


# Masukkan selection_mode ke dalam extra_args untuk Hydra
extra_args+=("processing.selection_mode=${selection_mode}")

# 5. EXECUTION
if [[ -z "$entrypoint" ]]; then usage; fi

multirun_args=""
if [[ -n "$model_list" ]]; then
    multirun_args="--multirun model.name=${model_list}"
fi

CMD="python3 -m $entrypoint $multirun_args ${extra_args[*]}"
echo "-----------------------------------"
echo "🛠️  Algorithm: $selection_mode"
echo "🛠️  Executing: $CMD"
eval $CMD