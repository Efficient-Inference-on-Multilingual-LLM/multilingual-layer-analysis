#!/bin/bash

set -e

WORKING_DIR="$(cd "$(dirname "$0")/.." && pwd)"
echo "Working directory: $WORKING_DIR"
cd "$WORKING_DIR"

if [[ -n "$VIRTUAL_ENV" ]]; then
    echo "Using virtual environment at $VIRTUAL_ENV"
    source "$VIRTUAL_ENV/bin/activate"
elif [[ -f "./venvs/bin/activate" ]]; then   
    echo "Activating virtual environment from ./venvs"
    source "./venvs/bin/activate"
else
    echo "No virtual environment detected."
    exit 1
fi

entrypoint=""
extra_args=()
model_list=""

usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -p                          Run in plotting mode"
    echo "  --models LIST               Comma-separated list of model names"
    echo "  --activation_path PATH      Path to the activation files"
    echo "  --save_path PATH            Path to save the plots"
    echo "  --help                      Display this help message"
    echo "Example:"
    echo "  $0 -p --models model1,model2 --activation_path /path/to/activations --save_path /path/to/save/plots"
    exit 1
}

SHORT_OPTS="p"
LONG_OPTS="activation_path:,save_path:,models:,help"

PARSED_OPTS=$(getopt --options $SHORT_OPTS --longoptions $LONG_OPTS --name "$0" -- "$@")
if [[ $? -ne 0 ]]; then
    usage
fi  

while [[ -n "$1" ]]; do
    case "$1" in
        -p)
            entrypoint="src.main.run_plotting"
            shift
            ;;
        --models)
            model_list="$2"
            shift 2
            ;;
        --activation_path)
            extra_args+=("--activation_path" "$2")
            shift 2
            ;;
        --save_path)
            extra_args+=("--save_path" "$2")
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Unexpected option: $1"
            usage
            ;;
    esac
done

multirun_args=""
if [[ -n "$model_list" ]]; then
    multirun_args="--multirun models=${model_list}"
fi
CMD="python3 -m $entrypoint $multirun_args ${extra_args[*]}"
echo "Running command: $CMD"
echo "-----------------------------------"
echo "Starting plotting"
eval $CMD
echo "Plotting completed"
echo "-----------------------------------"

