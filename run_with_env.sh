#!/bin/bash
export CUDA_VISIBLE_DEVICES=""
export JULIA_PKG_PRESERVE_TIERED_INSTALLED=true

# Detect if we're on Apple Silicon
if [[ "$OSTYPE" == "darwin"* ]]; then
    # Check if it's Apple Silicon
    if [[ $(uname -m) == "arm64" ]]; then
        # Optimized for Apple Silicon M4
        export JULIA_CPU_TARGET="native"
        # Alternative: specify Apple Silicon targets explicitly
        # export JULIA_CPU_TARGET="apple-m1;apple-m2;apple-m3"
    else
        # Intel Mac
        export JULIA_CPU_TARGET="generic;
sandybridge,clone_all;
ivybridge,-xsaveopt,clone_all;
haswell,-rdrnd,base(1)"
    fi
else
    # Linux/other systems - keep original x86 targets
    export JULIA_CPU_TARGET="generic;
sandybridge,clone_all;
ivybridge,-xsaveopt,clone_all;
haswell,-rdrnd,base(1)"
fi

# Determine the number of CPU cores
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS-specific core count
    CPU_CORES=$(sysctl -n hw.logicalcpu)
else
    # Linux-style core count
    CPU_CORES=$(nproc)
fi

export JULIA_NUM_PRECOMPILE_TASKS=$((CPU_CORES - 1))


julia --project=@. --threads=auto --gcthreads=4 -i
