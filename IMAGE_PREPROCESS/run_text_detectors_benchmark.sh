#!/usr/bin/env bash
DIR="$(cd "$(dirname "$0")" && pwd)"
CSV="../STATS/text-detectors-benchmark.csv"
PY="conda run -n TFG-Preprocess python"

echo "model,params_M,flops_G,pth_MB,onnx_MB" > "$CSV"

run() {
    local label="$1" script="$2"
    echo "Running $label..."
    output=$($PY "$script")
    echo "$output" | grep -v "^CSV:" | sed 's/^/  /'
    csv_line=$(echo "$output" | grep "^CSV:" | sed 's/^CSV://')
    echo "$label,$csv_line" >> "$CSV"
}

run "PAN++ R18"      "$DIR/text_detectors_benchmark/benchmark_pan_pp.py"
run "FAST-B-736"     "$DIR/text_detectors_benchmark/benchmark_fast.py"
run "MixNet FSNet-M" "$DIR/text_detectors_benchmark/benchmark_mixnet.py"

echo ""
column -t -s',' "$CSV"