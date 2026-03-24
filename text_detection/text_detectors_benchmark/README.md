# Text Detector Benchmarks

## Overview

This directory contains the scripts used to evaluate the temporal and computational performance of the text detectors integrated into the preprocessing pipeline.

## Table of Contents

| File / Directory | Type | Description |
| :--- | :--- | :--- |
| `benchmark_fast.py` | Script | Performance evaluation for the FAST model. |
| `benchmark_mixnet.py` | Script | Performance evaluation for the MixNet model. |
| `benchmark_pan_pp.py` | Script | Performance evaluation for the PAN++ model. |
| `README.md` | Doc | This documentation. |

## Usage

The scripts in this directory are generally executed through the main script located in the parent directory (`run_text_detectors_benchmark.sh`). Each script initializes the corresponding model and measures inference times.
