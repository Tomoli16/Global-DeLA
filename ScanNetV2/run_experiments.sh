#!/bin/bash

# Run experiments with different model configurations

echo "Starting DeLA ScanNetV2 experiments..."

# Run with traditional DeLA model
echo "Running with traditional DeLA model..."
python train.py --model dela_semseg --run_id dela_base

# Run with Flash Attention model - light preset
echo "Running with Flash Attention model (light)..."
python train.py --model dela_semseg_attn --flash_attn_preset light --run_id dela_attn_light

# Run with Flash Attention model - default preset
echo "Running with Flash Attention model (default)..."
python train.py --model dela_semseg_attn --flash_attn_preset default --run_id dela_attn_default

# Run with Flash Attention model - heavy preset
echo "Running with Flash Attention model (heavy)..."
python train.py --model dela_semseg_attn --flash_attn_preset heavy --run_id dela_attn_heavy

echo "All experiments started!"
