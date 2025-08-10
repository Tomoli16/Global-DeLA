#!/usr/bin/env python3
"""
Example script showing how to configure the training run with different settings.

This demonstrates how to use the configuration functions to set up different
experimental runs with various combinations of Flash Attention, Mamba2, and run IDs.
"""

from config import configure_flash_attention, configure_mamba2, configure_run

# Example 1: Basic Mamba2 experiment
print("=== Example 1: Basic Mamba2 ===")
configure_run("mamba2_basic")
configure_mamba2("default")
configure_flash_attention("disabled")
print("Run ID: mamba2_basic")
print("Mamba2: enabled (default)")
print("Flash Attention: disabled")
print()

# Example 2: Heavy Mamba2 + Flash Attention
print("=== Example 2: Heavy Mamba2 + Flash Attention ===")
configure_run("heavy_experiment")
configure_mamba2("heavy")
configure_flash_attention("default")
print("Run ID: heavy_experiment")
print("Mamba2: enabled (heavy)")
print("Flash Attention: enabled (default)")
print()

# Example 3: Light experiment for fast training
print("=== Example 3: Light experiment ===")
configure_run("light_fast")
configure_mamba2("light")
configure_flash_attention("light")
print("Run ID: light_fast")
print("Mamba2: enabled (light)")
print("Flash Attention: enabled (light)")
print()

# Example 4: Traditional LFP only (baseline)
print("=== Example 4: Traditional baseline ===")
configure_run("baseline_lfp")
configure_mamba2("disabled")
configure_flash_attention("disabled")
print("Run ID: baseline_lfp")
print("Mamba2: disabled")
print("Flash Attention: disabled")
print()

print("To run with these configurations, import and call the configure functions")
print("before importing other modules in your training script, or use command line arguments:")
print("python train.py --run_id mamba2_basic")
