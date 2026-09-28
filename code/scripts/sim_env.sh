#!/usr/bin/env bash
# Environment setup for EnergyPlus + Radiance simulators (Path-C).
# Source this file before running Path-C scripts.
export ENERGYPLUS_HOME="$HOME/local/EnergyPlus-26.1.0-6f2e40d102-Darwin-macOS13-arm64"
export RADIANCE_HOME="$HOME/local/radiance"
export RAYPATH=".:$RADIANCE_HOME/lib"
export PATH="$ENERGYPLUS_HOME:$RADIANCE_HOME/bin:$PATH"
