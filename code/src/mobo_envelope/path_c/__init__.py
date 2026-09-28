"""Path-C: industry-standard simulator binaries for spot-check verification.

Uses real EnergyPlus 26.1 and Radiance 6.0.2 to re-evaluate Pareto designs
(the complete front, or a representative k-medoids subset). Slower than Path-B
but provides the external-simulator reference used in the building-performance
community.

Modules:
  - energyplus_runner:  IDF generation (eppy) + binary execution
  - radiance_runner:    .rad scene generation + three-phase daylight method
  - subset_selector:    pick ~20 representative designs per case
"""
