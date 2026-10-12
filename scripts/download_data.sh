#!/usr/bin/env bash
# Download the hourly CMEMS surface-current subset used in this project.
# Requires a free Copernicus Marine account: run `copernicusmarine login` once first.
#
# Product: GLOBAL_ANALYSISFORECAST_PHY_001_024, dataset cmems_mod_glo_phy_anfc_merged-uv_PT1H-i
# (hourly total surface current = circulation + tides + Stokes drift, 1/12 deg).
set -euo pipefail

copernicusmarine subset \
  --dataset-id cmems_mod_glo_phy_anfc_merged-uv_PT1H-i \
  --variable utotal --variable vtotal \
  --minimum-longitude -13.083 --maximum-longitude -8.5 \
  --minimum-latitude 36.25 --maximum-latitude 42.417 \
  --minimum-depth 0 --maximum-depth 1 \
  --start-datetime 2024-01-01T00:00:00 --end-datetime 2026-01-01T00:00:00 \
  --output-directory data \
  --output-filename cmems_iberia_uv_2024_2025.nc
