# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SHUD (Simulator for Hydrologic Unstructured Domains) is a physically-based, multi-process distributed hydrological model using the Finite Volume Method (FVM). It simulates coupled surface/subsurface flow with SUNDIALS/CVODE as the ODE solver.

**Tech Stack**: C++14, SUNDIALS/CVODE v6.x, OpenMP (optional), Python (validation)

## Build Commands

```bash
# Prerequisites: install SUNDIALS/CVODE
./configure

# Compile (serial version)
make clean && make shud

# Compile with OpenMP
make shud_omp

# Run example watershed
./shud ccw
```

## Testing

```bash
# Python unit tests (TSR validation framework)
python3 -m unittest discover -s validation/tsr/py -p 'test_*.py'

# With coverage (must pass ≥90%)
python3 -m coverage run -m unittest discover -s validation/tsr/py -p 'test_*.py'
python3 -m coverage report --fail-under=90

# TSR pointwise comparison (C++ vs Python reference)
bash validation/tsr/run_tsr.sh
python3 validation/tsr/py/compare_tsr.py output/ccw.tsr --tol 1e-10
```

## Architecture

### State Vector Layout (Macros.hpp)
Global ODE state vector `y` with size `NY = 3*NumEle + NumRiv + NumLake`:
- `y[i]` — Surface ponding (SF) for element i
- `y[NumEle + i]` — Unsaturated zone (US)
- `y[2*NumEle + i]` — Groundwater (GW)
- `y[3*NumEle + i]` — River stage (RIV)
- `y[3*NumEle + NumRiv + i]` — Lake stage (LAKE)

### Core Modules

| Directory | Purpose |
|-----------|---------|
| `src/Model/` | Main driver loop (`shud.cpp`), RHS dispatcher (`f.cpp`) |
| `src/ModelData/` | Data structures, RHS implementation (`MD_f.cpp`, `MD_f_omp.cpp`), forcing updates, ET |
| `src/classes/` | Domain objects: `Element`, `River`, `Lake`, `TimeSeriesData`, I/O |
| `src/Equations/` | Physical process functions, CVODE config, TSR module |
| `validation/tsr/` | Python validation framework for Terrain Solar Radiation |

### Key Files

- `src/Model/shud.cpp` — Main integration loop, forcing updates, ET control
- `src/ModelData/MD_f.cpp` — Serial RHS implementation (`f_update`, `f_loop`, `f_applyDY`)
- `src/ModelData/MD_f_omp.cpp` — OpenMP-parallelized RHS
- `src/classes/Element.cpp` — Element hydraulics, soil parameter updates
- `src/Model/Macros.hpp` — State vector indices, physical constants, global flags

### Physical Processes

- **Element vertical**: precipitation → interception → infiltration → recharge → ET
- **Element lateral**: overland flow, saturated lateral flow between neighbors
- **River**: segment-based routing, channel hydraulics, downstream propagation
- **Lake**: water balance model (v2.0+)
- **TSR**: Terrain Solar Radiation correction (v2.1+)

## Development Notes

### Modifying RHS Physics
Changes to RHS calculations must be synchronized across both implementations:
- `src/ModelData/MD_f.cpp` (serial)
- `src/ModelData/MD_f_omp.cpp` (OpenMP)

### Global Flags (Macros.hpp)
- `global_implicit_mode` — Solver mode
- `global_verbose_mode` — Debug output
- `lakeon` — Enable lake module

### Input/Output
- Input: `.para` (config), `.tsd` (forcing), `.att` (element attributes), `.riv` (rivers), `.ic` (initial conditions)
- Output: Binary `.dat` files with 1024-byte header + time series

### GPU Acceleration (Planned)
See `docs/GPU加速方案.md` for detailed design. Key principle: RHS kernels must run entirely on device; no host/device data transfer during RHS callbacks.

## Domain Knowledge

Understanding of watershed hydrology concepts required:
- Overland flow, groundwater recharge, channel routing
- Finite Volume Method for mass conservation
- CVODE implicit/explicit ODE integration with Krylov linear solvers
