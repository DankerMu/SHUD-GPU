# Simulator for Hydrologic Unstructured Domains

## Brief

The Simulator for Hydrologic Unstructured Domains  (SHUD - pronounced “SHOULD”) is a multi-process, multi-scale hydrological model where major hydrological processes are fully coupled using the semi-discrete **Finite Volume Method** (FVM).

Ongoing applications of the SHUD model include hydrologic analyses of hillslope to regional scales  (1 $ m^2 $ to $10^6$ $\mbox{km}^2$), water resource and stormwater management, and interdisciplinary research for questions in limnology, agriculture, geochemistry, geomorphology, water quality, ecology, climate and land-use change. The strength of SHUD is its flexibility as a scientific and resource evaluation tool where modeling and simulation are required.



- **Maintainner**: Lele Shu ([shulele@lzb.ac.cn](mailto:shulele@lzb.ac.cn))
- **Website (中文)**: [www.shud.xyz/](https://www.shud.xyz/)
- **Website (English)**: [www.shud.xyz/en/](https://www.shud.xyz/en/)
- **User Guide**: [https://www.shud.xyz/book_cn/](https://www.shud.xyz/book_cn/)
- **Support tools**: rSHUD.  [https://github.com/SHUD-System/rSHUD](https://github.com/SHUD-System/rSHUD)
- **Programming**: C/C++
- **Platform**: Mac OS, Linux and Windows
- **Required library**:  SUNDIALS/CVODE V6.0+
- **Parallelization** : OpenMP

## Overview

The Simulator for Hydrologic Unstructured Domains (SHUD) is a multi-process, multi-scale model where major hydrologic processes are fully coupled using the **Finite Volume Method (FVM)**. SHUD encapsulates the strategy for the synthesis of multi-state distributed hydrologic models using the integral representation of the underlying physical process equations and state variables.


The conceptual structure of the ***two-state integral-balance*** model for soil moisture and groundwater dynamics was originally devised by (Duffy, 1996), in which the partial volumes occupied by unsaturated and saturated moisture storage were integrated directly into a local conservation equation. This two-state integral-balance structure simplified the hydrologic dynamics while preserving the natural spatial and temporal scales contributing to runoff response.

SHUD's design is based on a concise representation of a watershed and river basin's hydrodynamics, which allows for interactions among major physical processes operating simultaneously, but with the flexibility to add or drop state-process-constitutive relations depending on the objectives of the numerical experiment.

![figure1](Fig/figure1.png)

The latest version of SHUD (v2.0) supports the simulation of coupled lake model.
![Lake coupling](Fig/lake.png)


As an intellectual descendant of Penn State Integrated Hydrologic Model (**PIHM**), the SHUD model is a continuation of 16 years of PIHM model development in hydrology and related fields since the release of its first PIHM version  (Qu, 2004).

![Figure_tree](Fig/Figure_tree.png)

###The formulation and results from SHUD. 

- SHUD is a physically-based process spatially distributed catchment model. The model applies national geospatial data resources to simulate surface and subsurface flow in gaged or ungaged catchments. SHUD represents the spatial heterogeneity that influences the hydrology of the region based on national soil data and superficial geology. Several other groups have used PIHM, a SHUD ancestor to couple processes from biochemistry, reaction transport, landscape, geomorphology, limnology, and other related research areas.

- SHUD is a fully-coupled hydrologic model, where the conservative hydrologic fluxes are calculated within the same time step. The state variables are the height of ponding water on the land surface, soil moisture, groundwater level, and river stage, while fluxes are infiltration, overland flow, groundwater recharge, lateral groundwater flow, river discharge, and exchange between river and hillslope cells.

- The global ODE system in SHUD is solved with a state-of-the-art parallel ODE solver, known as CVODE developed at Lawrence Livermore National Laboratory.

- SHUD permits adaptable temporal and spatial resolution. The spatial resolution of the model varies from centimeters to kilometers based on modeling requirements computing resources. The internal time step of the iteration is adjustable and adaptive; it can export the status of a catchment at time-intervals from minutes to days.  The flexible spatial and temporal resolution of the model makes it valuable for coupling with other systems.

- SHUD can estimate either a long-term hydrologic yield or a single-event flood.

- SHUD is an open-source model, available on GitHub.

  


## Compilation (Linux or Mac) and run the example watersheds

**Step 0: download the latest source code**

```
git clone git@github.com:SHUD-System/SHUD.git
cd SHUD
```

**Step 1: Install SUNDIALS/CVODE 6.x:**

```
./configure
```

This configure is to download the SUNDIALS from GitHub and install it on your computer.

**Optional: CUDA-enabled build (Linux + NVIDIA only)**

SHUD provides a CUDA build (`shud_cuda`) and a runtime backend switch (`--backend cuda`) to offload the RHS to GPU via SUNDIALS `NVECTOR_CUDA`.

**GPU prerequisites**

- NVIDIA GPU with Compute Capability (CC) >= 7.0 (sm_70+)
- NVIDIA CUDA Toolkit >= 11.0 (`nvcc` available)
- CMake >= 3.18 (required by `./configure_cuda`)
- SUNDIALS/CVODE 6.x built with `ENABLE_CUDA=ON` (provides `libsundials_nveccuda`)

**Step 1b: Build CUDA-enabled SUNDIALS (NVECTOR_CUDA)**

```bash
# Default install prefix is $HOME/sundials (matches Makefile's SUNDIALS_DIR).
SUNDIALS_PREFIX="$HOME/sundials" CUDA_ARCHS="70;75;80;86" ./configure_cuda

# Verify the CUDA NVECTOR libraries exist:
ls "$HOME/sundials/lib" | grep -E 'sundials_nveccuda|sunmemcuda' || true
```

`CUDA_ARCHS` should match your GPU (examples: 70=V100, 75=T4, 80=A100, 86=RTX30).

**Step 1c: Build `shud_cuda`**

```bash
# If CUDA is installed elsewhere, point CUDA_HOME to it.
make shud_cuda CUDA_HOME=/usr/local/cuda

# If SUNDIALS is installed elsewhere:
# make shud_cuda SUNDIALS_DIR=/path/to/sundials CUDA_HOME=/usr/local/cuda
```

If you see `no kernel image is available for execution on the device` / `invalid device function` at runtime, rebuild for your GPU architecture. Example for A100 (sm_80):

```bash
CUDA_ARCHS="80" SUNDIALS_PREFIX="$HOME/sundials" ./configure_cuda
make shud_cuda CUDA_GENCODE='-gencode arch=compute_80,code=sm_80'
```

**Run on GPU**

```bash
./shud_cuda --backend cuda ccw
```

**Runtime options**

- `--backend cpu|omp|cuda`: select runtime backend (default `cpu`)
- `--precond` / `--no-precond`: enable/disable CVODE preconditioner (CUDA backend only; default ON for `--backend cuda`)

Examples:

```bash
./shud_cuda --backend cuda --no-precond ccw
./shud_cuda --backend omp -n 8 ccw
```

**Performance monitoring (CVODE_STATS)**

For performance monitoring/regression checks, run with `CVODE_STATS=1` and grep the parseable CVODE statistics line printed at the end of the run:

```bash
CVODE_STATS=1 ./shud_cuda --backend cuda ccw
# CVODE_STATS nfe=... nli=... nni=... netf=... npe=... nps=...
```

**Troubleshooting (CUDA)**

- `nvcc: command not found`: install CUDA Toolkit and/or set `CUDA_HOME` and add `$CUDA_HOME/bin` to `PATH`
- `cannot find -lsundials_nveccuda` or `--backend cuda requested, but this build does not enable CUDA`: build SUNDIALS with `./configure_cuda` and build/run `shud_cuda` (not `shud`)
- `no kernel image is available for execution on the device` / `invalid device function`: your GPU CC is not in `CUDA_ARCHS` / `CUDA_GENCODE`; rebuild with the correct arch
- `CUDA driver version is insufficient for CUDA runtime version`: update the NVIDIA driver or use a compatible CUDA Toolkit
- `error while loading shared libraries: libcudart.so...`: ensure CUDA runtime libs are discoverable (e.g. `$CUDA_HOME/lib64` in `LD_LIBRARY_PATH`) and rebuild

**Step 2: Compile SHUD with gcc**

```
make clean
make shud

```

If you don't use `gcc`, you may edit the *Makefile* before compiling.

**Step 3: Run the North Fork Cache Creek Watershed example**

```
./shud ccw
```

Semantics docs: time stepping + forcing/ET update order (Section 2.1) and RHS baseline (Serial reference): see `docs/baseline_semantics.md`.

The screen looks shoud be:
![screenshot](Fig/screenshot.png)

**Step4: Analysis the results of modeling.**

The output files from the SHUD model is save in `./output/ccw.out`.  The R package, SHUDtoolbox, helps to load the input/output files of SHUD. More details about prepare SHUD data, model input/output and visualization is available in SHUD website (https://www.shud.xyz) and help information of SHUDtoolbox.

## TSR validation (Python vs C++)

This repo includes a TSR (Terrain Solar Radiation) validation workflow under `validation/tsr/`.

Generate TSR=ON outputs for `ccw`:

```bash
bash validation/tsr/run_tsr.sh
```

Recompute TSR factor + `rn_t` in Python (C++-consistent) and compare pointwise:

```bash
python3 validation/tsr/py/compare_tsr.py output/ccw.tsr --tol 1e-10
```

Run Python unit tests (and coverage, optional):

```bash
python3 -m unittest discover -s validation/tsr/py -p 'test_*.py'
python3 -m coverage run -m unittest discover -s validation/tsr/py -p 'test_*.py'
python3 -m coverage report --fail-under=90
```
