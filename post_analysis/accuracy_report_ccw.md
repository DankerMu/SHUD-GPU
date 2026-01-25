# SHUD Backend Accuracy Comparison Report

- Project: `ccw`
- CPU dir: `output/ccw_cpu`
- OMP dir: `output/ccw_omp`
- CUDA dir: `output/ccw_cuda`
- Generated: `2026-01-25 10:28:33`

## Tolerances (relative to max(|CPU|) per file)

- CPU vs OMP: `rel_max <= 1e-10` (expect ~1e-10 or smaller)
- CPU vs CUDA: `rel_max <= 1e-06` (expect ~1e-6 or smaller)

## Summary

- Files scanned: `25` (`*.dat` under CPU dir), compared: `24`, skipped: `1`
- CPU vs OMP: PASS `8`, FAIL `16`, MISSING `0`, ERROR `0`
- Worst CPU vs OMP: `ccw.elevrech.dat` (`rel_max=4.793e-01`)
- CPU vs CUDA: PASS `7`, FAIL `17`, MISSING `0`, ERROR `0`
- Worst CPU vs CUDA: `ccw.eleysurf.dat` (`rel_max=5.992e+00`)

## Skipped files (CPU unreadable)

- `DY.dat`: output/ccw_cpu/DY.dat: empty .dat file

## Key variables

| Variable | File | CPU vs OMP rel_max | CPU vs CUDA rel_max | Notes |
|---|---|---:|---:|---|
| yGw (Groundwater) | `ccw.eleygw.dat` | 3.198e-03 | 1.308e-02 | OMP worst@t=1645920 min (1143.00 d), id=508; CUDA worst@t=2063520 min (1433.00 d), id=508 |
| ySf (Surface water depth) | `ccw.eleysurf.dat` | 1.505e-01 | 5.992e+00 | OMP worst@t=2151360 min (1494.00 d), id=508; CUDA worst@t=1556640 min (1081.00 d), id=1072 |
| yUs (Unsaturated zone) | `ccw.eleyunsat.dat` | 7.959e-03 | 1.194e-01 | OMP worst@t=2152800 min (1495.00 d), id=508; CUDA worst@t=1692000 min (1175.00 d), id=1019 |
| yRiv (River stage) | `ccw.rivystage.dat` | 1.676e-02 | 8.423e-02 | OMP worst@t=21600 min (15.00 d), id=1; CUDA worst@t=1690560 min (1174.00 d), id=1 |

## CPU vs OMP (all .dat)

| Type | File | Shape(TxN) | Scale(max|CPU|) | rel_max | rel_mean | rel_std | abs_max | Status | Worst |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| Flux/Diag | `ccw.elevrech.dat` | 1827x1147 | 1.083e-01 | 4.793e-01 | 1.023e-06 | 5.267e-04 | 5.191e-02 | FAIL | t=1681920 min (1168.00 d), id=508 |
| Flux/Diag | `ccw.elevinfil.dat` | 1827x1147 | 1.192e-01 | 3.976e-01 | 3.645e-03 | 9.938e-03 | 4.741e-02 | FAIL | t=2151360 min (1494.00 d), id=511 |
| Flux/Diag | `ccw.elevetev.dat` | 1827x1147 | 5.763e-03 | 2.359e-01 | 7.797e-04 | 2.885e-03 | 1.359e-03 | FAIL | t=1713600 min (1190.00 d), id=508 |
| State(y*) | `ccw.eleysurf.dat` | 1827x1147 | 4.521e-02 | 1.505e-01 | 7.520e-04 | 1.018e-03 | 6.806e-03 | FAIL | t=2151360 min (1494.00 d), id=508 |
| Flux/Diag | `ccw.elevettr.dat` | 1827x1147 | 8.289e-03 | 1.454e-01 | 3.080e-05 | 5.750e-04 | 1.205e-03 | FAIL | t=1363680 min (947.00 d), id=508 |
| Flux(Q*) | `ccw.eleqsurf.dat` | 1827x1147 | 2.724e+04 | 1.421e-01 | 3.291e-04 | 1.293e-03 | 3.869e+03 | FAIL | t=97920 min (68.00 d), id=678 |
| Flux/Diag | `ccw.eleveta.dat` | 1827x1147 | 1.043e-02 | 1.304e-01 | 4.544e-04 | 1.654e-03 | 1.359e-03 | FAIL | t=1713600 min (1190.00 d), id=508 |
| Flux/Diag | `ccw.elevexfil.dat` | 1827x1147 | 3.412e-04 | 2.392e-02 | 4.265e-07 | 6.709e-05 | 8.162e-06 | FAIL | t=982080 min (682.00 d), id=508 |
| Flux(Q*) | `ccw.eleqsub.dat` | 1827x1147 | 3.905e+04 | 1.867e-02 | 1.515e-04 | 5.711e-04 | 7.293e+02 | FAIL | t=658080 min (457.00 d), id=971 |
| State(y*) | `ccw.rivystage.dat` | 1827x103 | 1.022e+00 | 1.676e-02 | 7.958e-06 | 1.126e-04 | 1.713e-02 | FAIL | t=21600 min (15.00 d), id=1 |
| Flux(Q*) | `ccw.rivqsurf.dat` | 1827x103 | 2.454e+05 | 1.491e-02 | 3.344e-05 | 2.686e-04 | 3.659e+03 | FAIL | t=1552320 min (1078.00 d), id=65 |
| Flux(Q*) | `ccw.rivqsub.dat` | 1827x103 | 1.099e+05 | 1.404e-02 | 1.130e-05 | 1.287e-04 | 1.542e+03 | FAIL | t=21600 min (15.00 d), id=1 |
| Flux(Q*) | `ccw.rivqdown.dat` | 1827x103 | 6.802e+06 | 1.115e-02 | 4.676e-06 | 1.123e-04 | 7.584e+04 | FAIL | t=1553760 min (1079.00 d), id=1 |
| Flux(Q*) | `ccw.rivqup.dat` | 1827x103 | 6.833e+06 | 1.083e-02 | 4.299e-06 | 1.125e-04 | 7.401e+04 | FAIL | t=1553760 min (1079.00 d), id=2 |
| State(y*) | `ccw.eleyunsat.dat` | 1827x1147 | 1.208e+01 | 7.959e-03 | 6.106e-05 | 7.617e-05 | 9.614e-02 | FAIL | t=2152800 min (1495.00 d), id=508 |
| State(y*) | `ccw.eleygw.dat` | 1827x1147 | 3.007e+01 | 3.198e-03 | 5.409e-06 | 2.359e-05 | 9.614e-02 | FAIL | t=1645920 min (1143.00 d), id=508 |
| Flux/Diag | `ccw.elevetic.dat` | 1827x1147 | 1.808e-03 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | PASS | t=0 min (0.00 d), id=1 |
| Flux/Diag | `ccw.elevetp.dat` | 1827x1147 | 1.917e-02 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | PASS | t=0 min (0.00 d), id=1 |
| Flux/Diag | `ccw.elevnetprcp.dat` | 1827x1147 | 9.169e-02 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | PASS | t=0 min (0.00 d), id=1 |
| Flux/Diag | `ccw.elevprcp.dat` | 1827x1147 | 9.169e-02 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | PASS | t=0 min (0.00 d), id=1 |
| Other | `ccw.eleysnow.dat` | 1827x1147 | 7.140e-04 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | PASS | t=0 min (0.00 d), id=1 |
| Flux/Diag | `ccw.rn_factor.dat` | 1827x1147 | 9.940e-01 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | PASS | t=0 min (0.00 d), id=1 |
| Flux/Diag | `ccw.rn_h.dat` | 1827x1147 | 3.844e+02 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | PASS | t=0 min (0.00 d), id=1 |
| Flux/Diag | `ccw.rn_t.dat` | 1827x1147 | 3.847e+02 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | PASS | t=0 min (0.00 d), id=1 |

## CPU vs CUDA (all .dat)

| Type | File | Shape(TxN) | Scale(max|CPU|) | rel_max | rel_mean | rel_std | abs_max | Status | Worst |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| State(y*) | `ccw.eleysurf.dat` | 1827x1147 | 4.521e-02 | 5.992e+00 | 9.823e-03 | 1.039e-01 | 2.709e-01 | FAIL | t=1556640 min (1081.00 d), id=1072 |
| Flux(Q*) | `ccw.eleqsub.dat` | 1827x1147 | 3.905e+04 | 1.000e+00 | 9.991e-04 | 5.559e-03 | 3.905e+04 | FAIL | t=1552320 min (1078.00 d), id=1136 |
| Flux(Q*) | `ccw.eleqsurf.dat` | 1827x1147 | 2.724e+04 | 1.000e+00 | 1.538e-03 | 1.133e-02 | 2.724e+04 | FAIL | t=1550880 min (1077.00 d), id=183 |
| Flux/Diag | `ccw.eleveta.dat` | 1827x1147 | 1.043e-02 | 1.000e+00 | 1.334e-01 | 1.113e-01 | 1.043e-02 | FAIL | t=2373120 min (1648.00 d), id=511 |
| Flux/Diag | `ccw.elevetev.dat` | 1827x1147 | 5.763e-03 | 1.000e+00 | 6.962e-02 | 4.609e-02 | 5.763e-03 | FAIL | t=1801440 min (1251.00 d), id=232 |
| Flux/Diag | `ccw.elevettr.dat` | 1827x1147 | 8.289e-03 | 1.000e+00 | 1.142e-01 | 1.159e-01 | 8.289e-03 | FAIL | t=2373120 min (1648.00 d), id=511 |
| Flux/Diag | `ccw.elevexfil.dat` | 1827x1147 | 3.412e-04 | 1.000e+00 | 2.327e-05 | 4.054e-03 | 3.412e-04 | FAIL | t=100800 min (70.00 d), id=508 |
| Flux/Diag | `ccw.elevinfil.dat` | 1827x1147 | 1.192e-01 | 1.000e+00 | 2.072e-02 | 5.370e-02 | 1.192e-01 | FAIL | t=2623680 min (1822.00 d), id=900 |
| Flux/Diag | `ccw.elevrech.dat` | 1827x1147 | 1.083e-01 | 1.000e+00 | 5.920e-06 | 1.889e-03 | 1.083e-01 | FAIL | t=541440 min (376.00 d), id=508 |
| Flux(Q*) | `ccw.rivqdown.dat` | 1827x103 | 6.802e+06 | 1.000e+00 | 8.015e-04 | 9.888e-03 | 6.802e+06 | FAIL | t=1552320 min (1078.00 d), id=4 |
| Flux(Q*) | `ccw.rivqsub.dat` | 1827x103 | 1.099e+05 | 1.000e+00 | 3.434e-03 | 1.170e-02 | 1.099e+05 | FAIL | t=1552320 min (1078.00 d), id=1 |
| Flux(Q*) | `ccw.rivqsurf.dat` | 1827x103 | 2.454e+05 | 1.000e+00 | 1.415e-03 | 1.580e-02 | 2.454e+05 | FAIL | t=1552320 min (1078.00 d), id=65 |
| Flux(Q*) | `ccw.rivqup.dat` | 1827x103 | 6.833e+06 | 1.000e+00 | 7.370e-04 | 9.768e-03 | 6.833e+06 | FAIL | t=1552320 min (1078.00 d), id=2 |
| Flux/Diag | `ccw.elevetic.dat` | 1827x1147 | 1.808e-03 | 7.378e-01 | 2.272e-02 | 6.606e-02 | 1.334e-03 | FAIL | t=152640 min (106.00 d), id=1053 |
| State(y*) | `ccw.eleyunsat.dat` | 1827x1147 | 1.208e+01 | 1.194e-01 | 1.039e-03 | 4.080e-03 | 1.442e+00 | FAIL | t=1692000 min (1175.00 d), id=1019 |
| State(y*) | `ccw.rivystage.dat` | 1827x103 | 1.022e+00 | 8.423e-02 | 7.021e-05 | 7.281e-04 | 8.611e-02 | FAIL | t=1690560 min (1174.00 d), id=1 |
| State(y*) | `ccw.eleygw.dat` | 1827x1147 | 3.007e+01 | 1.308e-02 | 4.457e-05 | 1.820e-04 | 3.932e-01 | FAIL | t=2063520 min (1433.00 d), id=508 |
| Flux/Diag | `ccw.elevetp.dat` | 1827x1147 | 1.917e-02 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | PASS | t=0 min (0.00 d), id=1 |
| Flux/Diag | `ccw.elevnetprcp.dat` | 1827x1147 | 9.169e-02 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | PASS | t=0 min (0.00 d), id=1 |
| Flux/Diag | `ccw.elevprcp.dat` | 1827x1147 | 9.169e-02 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | PASS | t=0 min (0.00 d), id=1 |
| Other | `ccw.eleysnow.dat` | 1827x1147 | 7.140e-04 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | PASS | t=0 min (0.00 d), id=1 |
| Flux/Diag | `ccw.rn_factor.dat` | 1827x1147 | 9.940e-01 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | PASS | t=0 min (0.00 d), id=1 |
| Flux/Diag | `ccw.rn_h.dat` | 1827x1147 | 3.844e+02 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | PASS | t=0 min (0.00 d), id=1 |
| Flux/Diag | `ccw.rn_t.dat` | 1827x1147 | 3.847e+02 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | PASS | t=0 min (0.00 d), id=1 |

## Notes

- Metrics are computed on the intersection of time steps and column IDs (icol) between CPU and the target backend.
- Relative errors use `max(|CPU|)` per file as the normalization scale (avoids undefined relative errors near zero).
