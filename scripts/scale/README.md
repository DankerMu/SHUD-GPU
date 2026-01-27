# SCALE-001: Scaled synthetic cases

This directory provides a small generator that creates larger *synthetic* SHUD input cases by **tiling** an existing case (default: `ccw`) multiple times.

The generated case keeps all per-element attributes and time-series inputs identical to the base case; it only expands:
- elements + nodes (`.sp.mesh`)
- element attributes (`.sp.att`)
- river reaches + segments (`.sp.riv`, `.sp.rivseg`)
- initial conditions (`.cfg.ic`)

Tiles are laid out along the X axis (node X coordinates are shifted per tile), so the copies do not overlap geometrically.

## Usage

Generate by target `NY`:
```bash
python3 scripts/scale/generate_scaled_case.py --base ccw --ny 10000   --output ccw_ny1e4  --end-day 2
python3 scripts/scale/generate_scaled_case.py --base ccw --ny 100000  --output ccw_ny1e5  --end-day 2
python3 scripts/scale/generate_scaled_case.py --base ccw --ny 1000000 --output ccw_ny1e6  --end-day 2
```

Or generate by an explicit tiling factor:
```bash
python3 scripts/scale/generate_scaled_case.py --base ccw --factor 29 --output ccw_x29 --end-day 2
```

## Running

```bash
make shud
./shud --io off ccw_ny1e5
```

Tip: use `--io off` to minimize I/O noise when benchmarking compute.

