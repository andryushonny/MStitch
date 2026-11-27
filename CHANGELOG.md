# Changelog

## 0.1.0
- First version: C++ core with a Python module (nanobind), CPU backend.
- Takes MControl snaps as they come (Bayer `lossless` / `raw`, 8 and 12 bit), unpacks in C++ on all cores.
- Pairs from stage positions; phase correlation + NCC with sub-pixel refinement; robust weighted least squares.
- Modes `quick` (1/4 size) and `quality` (1/2 size); Voronoi render without double images; flat-field.
- `calibrate()` for objective profiles (stage→pixel matrix, repeatability, flat-field); reshoot list for the second pass.
- `Canvas`: pixel ↔ stage um, crops, grid cells, saving as tiles + manifest + poses.
