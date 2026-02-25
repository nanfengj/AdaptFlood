# LISFLOOD-GPU

**GPU-accelerated 2D flood simulation based on Kokkos**

基于 Kokkos 异构计算框架的 GPU 加速二维洪水模拟程序，兼容 LISFLOOD-FP 输入格式。

---

## Features

- **GPU Acceleration** — Based on [Kokkos](https://github.com/kokkos/kokkos), runs on NVIDIA (CUDA), AMD (HIP), and CPU (OpenMP) with a single codebase
- **LISFLOOD-FP Compatible** — Reads standard `.par`, `.dem.ascii`, `.bci`, `.bdy` input files
- **Adaptive Timestep** — CFL-based adaptive time stepping for stability and efficiency
- **Point Source Injection** — Supports thousands of QVAR/QFIX point sources (e.g., urban drainage overflow points)
- **Boundary Conditions** — FREE, HFIX, HVAR, QFIX, QVAR point boundaries + FREE line boundaries (N/S/E/W)
- **Rainfall Input** — Optional spatially-uniform time-varying rainfall
- **Stage Monitoring** — Track water depth at specified monitoring points over time
- **Async I/O** — Non-blocking file output using background threads
- **Multiple Output Formats** — ASC (ESRI ASCII Grid) and NetCDF

---

## Quick Start

### Prerequisites

| Requirement | Version |
|-------------|---------|
| C++ Compiler | C++17 support (GCC 9+, Clang 10+) |
| CMake | >= 3.16 |
| [Kokkos](https://github.com/kokkos/kokkos) | >= 4.0 (with CUDA or OpenMP backend) |
| NVIDIA GPU + CUDA | >= 11.0 (for GPU mode) |
| NetCDF (optional) | For `.nc` output |

### Build

```bash
# Clone
git clone https://github.com/nanfengj/lisflood-gpu.git
cd lisflood-gpu

# Build
mkdir build && cd build
cmake .. -DKokkos_DIR=/path/to/kokkos/lib/cmake/Kokkos
make -j$(nproc)
```

Or use the build script:

```bash
bash build.sh
```

### Run

```bash
./build/lisflood_gpu your_case.par
```

---

## Input Files

### Parameter File (`.par`)

```
DEMfile       dem.ascii
bcifile       bci.bci
bdyfile       bdy.bdy
resroot       results
sim_time      3600
saveint       600
massint       60
cfl           0.7
fpn           0.035
depththresh   0.001
acceleration
voutput
netcdf
asyncoutput
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `DEMfile` | DEM file path (ESRI ASCII format) | `dem.ascii` |
| `bcifile` | Boundary condition index file | `bci.bci` |
| `bdyfile` | Boundary condition data file | `bdy.bdy` |
| `resroot` | Output directory | `results` |
| `sim_time` | Simulation duration (seconds) | 3600 |
| `saveint` | Output interval (seconds) | 600 |
| `massint` | Mass balance report interval (seconds) | 60 |
| `cfl` | CFL number for adaptive timestep | 0.7 |
| `fpn` | Manning's n roughness coefficient | 0.035 |
| `depththresh` | Minimum depth threshold (m) | 0.001 |
| `acceleration` | Enable inertial acceleration formulation | off |
| `voutput` | Enable velocity output | off |
| `netcdf` | Enable NetCDF output format | off |
| `asyncoutput` | Enable asynchronous file I/O | on |
| `rainfile` | Rainfall time series file (optional) | — |
| `stagefile` | Stage monitoring points file (optional) | — |

### DEM File (ESRI ASCII Grid)

```
ncols        100
nrows        100
xllcorner    0.0
yllcorner    0.0
cellsize     10.0
NODATA_value -9999
<elevation data row by row>
```

### Boundary Condition Index (`.bci`)

```
P    col  row  QVAR  boundary_name
P    col  row  QFIX  0.5
P    col  row  FREE
N    0    100  FREE
```

- `P` = point boundary, `N/S/E/W` = line boundary (North/South/East/West)
- `QVAR` = time-varying discharge, `QFIX` = fixed discharge
- `HVAR` = time-varying water level, `HFIX` = fixed water level
- `FREE` = free outflow boundary

### Boundary Condition Data (`.bdy`)

```
boundary_name
num_entries  seconds
time1  value1
time2  value2
...
```

Each named boundary referenced in `.bci` should have a corresponding section in `.bdy`.

---

## Output

| File | Description |
|------|-------------|
| `results/H_0000.asc` ... | Water depth grids at each save interval |
| `results/H_final.asc` | Final water depth |
| `results/Vx_final.asc` | Final x-velocity (if `voutput` enabled) |
| `results/Vy_final.asc` | Final y-velocity (if `voutput` enabled) |
| `results/output.nc` | NetCDF output (if `netcdf` enabled) |
| `results/stage.csv` | Stage monitoring time series (if `stagefile` set) |

Console output includes mass balance reports:

```
t=600s: Tstep=0.85, Vol=125000m3, Area=50000m2
```

---

## Project Structure

```
lisflood-gpu/
├── CMakeLists.txt              # Build configuration
├── build.sh                    # Build helper script
├── src/
│   ├── main.cpp                # Main program entry
│   ├── types.hpp               # Data structures & Kokkos type aliases
│   ├── io.cpp / io.hpp         # File I/O (DEM, BCI, BDY, PAR, ASC, NetCDF)
│   ├── kernels.cpp / kernels.hpp   # GPU compute kernels
│   └── async_output.cpp / .hpp     # Async file output manager
├── lisflood_gpu_core.cpp       # Standalone GPU core library (reference)
├── test_point_source.cpp       # Point source unit test (50x50 grid)
└── test_kokkos_cuda.cpp        # Kokkos+CUDA environment test
```

---

## Algorithm

This program implements the **inertial formulation** of the 2D shallow water equations (de Almeida et al., 2012), the same numerical scheme used by [LISFLOOD-FP](https://www.seamlesswave.com/LISFLOOD8.0.html):

1. **Floodplain Q calculation** — Compute inter-cell discharge using the inertial approximation with Manning friction
2. **Boundary conditions** — Apply FREE outflow at domain edges
3. **Point source injection** — Add flow from QVAR/QFIX sources as `dH = Q * dt / cell_area`
4. **Rainfall** — Add uniform rainfall across all valid cells
5. **Update H** — Update water depth from net discharge divergence
6. **Adaptive timestep** — CFL-based: `dt = CFL * dx / sqrt(g * h_max)`

### Key GPU Kernels

| Kernel | Function |
|--------|----------|
| `FloodplainQ_GPU` | Compute Q fluxes on all cell interfaces |
| `UpdateH_GPU` | Update water depth from flux divergence |
| `BCs_GPU` | Apply line boundary conditions |
| `ApplyPointBCs_GPU` | Apply point boundary conditions |
| `Rainfall_GPU` | Add rainfall to all valid cells |
| `ComputeVelocity_GPU` | Compute velocity field from Q and H |

---

## Performance

Tested on the Haizhu District (Guangzhou, China) urban flooding case:

| Hardware | Grid Size | Resolution | Sim Time | Wall Time |
|----------|-----------|------------|----------|-----------|
| RTX 4060 Laptop (8GB) | 4500万 cells | 2m | 5.75 hours | ~17 min |
| RTX 4090 (24GB) | 4500万 cells | 2m | 5.75 hours | ~4 min |

- 8,210 point sources (drainage overflow), 2,087 active
- Total injected volume: ~260,000 m³

---

## Development Environment

| Item | Value |
|------|-------|
| GPU | NVIDIA GeForce RTX 4060 Laptop (8GB) |
| CUDA | 12.6 |
| Kokkos | 4.4.99 (CUDA + OpenMP backend) |
| OS | WSL2 Ubuntu on Windows |

---

## License

This project is open-source. Feel free to use, modify, and distribute.

---

## Acknowledgments

- [LISFLOOD-FP](https://www.seamlesswave.com/LISFLOOD8.0.html) — Original CPU flood model by University of Bristol
- [Kokkos](https://github.com/kokkos/kokkos) — Performance-portable parallel programming framework
