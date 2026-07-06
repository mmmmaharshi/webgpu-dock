# webgpu-dock

GPU-accelerated molecular docking in the browser using WebGPU compute shaders. Predicts how small molecules bind to protein targets — entirely client-side, no Python or native code.

Built on AutoDock4's force field (Lennard-Jones 12-6 + Coulomb with Mehler-Solmajer dielectric) with a three-stage pipeline:

1. **Affinity grid** — precompute LJ+Coulomb potentials on a 3D grid around the pocket (WGSL compute shader)
2. **Coarse global search** — score 70k+ poses per conformer via trilinear-interpolated grid lookups, seeded with quaternion-based rotations over uniform translations
3. **BFGS local refinement** — full pair-wise scoring with angle-axis parameterization and Armijo line search

## Systems (8 benchmarks)

| System | Target | Ligand |
|---|---|---|
| 1IEP | Imatinib (Gleevec) | ABL kinase |
| 1HSG | Indinavir | HIV-1 protease |
| 1STP | Biotin | Streptavidin |
| 3PTB | Benzamidine | Trypsin |
| 1AC8 | Temozolomide | Alkyltransferase |
| 3CE3 | Factor Xa inhibitor | Coagulation factor Xa |
| 3TMN | Thermolysin inhibitor | Thermolysin |
| 7CPA | Carboxypeptidase A inhibitor | Carboxypeptidase A |

## Usage

```bash
bun install
bun start        # build + serve on http://localhost:8080
```

Open `http://localhost:8080` in a WebGPU-capable browser (Chrome 113+). Benchmark runs automatically on page load.

## Scripts

| Command | Action |
|---|---|
| `bun start` | Build app + start server |
| `bun dev` | Build + hot-reload server |
| `bun run build:app` | Bundle TypeScript for browser |
| `bun run typecheck` | Type-check without emitting |

## Architecture

All computation runs on the client via WebGPU WGSL shaders. The Bun server is a static file server that auto-downloads benchmark PDBQT files on first start.

Three shaders in `src/shader.ts`:
- `SHADER_SRC` — full pair-wise scoring (BFGS refinement)
- `SHADER_BUILD_GRID_SRC` — affinity grid construction
- `SHADER_GRID_SRC` — grid-based trilinear-interpolated scoring

No runtime dependencies. Dev dependencies: Bun, TypeScript, `@webgpu/types`.
