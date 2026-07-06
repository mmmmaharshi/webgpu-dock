export const SHADER_SRC = `
struct Params {
  numProtein: u32,
  numLigand: u32,
  numPoses: u32,
  _pad: u32,
};

@group(0) @binding(0) var<storage, read> proteinAtoms: array<f32>; // stride 6
@group(0) @binding(1) var<storage, read> ligandAtoms: array<f32>;  // stride 6 (local coords)
@group(0) @binding(2) var<storage, read> poses: array<f32>;        // stride 12
@group(0) @binding(3) var<storage, read_write> energies: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

const COULOMB_K: f32 = 332.0636;

// Mehler-Solmajer distance-dependent dielectric (AutoDock4's default
// electrostatics model). Must match src/scoring.ts's CPU dielectric()
// exactly, or the GPU search and CPU diagnostics will disagree.
const DIELECTRIC_A: f32 = -8.5525;
const DIELECTRIC_B: f32 = 86.9525;
const DIELECTRIC_LAMBDA: f32 = 0.003627;
const DIELECTRIC_K: f32 = 7.7839;

fn dielectric(r: f32) -> f32 {
  return DIELECTRIC_A + DIELECTRIC_B / (1.0 + DIELECTRIC_K * exp(-DIELECTRIC_LAMBDA * DIELECTRIC_B * r));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let poseIdx = gid.x;
  if (poseIdx >= params.numPoses) {
    return;
  }

  let base = poseIdx * 12u;
  let r0 = poses[base + 0u]; let r1 = poses[base + 1u]; let r2 = poses[base + 2u];
  let r3 = poses[base + 3u]; let r4 = poses[base + 4u]; let r5 = poses[base + 5u];
  let r6 = poses[base + 6u]; let r7 = poses[base + 7u]; let r8 = poses[base + 8u];
  let tx = poses[base + 9u]; let ty = poses[base + 10u]; let tz = poses[base + 11u];

  var totalEnergy: f32 = 0.0;
  var clashCount: u32 = 0u;

  for (var i: u32 = 0u; i < params.numLigand; i = i + 1u) {
    let lBase = i * 6u;
    let lx0 = ligandAtoms[lBase];
    let ly0 = ligandAtoms[lBase + 1u];
    let lz0 = ligandAtoms[lBase + 2u];
    let lCharge = ligandAtoms[lBase + 3u];
    let lSigma  = ligandAtoms[lBase + 4u];
    let lEps    = ligandAtoms[lBase + 5u];

    let lx = r0 * lx0 + r1 * ly0 + r2 * lz0 + tx;
    let ly = r3 * lx0 + r4 * ly0 + r5 * lz0 + ty;
    let lz = r6 * lx0 + r7 * ly0 + r8 * lz0 + tz;

    for (var j: u32 = 0u; j < params.numProtein; j = j + 1u) {
      let qBase = j * 6u;
      let dx = lx - proteinAtoms[qBase];
      let dy = ly - proteinAtoms[qBase + 1u];
      let dz = lz - proteinAtoms[qBase + 2u];
      let pCharge = proteinAtoms[qBase + 3u];
      let pSigma  = proteinAtoms[qBase + 4u];
      let pEps    = proteinAtoms[qBase + 5u];

      let r2 = dx * dx + dy * dy + dz * dz;
      if (r2 < 1.0) { clashCount = clashCount + 1u; }
      let r2v = max(r2, 1.0);
      let r  = sqrt(r2v);

      let sigma   = 0.5 * (lSigma + pSigma);
      let epsilon = sqrt(lEps * pEps);
      let sr6  = pow(sigma / r, 6.0);
      let sr12 = sr6 * sr6;
      var lj = 4.0 * epsilon * (sr12 - sr6);
      lj = min(lj, 10.0);

      let coulomb = COULOMB_K * lCharge * pCharge / (dielectric(r) * r);

      totalEnergy = totalEnergy + lj + coulomb;
    }
  }

  if (clashCount > 3u) { totalEnergy = 1e10; }
  totalEnergy = min(totalEnergy, 10000.0);
  energies[poseIdx] = totalEnergy;
}
`;

export const SHADER_BUILD_GRID_SRC = `
struct BuildParams {
  originX: f32, originY: f32, originZ: f32,
  spacing: f32,
  dimX: u32, dimY: u32, dimZ: u32,
  numProtein: u32,
  numUniqueTypes: u32,
  coulLayer: u32,
  _pad: u32,
};

@group(0) @binding(0) var<storage, read> proteinAtoms: array<f32>;
@group(0) @binding(1) var<storage, read> ligParams: array<f32>;
@group(0) @binding(2) var<storage, read_write> gridOut: array<f32>;
@group(0) @binding(3) var<uniform> params: BuildParams;

const DIELECTRIC_A: f32 = -8.5525;
const DIELECTRIC_B: f32 = 86.9525;
const DIELECTRIC_LAMBDA: f32 = 0.003627;
const DIELECTRIC_K: f32 = 7.7839;

fn dielectric(r: f32) -> f32 {
  return DIELECTRIC_A + DIELECTRIC_B / (1.0 + DIELECTRIC_K * exp(-DIELECTRIC_LAMBDA * DIELECTRIC_B * r));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let nx = params.dimX;
  let ny = params.dimY;
  let nz = params.dimZ;
  let totalPoints = nx * ny * nz;
  if (idx >= totalPoints) { return; }

  let iz = idx / (nx * ny);
  let iy = (idx % (nx * ny)) / nx;
  let ix = idx % nx;

  let gx = params.originX + (f32(ix) + 0.5) * params.spacing;
  let gy = params.originY + (f32(iy) + 0.5) * params.spacing;
  let gz = params.originZ + (f32(iz) + 0.5) * params.spacing;

  let numTypes = params.numUniqueTypes;
  var ljVals: array<f32, 64>;
  for (var t: u32 = 0u; t < 64u; t = t + 1u) { ljVals[t] = 0.0; }

  var coulombPot: f32 = 0.0;

  for (var pi: u32 = 0u; pi < params.numProtein; pi = pi + 1u) {
    let base = pi * 6u;
    let dx = gx - proteinAtoms[base];
    let dy = gy - proteinAtoms[base + 1u];
    let dz = gz - proteinAtoms[base + 2u];
    let r2 = dx * dx + dy * dy + dz * dz;
    if (r2 > 400.0 || r2 == 0.0) { continue; }
    let r = sqrt(r2);
    let pCharge = proteinAtoms[base + 3u];
    let pSigma = proteinAtoms[base + 4u];
    let pEps = proteinAtoms[base + 5u];

    for (var lt: u32 = 0u; lt < numTypes; lt = lt + 1u) {
      let lSigma = ligParams[lt * 2u];
      let lEps = ligParams[lt * 2u + 1u];
      let sigma = 0.5 * (lSigma + pSigma);
      let epsilon = sqrt(lEps * pEps);
      let sr = sigma / r;
      let sr6 = sr * sr * sr * sr * sr * sr;
      let sr12 = sr6 * sr6;
      var lj = 4.0 * epsilon * (sr12 - sr6);
      lj = min(lj, 10.0);
      ljVals[lt] = ljVals[lt] + lj;
    }

    coulombPot = coulombPot + pCharge / (dielectric(r) * r);
  }

  let layerSize = nx * ny * nz;
  for (var lt: u32 = 0u; lt < numTypes; lt = lt + 1u) {
    gridOut[lt * layerSize + idx] = ljVals[lt];
  }
  gridOut[params.coulLayer * layerSize + idx] = coulombPot;
}
`;

export const SHADER_GRID_SRC = `
struct Params {
  numProtein: u32,
  numLigand: u32,
  numPoses: u32,
  _pad: u32,
};

struct GridParams {
  originX: f32, originY: f32, originZ: f32,
  spacing: f32,
  dimX: u32, dimY: u32, dimZ: u32,
  numLigandTypes: u32,
  coulLayer: u32,
  _pad: u32,
};

@group(0) @binding(0) var<storage, read> ligandAtoms: array<f32>;
@group(0) @binding(1) var<storage, read> poses: array<f32>;
@group(0) @binding(2) var<storage, read_write> energies: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read> gridData: array<f32>;
@group(0) @binding(5) var<uniform> gridParams: GridParams;

const COULOMB_K: f32 = 332.0636;

fn trilinearInterp(gx: f32, gy: f32, gz: f32, layerBase: u32) -> f32 {
  let ix = u32(floor(gx));
  let iy = u32(floor(gy));
  let iz = u32(floor(gz));
  let fx = gx - f32(ix);
  let fy = gy - f32(iy);
  let fz = gz - f32(iz);

  let dimX = gridParams.dimX;
  let dimY = gridParams.dimY;
  let dimZ = gridParams.dimZ;
  let strideXY = dimX * dimY;

  let x0 = min(ix, dimX - 1u);
  let x1 = min(ix + 1u, dimX - 1u);
  let y0 = min(iy, dimY - 1u);
  let y1 = min(iy + 1u, dimY - 1u);
  let z0 = min(iz, dimZ - 1u);
  let z1 = min(iz + 1u, dimZ - 1u);

  let v000 = gridData[layerBase + x0 + y0 * dimX + z0 * strideXY];
  let v001 = gridData[layerBase + x0 + y0 * dimX + z1 * strideXY];
  let v010 = gridData[layerBase + x0 + y1 * dimX + z0 * strideXY];
  let v011 = gridData[layerBase + x0 + y1 * dimX + z1 * strideXY];
  let v100 = gridData[layerBase + x1 + y0 * dimX + z0 * strideXY];
  let v101 = gridData[layerBase + x1 + y0 * dimX + z1 * strideXY];
  let v110 = gridData[layerBase + x1 + y1 * dimX + z0 * strideXY];
  let v111 = gridData[layerBase + x1 + y1 * dimX + z1 * strideXY];

  let c00 = v000 * (1.0 - fx) + v100 * fx;
  let c01 = v001 * (1.0 - fx) + v101 * fx;
  let c10 = v010 * (1.0 - fx) + v110 * fx;
  let c11 = v011 * (1.0 - fx) + v111 * fx;
  let c0 = c00 * (1.0 - fy) + c10 * fy;
  let c1 = c01 * (1.0 - fy) + c11 * fy;
  return c0 * (1.0 - fz) + c1 * fz;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let poseIdx = gid.x;
  if (poseIdx >= params.numPoses) { return; }

  let base = poseIdx * 12u;
  let r0 = poses[base + 0u]; let r1 = poses[base + 1u]; let r2 = poses[base + 2u];
  let r3 = poses[base + 3u]; let r4 = poses[base + 4u]; let r5 = poses[base + 5u];
  let r6 = poses[base + 6u]; let r7 = poses[base + 7u]; let r8 = poses[base + 8u];
  let tx = poses[base + 9u]; let ty = poses[base + 10u]; let tz = poses[base + 11u];

  var totalEnergy: f32 = 0.0;

  let invSpacing = 1.0 / gridParams.spacing;

  for (var i: u32 = 0u; i < params.numLigand; i = i + 1u) {
    let lBase = i * 6u;
    let lx0 = ligandAtoms[lBase];
    let ly0 = ligandAtoms[lBase + 1u];
    let lz0 = ligandAtoms[lBase + 2u];
    let lCharge = ligandAtoms[lBase + 3u];
    let layerIdx = u32(ligandAtoms[lBase + 4u]);

    let wx = r0 * lx0 + r1 * ly0 + r2 * lz0 + tx;
    let wy = r3 * lx0 + r4 * ly0 + r5 * lz0 + ty;
    let wz = r6 * lx0 + r7 * ly0 + r8 * lz0 + tz;

    let gx = (wx - gridParams.originX) * invSpacing;
    let gy = (wy - gridParams.originY) * invSpacing;
    let gz = (wz - gridParams.originZ) * invSpacing;

    if (gx < 0.0 || gy < 0.0 || gz < 0.0 ||
        gx >= f32(gridParams.dimX - 1u) ||
        gy >= f32(gridParams.dimY - 1u) ||
        gz >= f32(gridParams.dimZ - 1u)) {
      continue;
    }

    let layerSize = gridParams.dimX * gridParams.dimY * gridParams.dimZ;
    let coulombPot = trilinearInterp(gx, gy, gz, gridParams.coulLayer * layerSize);
    let lj = trilinearInterp(gx, gy, gz, layerIdx * layerSize);

    totalEnergy = totalEnergy + lj + COULOMB_K * lCharge * coulombPot;
  }

  totalEnergy = min(totalEnergy, 10000.0);
  energies[poseIdx] = totalEnergy;
}
`;
