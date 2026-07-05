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
