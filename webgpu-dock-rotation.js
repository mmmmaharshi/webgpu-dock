const COULOMB_K = 332.0636; // kcal * Angstrom / (mol * e^2)

const SHADER_SRC = `
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

  for (var i: u32 = 0u; i < params.numLigand; i = i + 1u) {
    let lBase = i * 6u;
    let lx0 = ligandAtoms[lBase];
    let ly0 = ligandAtoms[lBase + 1u];
    let lz0 = ligandAtoms[lBase + 2u];
    let lCharge = ligandAtoms[lBase + 3u];
    let lSigma  = ligandAtoms[lBase + 4u];
    let lEps    = ligandAtoms[lBase + 5u];

    // rotate then translate ligand atom into pose position
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

      let r2v = max(dx * dx + dy * dy + dz * dz, 0.01);
      let r  = sqrt(r2v);

      let sigma   = 0.5 * (lSigma + pSigma);
      let epsilon = sqrt(lEps * pEps);
      let sr6  = pow(sigma / r, 6.0);
      let sr12 = sr6 * sr6;
      var lj = 4.0 * epsilon * (sr12 - sr6);
      lj = min(lj, 1e6);

      let coulomb = COULOMB_K * lCharge * pCharge / r;

      totalEnergy = totalEnergy + lj + coulomb;
    }
  }

  energies[poseIdx] = totalEnergy;
}
`;

// ---------------------------------------------------------------------------
// Synthetic system (swap for real PDB/SDF parsing later)
// ---------------------------------------------------------------------------

function randRange(a, b) {
  return a + Math.random() * (b - a);
}

function makeAtomArray(n, spread) {
  const arr = new Float32Array(n * 6);
  for (let i = 0; i < n; i++) {
    const base = i * 6;
    arr[base + 0] = randRange(-spread, spread);
    arr[base + 1] = randRange(-spread, spread);
    arr[base + 2] = randRange(-spread, spread);
    arr[base + 3] = randRange(-0.6, 0.6); // charge
    arr[base + 4] = randRange(2.8, 3.9); // sigma
    arr[base + 5] = randRange(0.05, 0.25); // epsilon
  }
  return arr;
}

// ---------------------------------------------------------------------------
// Rotation sampling: random unit quaternions -> 3x3 rotation matrices.
// Uniform over SO(3) (Marsaglia / Shoemake method).
// ---------------------------------------------------------------------------

function randomUnitQuaternion() {
  const u1 = Math.random(),
    u2 = Math.random(),
    u3 = Math.random();
  const s1 = Math.sqrt(1 - u1),
    s2 = Math.sqrt(u1);
  const x = s1 * Math.sin(2 * Math.PI * u2);
  const y = s1 * Math.cos(2 * Math.PI * u2);
  const z = s2 * Math.sin(2 * Math.PI * u3);
  const w = s2 * Math.cos(2 * Math.PI * u3);
  return [x, y, z, w];
}

function quatToMatrix(x, y, z, w) {
  // row-major 3x3
  return [
    1 - 2 * (y * y + z * z),
    2 * (x * y - w * z),
    2 * (x * z + w * y),
    2 * (x * y + w * z),
    1 - 2 * (x * x + z * z),
    2 * (y * z - w * x),
    2 * (x * z - w * y),
    2 * (y * z + w * x),
    1 - 2 * (x * x + y * y),
  ];
}

function generateRotations(numRotations) {
  const mats = [];
  // always include identity so "no rotation" is in the search space
  mats.push([1, 0, 0, 0, 1, 0, 0, 0, 1]);
  for (let i = 1; i < numRotations; i++) {
    const [x, y, z, w] = randomUnitQuaternion();
    mats.push(quatToMatrix(x, y, z, w));
  }
  return mats;
}

// ---------------------------------------------------------------------------
// Pose set: every rotation x every translation on a grid.
// ---------------------------------------------------------------------------

function buildPoses(numRotations, translationRange, translationStep) {
  const rotations = generateRotations(numRotations);

  const coords = [];
  for (let v = -translationRange; v <= translationRange; v += translationStep)
    coords.push(v);

  const numTranslations = coords.length ** 3;
  const numPoses = rotations.length * numTranslations;
  const poses = new Float32Array(numPoses * 12);

  let poseIdx = 0;
  for (const R of rotations) {
    for (const tx of coords) {
      for (const ty of coords) {
        for (const tz of coords) {
          const base = poseIdx * 12;
          poses.set(R, base);
          poses[base + 9] = tx;
          poses[base + 10] = ty;
          poses[base + 11] = tz;
          poseIdx++;
        }
      }
    }
  }

  return { poses, numPoses, numRotations: rotations.length, numTranslations };
}

// ---------------------------------------------------------------------------
// CPU baseline (mirrors the shader math term for term)
// ---------------------------------------------------------------------------

function scoreAllPosesCPU(
  proteinAtoms,
  numProtein,
  ligandAtoms,
  numLigand,
  poses,
  numPoses,
) {
  const energies = new Float32Array(numPoses);

  for (let p = 0; p < numPoses; p++) {
    const base = p * 12;
    const r0 = poses[base],
      r1 = poses[base + 1],
      r2 = poses[base + 2];
    const r3 = poses[base + 3],
      r4 = poses[base + 4],
      r5 = poses[base + 5];
    const r6 = poses[base + 6],
      r7 = poses[base + 7],
      r8 = poses[base + 8];
    const tx = poses[base + 9],
      ty = poses[base + 10],
      tz = poses[base + 11];

    let total = 0;
    for (let i = 0; i < numLigand; i++) {
      const lBase = i * 6;
      const lx0 = ligandAtoms[lBase],
        ly0 = ligandAtoms[lBase + 1],
        lz0 = ligandAtoms[lBase + 2];
      const lCharge = ligandAtoms[lBase + 3],
        lSigma = ligandAtoms[lBase + 4],
        lEps = ligandAtoms[lBase + 5];

      const lx = r0 * lx0 + r1 * ly0 + r2 * lz0 + tx;
      const ly = r3 * lx0 + r4 * ly0 + r5 * lz0 + ty;
      const lz = r6 * lx0 + r7 * ly0 + r8 * lz0 + tz;

      for (let j = 0; j < numProtein; j++) {
        const qBase = j * 6;
        const dx = lx - proteinAtoms[qBase];
        const dy = ly - proteinAtoms[qBase + 1];
        const dz = lz - proteinAtoms[qBase + 2];
        const pCharge = proteinAtoms[qBase + 3],
          pSigma = proteinAtoms[qBase + 4],
          pEps = proteinAtoms[qBase + 5];

        const r2v = Math.max(dx * dx + dy * dy + dz * dz, 0.01);
        const r = Math.sqrt(r2v);

        const sigma = 0.5 * (lSigma + pSigma);
        const epsilon = Math.sqrt(lEps * pEps);
        const sr6 = Math.pow(sigma / r, 6);
        const sr12 = sr6 * sr6;
        let lj = 4 * epsilon * (sr12 - sr6);
        lj = Math.min(lj, 1e6);

        const coulomb = (COULOMB_K * lCharge * pCharge) / r;

        total += lj + coulomb;
      }
    }
    energies[p] = total;
  }

  return energies;
}

// ---------------------------------------------------------------------------
// GPU pass
// ---------------------------------------------------------------------------

let _device = null;
let _pipeline = null;

async function ensureDevice() {
  if (_device) return _device;
  if (!("gpu" in navigator))
    throw new Error("WebGPU not available in this browser.");
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error("No WebGPU adapter found.");
  _device = await adapter.requestDevice();
  return _device;
}

async function scoreAllPosesGPU(
  proteinAtoms,
  numProtein,
  ligandAtoms,
  numLigand,
  poses,
  numPoses,
) {
  const device = await ensureDevice();

  if (!_pipeline) {
    const module = device.createShaderModule({ code: SHADER_SRC });
    _pipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module, entryPoint: "main" },
    });
  }

  const proteinBuf = device.createBuffer({
    size: proteinAtoms.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const ligandBuf = device.createBuffer({
    size: ligandAtoms.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const posesBuf = device.createBuffer({
    size: poses.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const energiesBuf = device.createBuffer({
    size: numPoses * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const readbackBuf = device.createBuffer({
    size: numPoses * 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const paramsBuf = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  device.queue.writeBuffer(proteinBuf, 0, proteinAtoms);
  device.queue.writeBuffer(ligandBuf, 0, ligandAtoms);
  device.queue.writeBuffer(posesBuf, 0, poses);
  device.queue.writeBuffer(
    paramsBuf,
    0,
    new Uint32Array([numProtein, numLigand, numPoses, 0]),
  );

  const bindGroup = device.createBindGroup({
    layout: _pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: proteinBuf } },
      { binding: 1, resource: { buffer: ligandBuf } },
      { binding: 2, resource: { buffer: posesBuf } },
      { binding: 3, resource: { buffer: energiesBuf } },
      { binding: 4, resource: { buffer: paramsBuf } },
    ],
  });

  const t0 = performance.now();

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(_pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(numPoses / 64));
  pass.end();
  encoder.copyBufferToBuffer(energiesBuf, 0, readbackBuf, 0, numPoses * 4);
  device.queue.submit([encoder.finish()]);

  await readbackBuf.mapAsync(GPUMapMode.READ);
  const t1 = performance.now();

  const mapped = new Float32Array(readbackBuf.getMappedRange());
  const energies = new Float32Array(mapped); // copy out before unmap
  readbackBuf.unmap();

  return { energies, gpuMs: t1 - t0 };
}

// ---------------------------------------------------------------------------
// Demo runner — prints to console
// ---------------------------------------------------------------------------

function bestOf(energies) {
  let best = Infinity,
    bestIdx = -1;
  for (let i = 0; i < energies.length; i++) {
    if (energies[i] < best) {
      best = energies[i];
      bestIdx = i;
    }
  }
  return { best, bestIdx };
}

async function runDemo({
  numProteinAtoms = 120,
  numLigandAtoms = 14,
  numRotations = 12, // includes identity + (numRotations-1) random rotations
  translationRange = 5, // Angstrom, half-range of grid
  translationStep = 1, // Angstrom
} = {}) {
  const proteinAtoms = makeAtomArray(numProteinAtoms, 8.0);
  const ligandAtoms = makeAtomArray(numLigandAtoms, 1.5);
  const {
    poses,
    numPoses,
    numRotations: rCount,
    numTranslations,
  } = buildPoses(numRotations, translationRange, translationStep);

  console.log(
    `System: ${numProteinAtoms} protein atoms, ${numLigandAtoms} ligand atoms`,
  );
  console.log(
    `Pose set: ${rCount} rotations x ${numTranslations} translations = ${numPoses} total poses`,
  );

  const cpuT0 = performance.now();
  const cpuEnergies = scoreAllPosesCPU(
    proteinAtoms,
    numProteinAtoms,
    ligandAtoms,
    numLigandAtoms,
    poses,
    numPoses,
  );
  const cpuMs = performance.now() - cpuT0;
  const cpuBest = bestOf(cpuEnergies);
  console.log(
    `CPU: ${cpuMs.toFixed(1)} ms, best energy ${cpuBest.best.toFixed(3)} kcal/mol at pose #${cpuBest.bestIdx}`,
  );

  const { energies: gpuEnergies, gpuMs } = await scoreAllPosesGPU(
    proteinAtoms,
    numProteinAtoms,
    ligandAtoms,
    numLigandAtoms,
    poses,
    numPoses,
  );
  const gpuBest = bestOf(gpuEnergies);
  console.log(
    `GPU: ${gpuMs.toFixed(1)} ms, best energy ${gpuBest.best.toFixed(3)} kcal/mol at pose #${gpuBest.bestIdx}`,
  );
  console.log(`Speedup: ${(cpuMs / gpuMs).toFixed(1)}x`);

  let maxDiff = 0;
  for (let i = 0; i < numPoses; i++) {
    if (Math.abs(cpuEnergies[i]) > 1e5) continue;
    maxDiff = Math.max(maxDiff, Math.abs(cpuEnergies[i] - gpuEnergies[i]));
  }
  console.log(
    `Max CPU/GPU energy diff: ${maxDiff.toExponential(2)} kcal/mol (should be small — float32 rounding only)`,
  );

  return {
    proteinAtoms,
    ligandAtoms,
    poses,
    cpuEnergies,
    gpuEnergies,
    cpuMs,
    gpuMs,
  };
}

runDemo();
