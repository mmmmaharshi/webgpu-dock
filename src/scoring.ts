import { COULOMB_K, DIELECTRIC_A, DIELECTRIC_B, DIELECTRIC_LAMBDA, DIELECTRIC_K } from "./constants";
import { SHADER_SRC } from "./shader";

// Mehler-Solmajer distance-dependent dielectric (same model as AutoDock4's
// default electrostatics). See constants.ts for why this matters: a bare
// 1/r Coulomb term blows up at short range and dominates the score.
function dielectric(r: number): number {
  return DIELECTRIC_A + DIELECTRIC_B / (1 + DIELECTRIC_K * Math.exp(-DIELECTRIC_LAMBDA * DIELECTRIC_B * r));
}

let _device: GPUDevice | null = null;
let _pipeline: GPUComputePipeline | null = null;

export function bestOf(energies: Float32Array): { best: number; bestIdx: number } {
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

export function evaluatePose(
  ligandAtoms: Float32Array,
  numLigand: number,
  proteinAtoms: Float32Array,
  numProtein: number,
  pose: Float32Array,
  poseOffset: number,
): { energy: number; clashes: number; ljSum: number; coulombSum: number; nearClashes: number } {
  const base = poseOffset * 12;
  const r0 = pose[base], r1 = pose[base + 1], r2 = pose[base + 2];
  const r3 = pose[base + 3], r4 = pose[base + 4], r5 = pose[base + 5];
  const r6 = pose[base + 6], r7 = pose[base + 7], r8 = pose[base + 8];
  const tx = pose[base + 9], ty = pose[base + 10], tz = pose[base + 11];

  let total = 0;
  let clashes = 0;
  let nearClashes = 0;
  let ljSum = 0;
  let coulombSum = 0;

  for (let i = 0; i < numLigand; i++) {
    const lBase = i * 6;
    const lx0 = ligandAtoms[lBase], ly0 = ligandAtoms[lBase + 1], lz0 = ligandAtoms[lBase + 2];
    const lCharge = ligandAtoms[lBase + 3], lSigma = ligandAtoms[lBase + 4], lEps = ligandAtoms[lBase + 5];

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

      const r2 = dx * dx + dy * dy + dz * dz;
      if (r2 < 1.0) clashes++;
      if (r2 >= 1.0 && r2 < 4.0) nearClashes++;
      const r2v = Math.max(r2, 1.0);
      const r = Math.sqrt(r2v);

      const sigma = 0.5 * (lSigma + pSigma);
      const epsilon = Math.sqrt(lEps * pEps);
      const sr6 = Math.pow(sigma / r, 6);
      const sr12 = sr6 * sr6;
      let lj = 4 * epsilon * (sr12 - sr6);
      lj = Math.min(lj, 10);

      const coulomb = (COULOMB_K * lCharge * pCharge) / (dielectric(r) * r);
      total += lj + coulomb;
      ljSum += lj;
      coulombSum += coulomb;
    }
  }
  if (clashes > 3) { total = 1e10; }
  total = Math.min(total, 10000);
  return { energy: total, clashes, ljSum, coulombSum, nearClashes };
}

export function scoreAllPosesCPU(
  proteinAtoms: Float32Array,
  numProtein: number,
  ligandAtoms: Float32Array,
  numLigand: number,
  poses: Float32Array,
  numPoses: number,
  onProgress?: (pct: number) => void,
): Float32Array {
  const energies = new Float32Array(numPoses);
  const reportEvery = Math.max(1, Math.floor(numPoses / 50));

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
    let clashCount = 0;
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

        if (dx * dx + dy * dy + dz * dz < 1.0) clashCount++;
        const r2v = Math.max(dx * dx + dy * dy + dz * dz, 1.0);
        const r = Math.sqrt(r2v);

        const sigma = 0.5 * (lSigma + pSigma);
        const epsilon = Math.sqrt(lEps * pEps);
        const sr6 = Math.pow(sigma / r, 6);
        const sr12 = sr6 * sr6;
        let lj = 4 * epsilon * (sr12 - sr6);
        lj = Math.min(lj, 10);

        const coulomb = (COULOMB_K * lCharge * pCharge) / (dielectric(r) * r);

        total += lj + coulomb;
      }
    }
    if (clashCount > 3) total = 1e10;
    total = Math.min(total, 10000);
    energies[p] = total;

    if (onProgress && p % reportEvery === 0) {
      onProgress((p / numPoses) * 100);
    }
  }
  if (onProgress) onProgress(100);

  return energies;
}

async function ensureDevice(): Promise<GPUDevice> {
  if (_device) return _device;
  if (!("gpu" in navigator))
    throw new Error("WebGPU not available in this browser.");
  let adapter: GPUAdapter | null = null;
  for (let attempt = 0; attempt < 3; attempt++) {
    adapter = await navigator.gpu.requestAdapter();
    if (adapter) break;
    await new Promise((r) => setTimeout(r, 500 * (attempt + 1)));
  }
  if (!adapter) throw new Error("No WebGPU adapter found.");
  _device = await adapter.requestDevice();
  return _device;
}

export async function scoreAllPosesGPU(
  proteinAtoms: Float32Array,
  numProtein: number,
  ligandAtoms: Float32Array,
  numLigand: number,
  poses: Float32Array,
  numPoses: number,
): Promise<{ energies: Float32Array; gpuMs: number }> {
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

  device.queue.writeBuffer(proteinBuf, 0, proteinAtoms.buffer);
  device.queue.writeBuffer(ligandBuf, 0, ligandAtoms.buffer);
  device.queue.writeBuffer(posesBuf, 0, poses.buffer);
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
  const energies = new Float32Array(mapped);
  readbackBuf.unmap();

  return { energies, gpuMs: t1 - t0 };
}
