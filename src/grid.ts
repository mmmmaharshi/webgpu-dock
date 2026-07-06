import type { Atom, AD4Params } from "./types";
import { RII_TO_SIGMA } from "./constants";
import { SHADER_BUILD_GRID_SRC } from "./shader";

export interface GridData {
  data: Float32Array;
  dims: [number, number, number];
  origin: [number, number, number];
  spacing: number;
  typeToLayer: Record<string, number>;
  numLayers: number;
}

let _device: GPUDevice | null = null;
let _pipeline: GPUComputePipeline | null = null;

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

export async function buildAffinityGrid(
  proteinAtoms: Atom[],
  ad4Table: Record<string, AD4Params>,
  ligandTypes: string[],
  pocketCenter: { x: number; y: number; z: number },
  translationRange: number,
  spacing = 0.375,
  padding = 4,
  onProgress?: (pct: number) => void,
): Promise<GridData> {
  const halfExtent = translationRange + padding;
  const nx = Math.ceil((2 * halfExtent) / spacing);
  const ny = Math.ceil((2 * halfExtent) / spacing);
  const nz = Math.ceil((2 * halfExtent) / spacing);
  const dims: [number, number, number] = [nx, ny, nz];

  const origin: [number, number, number] = [
    pocketCenter.x - halfExtent,
    pocketCenter.y - halfExtent,
    pocketCenter.z - halfExtent,
  ];

  const uniqueTypes = [...new Set(ligandTypes)];
  const typeToLayer: Record<string, number> = {};
  uniqueTypes.forEach((t, i) => { typeToLayer[t] = i; });
  const numUniqueTypes = uniqueTypes.length;
  const numLayers = numUniqueTypes + 1;
  const coulLayer = numUniqueTypes;
  const layerSize = nx * ny * nz;
  const totalSize = numLayers * layerSize;

  const device = await ensureDevice();

  if (!_pipeline) {
    const module = device.createShaderModule({ code: SHADER_BUILD_GRID_SRC });
    _pipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module, entryPoint: "main" },
    });
  }

  const protArr = new Float32Array(proteinAtoms.length * 6);
  for (let i = 0; i < proteinAtoms.length; i++) {
    const prot = proteinAtoms[i];
    const base = i * 6;
    protArr[base] = prot.x;
    protArr[base + 1] = prot.y;
    protArr[base + 2] = prot.z;
    protArr[base + 3] = prot.charge;
    const p = ad4Table[prot.atomType];
    protArr[base + 4] = p ? p.Rii * RII_TO_SIGMA : 4.0 * RII_TO_SIGMA;
    protArr[base + 5] = p ? p.epsii : 0.15;
  }

  const ligArr = new Float32Array(numUniqueTypes * 2);
  for (let i = 0; i < numUniqueTypes; i++) {
    const p = ad4Table[uniqueTypes[i]];
    ligArr[i * 2] = p ? p.Rii * RII_TO_SIGMA : 4.0 * RII_TO_SIGMA;
    ligArr[i * 2 + 1] = p ? p.epsii : 0.15;
  }

  const protBuf = device.createBuffer({
    size: protArr.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const ligBuf = device.createBuffer({
    size: ligArr.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const gridBuf = device.createBuffer({
    size: totalSize * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const paramsBuf = device.createBuffer({
    size: 48,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const readbackBuf = device.createBuffer({
    size: totalSize * 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  device.queue.writeBuffer(protBuf, 0, protArr.buffer);
  device.queue.writeBuffer(ligBuf, 0, ligArr.buffer);
  {
    const ab = new ArrayBuffer(48);
    const f32 = new Float32Array(ab);
    const u32 = new Uint32Array(ab);
    f32[0] = origin[0]; f32[1] = origin[1]; f32[2] = origin[2];
    f32[3] = spacing;
    u32[4] = nx; u32[5] = ny; u32[6] = nz;
    u32[7] = proteinAtoms.length;
    u32[8] = numUniqueTypes;
    u32[9] = coulLayer;
    device.queue.writeBuffer(paramsBuf, 0, ab);
  }

  const bindGroup = device.createBindGroup({
    layout: _pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: protBuf } },
      { binding: 1, resource: { buffer: ligBuf } },
      { binding: 2, resource: { buffer: gridBuf } },
      { binding: 3, resource: { buffer: paramsBuf } },
    ],
  });

  if (onProgress) onProgress(10);

  const t0 = performance.now();
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(_pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil((nx * ny * nz) / 64));
  pass.end();
  encoder.copyBufferToBuffer(gridBuf, 0, readbackBuf, 0, totalSize * 4);
  device.queue.submit([encoder.finish()]);
  await readbackBuf.mapAsync(GPUMapMode.READ);
  const t1 = performance.now();

  if (onProgress) onProgress(90);

  const mapped = new Float32Array(readbackBuf.getMappedRange());
  const gridData = new Float32Array(mapped);
  readbackBuf.unmap();

  protBuf.destroy();
  ligBuf.destroy();
  gridBuf.destroy();
  paramsBuf.destroy();
  readbackBuf.destroy();

  if (onProgress) onProgress(100);

  console.log(`Grid built on GPU in ${(t1 - t0).toFixed(1)}ms`);

  return { data: gridData, dims, origin, spacing, typeToLayer, numLayers };
}

export function atomsToArrayRealGrid(
  baseAtoms: Atom[],
  typeToLayer: Record<string, number>,
  ad4Table: Record<string, AD4Params>,
): Float32Array {
  const arr = new Float32Array(baseAtoms.length * 6);
  baseAtoms.forEach((a, i) => {
    const base = i * 6;
    arr[base] = a.x;
    arr[base + 1] = a.y;
    arr[base + 2] = a.z;
    arr[base + 3] = a.charge;
    arr[base + 4] = typeToLayer[a.atomType] ?? 0;
    const params = ad4Table[a.atomType];
    arr[base + 5] = params ? params.epsii : 0.15;
  });
  return arr;
}
