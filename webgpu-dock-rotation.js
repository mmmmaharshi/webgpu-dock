// Redirect console output to the #log element on the page
{
  const el = () => document.getElementById('log');
  const { log: origLog, warn: origWarn } = console;
  console.log = (...args) => {
    origLog(...args);
    const d = el();
    if (d) d.textContent += args.join(' ') + '\n';
  };
  console.warn = (...args) => {
    origWarn(...args);
    const d = el();
    if (d) d.innerHTML += `<span class="warn">${args.join(' ')}</span>\n`;
  };
}

// Simple progress bar controller
const bar = (() => {
  const fill  = () => document.getElementById('bar-fill');
  const label = () => document.getElementById('bar-label');
  const statusEl = () => document.getElementById('status');
  return {
    setStatus(msg) {
      const s = statusEl();
      if (s) s.textContent = msg;
    },
    setProgress(pct) {
      const f = fill(), l = label();
      if (f) f.style.width = Math.min(pct, 100) + '%';
      if (l) l.textContent = Math.floor(Math.min(pct, 100)) + '%';
    },
    done() {
      this.setProgress(100);
      this.setStatus('Done');
    },
    /** Run an async operation with progress reporting – fn receives (setProgress, setStatus) */
    async run(initialStatus, fn) {
      this.setStatus(initialStatus);
      this.setProgress(0);
      const result = await fn(
        (pct) => this.setProgress(pct),
        (msg) => this.setStatus(msg),
      );
      return result;
    },
  };
})();

const COULOMB_K = 332.0636; // kcal * Angstrom / (mol * e^2)

// Convert AD4 Rii (energy-minimum distance) to LJ sigma.
// Standard relation: Rii = sigma * 2^(1/6)  =>  sigma = Rii / 2^(1/6)
const RII_TO_SIGMA = 1 / Math.pow(2, 1 / 6);

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
// PDBQT parser — extracts real coords, Gasteiger charges, AutoDock types
// ---------------------------------------------------------------------------

function parsePDBQT(text) {
  const atoms = [];
  for (const line of text.split("\n")) {
    const record = line.slice(0, 6).trim();
    if (record !== "ATOM" && record !== "HETATM") continue;

    const x = parseFloat(line.slice(30, 38));
    const y = parseFloat(line.slice(38, 46));
    const z = parseFloat(line.slice(46, 54));
    const charge = parseFloat(line.slice(70, 76));
    const atomType = line.slice(77, 79).trim();

    atoms.push({ x, y, z, charge, atomType });
  }
  return atoms;
}

// ---------------------------------------------------------------------------
// Ligand PDBQT parser — preserves BRANCH/ENDBRANCH torsion tree
// ---------------------------------------------------------------------------

function parseLigandPDBQT(text) {
  const atoms = [];
  const serialToIndex = {};
  const branches = [];
  const stack = [];
  let openCounter = 0;

  for (const rawLine of text.split("\n")) {
    const trimmed = rawLine.trim();
    if (!trimmed) continue;
    const tag = trimmed.split(/\s+/)[0];

    if (tag === "ATOM" || tag === "HETATM") {
      const serial = parseInt(rawLine.slice(6, 11), 10);
      const x = parseFloat(rawLine.slice(30, 38));
      const y = parseFloat(rawLine.slice(38, 46));
      const z = parseFloat(rawLine.slice(46, 54));
      const charge = parseFloat(rawLine.slice(70, 76));
      const atomType = rawLine.slice(77, 79).trim();

      const idx = atoms.length;
      atoms.push({ serial, x, y, z, charge, atomType });
      serialToIndex[serial] = idx;

      for (const b of stack) b.indices.add(idx);
    } else if (tag === "BRANCH") {
      const parts = trimmed.split(/\s+/);
      const branch = {
        parentSerial: parseInt(parts[1], 10),
        childSerial: parseInt(parts[2], 10),
        indices: new Set(),
        openOrder: openCounter++,
      };
      stack.push(branch);
      branches.push(branch);
    } else if (tag === "ENDBRANCH") {
      stack.pop();
    }
  }

  branches.sort((a, b) => a.openOrder - b.openOrder);
  return { atoms, serialToIndex, branches };
}

// ---------------------------------------------------------------------------
// Load real AD4 vdW parameters from AutoDock's public parameter file
// ---------------------------------------------------------------------------

async function loadAD4Params() {
  const url =
    "https://raw.githubusercontent.com/ccsb-scripps/AutoDock-Vina/develop/data/AD4_parameters.dat";
  const text = await fetch(url).then((r) => r.text());
  const table = {};
  for (const line of text.split("\n")) {
    if (!line.startsWith("atom_par")) continue;
    const parts = line.trim().split(/\s+/);
    table[parts[1]] = {
      Rii: parseFloat(parts[2]),
      epsii: parseFloat(parts[3]),
    };
  }
  return table;
}

// ---------------------------------------------------------------------------
// Convert parsed atoms + AD4 table → flat Float32Array (stride 6)
// ---------------------------------------------------------------------------

function atomsToArrayReal(atoms, ad4Table) {
  const arr = new Float32Array(atoms.length * 6);
  atoms.forEach((a, i) => {
    const params = ad4Table[a.atomType];
    if (!params) {
      console.warn(
        `No AD4 param for atom type "${a.atomType}", using generic carbon fallback`,
      );
    }
    const Rii = params ? params.Rii : 4.0;
    const epsii = params ? params.epsii : 0.15;

    const base = i * 6;
    arr[base] = a.x;
    arr[base + 1] = a.y;
    arr[base + 2] = a.z;
    arr[base + 3] = a.charge;
    arr[base + 4] = Rii * RII_TO_SIGMA;
    arr[base + 5] = epsii;
  });
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
// Torsion search helpers
// ---------------------------------------------------------------------------

function rotateAroundAxis(px, py, pz, ax, ay, az, dx, dy, dz, angle) {
  const vx = px - ax, vy = py - ay, vz = pz - az;
  const cosA = Math.cos(angle), sinA = Math.sin(angle);
  const dot = vx * dx + vy * dy + vz * dz;
  const crossX = dy * vz - dz * vy;
  const crossY = dz * vx - dx * vz;
  const crossZ = dx * vy - dy * vx;
  return [
    vx * cosA + crossX * sinA + dx * dot * (1 - cosA) + ax,
    vy * cosA + crossY * sinA + dy * dot * (1 - cosA) + ay,
    vz * cosA + crossZ * sinA + dz * dot * (1 - cosA) + az,
  ];
}

function applyTorsions(atoms, serialToIndex, branches, angles) {
  const coords = atoms.map(a => ({ x: a.x, y: a.y, z: a.z }));
  branches.forEach((branch, i) => {
    const angle = angles[i];
    if (!angle) return;
    const pIdx = serialToIndex[branch.parentSerial];
    const cIdx = serialToIndex[branch.childSerial];
    if (pIdx === undefined || cIdx === undefined) return;
    const px = coords[pIdx].x, py = coords[pIdx].y, pz = coords[pIdx].z;
    const cx = coords[cIdx].x, cy = coords[cIdx].y, cz = coords[cIdx].z;
    let dx = cx - px, dy = cy - py, dz = cz - pz;
    const len = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1;
    dx /= len; dy /= len; dz /= len;
    for (const idx of branch.indices) {
      const c = coords[idx];
      const [nx, ny, nz] = rotateAroundAxis(c.x, c.y, c.z, px, py, pz, dx, dy, dz, angle);
      c.x = nx; c.y = ny; c.z = nz;
    }
  });
  return atoms.map((a, i) => ({ ...a, x: coords[i].x, y: coords[i].y, z: coords[i].z }));
}

function checkBondSanity(atoms, serialToIndex, branches, label) {
  for (const branch of branches) {
    const pIdx = serialToIndex[branch.parentSerial];
    const cIdx = serialToIndex[branch.childSerial];
    const p = atoms[pIdx], c = atoms[cIdx];
    const dist = Math.sqrt((p.x - c.x) ** 2 + (p.y - c.y) ** 2 + (p.z - c.z) ** 2);
    console.log(`  ${label} bond ${branch.parentSerial}-${branch.childSerial}: ${dist.toFixed(2)} Å`);
  }
}

function generateConformers(numConformers, numBranches) {
  const arr = [new Array(numBranches).fill(0)]; // rigid baseline
  for (let i = 1; i < numConformers; i++) {
    const angles = new Array(numBranches).fill(0);
    const numBondsToMove = 1 + Math.floor(Math.random() * 3); // 1-3 bonds
    const chosenBonds = new Set();
    while (chosenBonds.size < numBondsToMove) {
      chosenBonds.add(Math.floor(Math.random() * numBranches));
    }
    for (const b of chosenBonds) {
      angles[b] = Math.random() * 2 * Math.PI;
    }
    arr.push(angles);
  }
  return arr;
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
  onProgress,
) {
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

    if (onProgress && p % reportEvery === 0) {
      onProgress((p / numPoses) * 100);
    }
  }
  if (onProgress) onProgress(100);

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
  proteinPDBQT = "protein.pdbqt",
  ligandPDBQT = "ligand.pdbqt",
  numRotations = 12,
  translationRange = 5,
  translationStep = 1,
  numConformers = 40,
} = {}) {
  const tStart = performance.now();

  // ── Stage 1: Load AD4 params ──────────────────────────────────────────
  const ad4Table = await bar.run('Loading AD4 parameters...', async (pg) => {
    const tbl = await loadAD4Params();
    pg(100);
    return tbl;
  });
  console.log(`Loaded AD4 table with ${Object.keys(ad4Table).length} atom types`);

  // ── Stage 2: Fetch & parse PDBQT files ────────────────────────────────
  const [proteinAtomsRaw, ligandResult] = await bar.run('Fetching PDBQT files...', async (pg) => {
    pg(10);
    const [pt, lt] = await Promise.all([
      fetch(proteinPDBQT).then((r) => r.text()),
      fetch(ligandPDBQT).then((r) => r.text()),
    ]);
    pg(50);
    const par = parsePDBQT(pt);
    const lar = parseLigandPDBQT(lt);
    pg(100);
    return [par, lar];
  });
  const { atoms: ligandAtomsBase, serialToIndex, branches } = ligandResult;
  console.log(`Parsed ${proteinAtomsRaw.length} protein atoms, ${ligandAtomsBase.length} ligand atoms, ${branches.length} rotatable bonds`);

  // ── Stage 3: Convert protein atoms to array ───────────────────────────
  const proteinAtoms = await bar.run('Converting atoms...', async (pg) => {
    pg(50);
    const pa = atomsToArrayReal(proteinAtomsRaw, ad4Table);
    pg(100);
    return pa;
  });
  const numProtein = proteinAtomsRaw.length;
  const numLigand = ligandAtomsBase.length;

  // ── Stage 4: Build pose set ───────────────────────────────────────────
  const { poses, numPoses, numRotations: rCount, numTranslations } = await bar.run('Building pose set...', async (pg) => {
    const result = buildPoses(numRotations, translationRange, translationStep);
    pg(100);
    return result;
  });
  bar.setStatus(`Pose set: ${numPoses} poses (${rCount} rotations × ${numTranslations} translations)`);
  console.log(`Pose set: ${rCount} rotations × ${numTranslations} translations = ${numPoses} poses`);

  if (!('gpu' in navigator)) {
    console.log('WebGPU not available — cannot run torsion search efficiently');
    bar.setStatus('WebGPU required');
    bar.done();
    return;
  }

  // ── Stage 5: Generate conformers & score ──────────────────────────────
  const conformers = generateConformers(numConformers, branches.length);
  console.log(`Searching ${numConformers} conformers × ${numPoses} poses = ${numConformers * numPoses} total`);

  let globalBest = Infinity;
  let globalBestIdx = -1;
  let globalBestConf = -1;
  let globalBestAngles = null;

  for (let cIdx = 0; cIdx < conformers.length; cIdx++) {
    bar.setStatus(`Conformer ${cIdx + 1}/${conformers.length}...`);
    bar.setProgress((cIdx / conformers.length) * 100);

    const bentAtoms = applyTorsions(ligandAtomsBase, serialToIndex, branches, conformers[cIdx]);
    if (cIdx > 0) checkBondSanity(bentAtoms, serialToIndex, branches, `conformer #${cIdx}`);
    const ligandArr = atomsToArrayReal(bentAtoms, ad4Table);

    const { energies, gpuMs } = await scoreAllPosesGPU(
      proteinAtoms, numProtein, ligandArr, numLigand, poses, numPoses,
    );
    const { best, bestIdx } = bestOf(energies);
    const tag = conformers[cIdx].every(a => a === 0) ? ' (rigid)' : '';
    console.log(`  Conformer #${cIdx}${tag}: best ${best.toFixed(2)} kcal/mol (pose #${bestIdx}, ${gpuMs.toFixed(0)}ms)`);

    if (best < globalBest) {
      globalBest = best;
      globalBestIdx = bestIdx;
      globalBestConf = cIdx;
      globalBestAngles = conformers[cIdx];
    }
  }

  bar.setProgress(100);
  const totalSec = ((performance.now() - tStart) / 1000).toFixed(1);

  console.log(`\nBEST OVERALL: ${globalBest.toFixed(3)} kcal/mol (conformer #${globalBestConf}, pose #${globalBestIdx})`);
  console.log(`Total run time: ${totalSec}s`);
  bar.setStatus(`Best: ${globalBest.toFixed(1)} kcal/mol — Done`);
  bar.done();

  return { globalBest, globalBestIdx, globalBestConf, globalBestAngles };
}

runDemo();
