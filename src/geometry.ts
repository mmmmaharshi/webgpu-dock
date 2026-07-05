import type { LigandAtom, Branch, PoseSet } from "./types";

export function randomUnitQuaternion(): [number, number, number, number] {
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

export function quatToMatrix(x: number, y: number, z: number, w: number): number[] {
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

export function generateRotations(numRotations: number): number[][] {
  const mats: number[][] = [];
  mats.push([1, 0, 0, 0, 1, 0, 0, 0, 1]);
  for (let i = 1; i < numRotations; i++) {
    const [x, y, z, w] = randomUnitQuaternion();
    mats.push(quatToMatrix(x, y, z, w));
  }
  return mats;
}

export function buildPoses(
  numRotations: number,
  translationRange: number,
  translationStep: number,
): PoseSet {
  const rotations = generateRotations(numRotations);

  const coords: number[] = [];
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

export function rotateAroundAxis(
  px: number,
  py: number,
  pz: number,
  ax: number,
  ay: number,
  az: number,
  dx: number,
  dy: number,
  dz: number,
  angle: number,
): [number, number, number] {
  const vx = px - ax,
    vy = py - ay,
    vz = pz - az;
  const cosA = Math.cos(angle),
    sinA = Math.sin(angle);
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

export function applyTorsions(
  atoms: LigandAtom[],
  serialToIndex: Record<number, number>,
  branches: Branch[],
  angles: number[],
): LigandAtom[] {
  const coords = atoms.map((a) => ({ x: a.x, y: a.y, z: a.z }));
  branches.forEach((branch: Branch, i: number) => {
    const angle = angles[i];
    if (!angle) return;
    const pIdx = serialToIndex[branch.parentSerial];
    const cIdx = serialToIndex[branch.childSerial];
    if (pIdx === undefined || cIdx === undefined) return;
    const px = coords[pIdx].x,
      py = coords[pIdx].y,
      pz = coords[pIdx].z;
    const cx = coords[cIdx].x,
      cy = coords[cIdx].y,
      cz = coords[cIdx].z;
    let dx = cx - px,
      dy = cy - py,
      dz = cz - pz;
    const len = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1;
    dx /= len;
    dy /= len;
    dz /= len;
    for (const idx of branch.indices) {
      const c = coords[idx];
      const [nx, ny, nz] = rotateAroundAxis(
        c.x,
        c.y,
        c.z,
        px,
        py,
        pz,
        dx,
        dy,
        dz,
        angle,
      );
      c.x = nx;
      c.y = ny;
      c.z = nz;
    }
  });
  return atoms.map((a, i) => ({
    ...a,
    x: coords[i].x,
    y: coords[i].y,
    z: coords[i].z,
  }));
}

export function checkBondSanity(
  atoms: LigandAtom[],
  serialToIndex: Record<number, number>,
  branches: Branch[],
  label: string,
): void {
  for (const branch of branches) {
    const pIdx = serialToIndex[branch.parentSerial];
    const cIdx = serialToIndex[branch.childSerial];
    const p = atoms[pIdx],
      c = atoms[cIdx];
    const dist = Math.sqrt(
      (p.x - c.x) ** 2 + (p.y - c.y) ** 2 + (p.z - c.z) ** 2,
    );
    console.log(
      `  ${label} bond ${branch.parentSerial}-${branch.childSerial}: ${dist.toFixed(2)} Å`,
    );
  }
}

export function generateConformers(
  numConformers: number,
  numBranches: number,
): number[][] {
  if (numBranches === 0) return [new Array(0)];
  const arr: number[][] = [new Array(numBranches).fill(0)];
  for (let i = 1; i < numConformers; i++) {
    const angles = new Array(numBranches).fill(0);
    const numBondsToMove = 1 + Math.floor(Math.random() * 3);
    const chosenBonds = new Set<number>();
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

export function computeCenter(atoms: { x: number; y: number; z: number }[]): {
  x: number;
  y: number;
  z: number;
} {
  let sx = 0,
    sy = 0,
    sz = 0;
  for (const a of atoms) {
    sx += a.x;
    sy += a.y;
    sz += a.z;
  }
  return { x: sx / atoms.length, y: sy / atoms.length, z: sz / atoms.length };
}

export function recenterLigand(
  atoms: LigandAtom[],
  targetCenter: { x: number; y: number; z: number },
): LigandAtom[] {
  const cur = computeCenter(atoms);
  const dx = targetCenter.x - cur.x;
  const dy = targetCenter.y - cur.y;
  const dz = targetCenter.z - cur.z;
  return atoms.map((a) => ({ ...a, x: a.x + dx, y: a.y + dy, z: a.z + dz }));
}
