import type { SinglePoseScorer } from "./scoring";

export interface BfgsResult {
  center: { x: number; y: number; z: number };
  rotation: number[];
  energy: number;
  iterations: number;
  converged: boolean;
}

function angleAxisToMatrix(ax: number, ay: number, az: number): number[] {
  const theta = Math.sqrt(ax * ax + ay * ay + az * az);
  if (theta < 1e-10) {
    return [1, 0, 0, 0, 1, 0, 0, 0, 1];
  }
  const ux = ax / theta, uy = ay / theta, uz = az / theta;
  const c = Math.cos(theta), s = Math.sin(theta), C = 1 - c;
  return [
    c + ux * ux * C,     ux * uy * C - uz * s, ux * uz * C + uy * s,
    uy * ux * C + uz * s, c + uy * uy * C,     uy * uz * C - ux * s,
    uz * ux * C - uy * s, uz * uy * C + ux * s, c + uz * uz * C,
  ];
}

function matrixToAngleAxis(m: number[]): [number, number, number] {
  const trace = m[0] + m[4] + m[8];
  if (trace > 2.999) {
    return [0, 0, 0];
  }
  const theta = Math.acos(Math.min(1, Math.max(-1, (trace - 1) / 2)));
  if (theta < 1e-6) {
    const x = 0.5 * (m[7] - m[5]);
    const y = 0.5 * (m[2] - m[6]);
    const z = 0.5 * (m[3] - m[1]);
    return [x, y, z];
  }
  const s = 1 / (2 * Math.sin(theta));
  return [s * (m[7] - m[5]), s * (m[2] - m[6]), s * (m[3] - m[1])];
}

function stateToPose(state: Float64Array): Float32Array {
  const rot = angleAxisToMatrix(state[3], state[4], state[5]);
  return new Float32Array([
    rot[0], rot[1], rot[2], rot[3], rot[4], rot[5], rot[6], rot[7], rot[8],
    state[0], state[1], state[2],
  ]);
}

async function computeGradient(
  scorer: SinglePoseScorer,
  state: Float64Array,
  eps: number,
): Promise<Float64Array> {
  const batch = new Float32Array(12 * 12);
  for (let i = 0; i < 6; i++) {
    const fwd = new Float64Array(state);
    fwd[i] += eps;
    batch.set(stateToPose(fwd), i * 12);

    const bck = new Float64Array(state);
    bck[i] -= eps;
    batch.set(stateToPose(bck), (i + 6) * 12);
  }

  const energies = await scorer.scoreBatch(batch, 12);
  const grad = new Float64Array(6);
  for (let i = 0; i < 6; i++) {
    grad[i] = (energies[i] - energies[i + 6]) / (2 * eps);
  }
  return grad;
}

export async function bfgsRefine(
  scorer: SinglePoseScorer,
  startCenter: { x: number; y: number; z: number },
  startRotation: number[],
  maxIter = 50,
  gtol = 1e-4,
): Promise<BfgsResult> {
  const aa = matrixToAngleAxis(startRotation);
  const state = new Float64Array([startCenter.x, startCenter.y, startCenter.z, aa[0], aa[1], aa[2]]);

  const H = new Float64Array(36);
  for (let i = 0; i < 6; i++) H[i * 6 + i] = 0.1;

  const eps = 0.001;

  let pose = stateToPose(state);
  let currentEnergy = await scorer.score(pose);
  let currentGrad = await computeGradient(scorer, state, eps);

  let prevEnergy = currentEnergy + 1;

  let iter = 0;
  for (; iter < maxIter; iter++) {
    if (Math.abs(prevEnergy - currentEnergy) < 1e-6 && iter > 0) break;

    let gradNorm = 0;
    for (let i = 0; i < 6; i++) gradNorm += currentGrad[i] * currentGrad[i];
    if (Math.sqrt(gradNorm) < gtol) break;

    const dir = new Float64Array(6);
    for (let i = 0; i < 6; i++) {
      let sum = 0;
      for (let j = 0; j < 6; j++) sum += H[j * 6 + i] * currentGrad[j];
      dir[i] = -sum;
    }

    const gd = dir.reduce((s, d, i) => s + d * currentGrad[i], 0);
    if (gd >= 0) {
      for (let i = 0; i < 6; i++) dir[i] = -currentGrad[i];
    }

    const c1 = 1e-4;
    let alpha = 1.0;
    let newState: Float64Array | null = null;
    let newEnergy = Infinity;

    for (let ls = 0; ls < 20; ls++) {
      const cand = new Float64Array(state);
      for (let i = 0; i < 6; i++) cand[i] += alpha * dir[i];
      const e = await scorer.score(stateToPose(cand));
      if (e <= currentEnergy + c1 * alpha * gd) {
        newState = cand;
        newEnergy = e;
        break;
      }
      alpha *= 0.5;
    }

    if (!newState) break;

    const s = new Float64Array(6);
    for (let i = 0; i < 6; i++) s[i] = newState[i] - state[i];

    const newGrad = await computeGradient(scorer, newState, eps);

    prevEnergy = currentEnergy;
    currentEnergy = newEnergy;

    const y = new Float64Array(6);
    for (let i = 0; i < 6; i++) y[i] = newGrad[i] - currentGrad[i];

    const ys = y.reduce((sum, yi, i) => sum + yi * s[i], 0);
    if (ys > 1e-10) {
      const rho = 1 / ys;
      const Hy = new Float64Array(6);
      for (let i = 0; i < 6; i++) {
        let sum = 0;
        for (let j = 0; j < 6; j++) sum += H[j * 6 + i] * y[j];
        Hy[i] = sum;
      }
      const yHy = y.reduce((sum, yi, i) => sum + yi * Hy[i], 0);
      const factor = rho * yHy + 1;
      for (let i = 0; i < 6; i++) {
        for (let j = 0; j < 6; j++) {
          H[j * 6 + i] += rho * (factor * s[i] * s[j] - Hy[i] * s[j] - s[i] * Hy[j]);
        }
      }
    }

    for (let i = 0; i < 6; i++) state[i] = newState[i];
    currentGrad = newGrad;
    pose = stateToPose(state);
  }

  const finalEnergy = await scorer.score(stateToPose(state));
  const rot = angleAxisToMatrix(state[3], state[4], state[5]);

  return {
    center: { x: state[0], y: state[1], z: state[2] },
    rotation: rot,
    energy: finalEnergy,
    iterations: iter,
    converged: iter < maxIter,
  };
}
