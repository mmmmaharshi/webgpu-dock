import type { BenchResult, DemoOptions, AD4Params, LigandAtom } from "./types";
import { bar } from "./ui";
import { parsePDBQT, parseLigandPDBQT, loadAD4Params, atomsToArrayReal } from "./parsers";
import { buildPoses, applyTorsions, generateConformers, computeCenter, toLocalOrigin } from "./geometry";
import { createSinglePoseScorer, scoreAllPosesGPUWithGrid, bestOf, evaluatePose } from "./scoring";
import { buildAffinityGrid, atomsToArrayRealGrid } from "./grid";
import { bfgsRefine } from "./optimize";

export async function runDemo({
  proteinPDBQT = "data/protein.pdbqt",
  ligandPDBQT = "data/ligand.pdbqt",
  numRotations = 100,
  translationRange = 4,
  translationStep = 1,
  numConformers = 20,
  systemName = "Unknown",
  knownCenter = [0, 0, 0],
}: DemoOptions = {}): Promise<BenchResult | undefined> {
  const tStart = performance.now();

  const ad4Table = await bar.run("Loading AD4 parameters...", async (pg) => {
    const tbl = await loadAD4Params();
    pg(100);
    return tbl;
  });
  console.log(
    `Loaded AD4 table with ${Object.keys(ad4Table).length} atom types`,
  );

  const [proteinAtomsRaw, ligandResult] = await bar.run(
    "Loading PDB/PDBQT files...",
    async (pg) => {
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
    },
  );
  const { atoms: rawLigandAtoms, serialToIndex, branches } = ligandResult;
  const pocketCenter = {
    x: knownCenter[0],
    y: knownCenter[1],
    z: knownCenter[2],
  };
  const ligandAtomsBase = toLocalOrigin(rawLigandAtoms);
  console.log(
    `Pocket center: (${pocketCenter.x.toFixed(3)}, ${pocketCenter.y.toFixed(3)}, ${pocketCenter.z.toFixed(3)})`,
  );
  console.log(
    `Known center:  (${knownCenter[0].toFixed(3)}, ${knownCenter[1].toFixed(3)}, ${knownCenter[2].toFixed(3)})`,
  );
  console.log(
    `Parsed ${proteinAtomsRaw.length} protein atoms, ${ligandAtomsBase.length} ligand atoms, ${branches.length} rotatable bonds`,
  );

  const proteinAtoms = await bar.run("Converting atoms...", async (pg) => {
    pg(50);
    const pa = atomsToArrayReal(proteinAtomsRaw, ad4Table);
    pg(100);
    return pa;
  });
  const numProtein = proteinAtomsRaw.length;
  const numLigand = ligandAtomsBase.length;

  let tGrid = 0;
  const grid = await bar.run("Building affinity grid...", async (pg) => {
    const t0 = performance.now();
    const ligandTypes = [...new Set(ligandAtomsBase.map(a => a.atomType))];
    const g = await buildAffinityGrid(proteinAtomsRaw, ad4Table, ligandTypes, pocketCenter, translationRange, 0.375, 6, pg);
    tGrid = performance.now() - t0;
    return g;
  });
  console.log(
    `Grid: ${grid.dims[0]}×${grid.dims[1]}×${grid.dims[2]} × ${grid.numLayers} layers, ${grid.spacing.toFixed(3)}Å spacing, ${tGrid.toFixed(0)}ms`,
  );

  const {
    poses,
    numPoses,
    numRotations: rCount,
    numTranslations,
  } = await bar.run("Building pose set...", async (pg) => {
    const result = buildPoses(numRotations, translationRange, translationStep, pocketCenter);
    pg(100);
    return result;
  });
  bar.setStatus(
    `Pose set: ${numPoses} poses (${rCount} rotations × ${numTranslations} translations)`,
  );
  console.log(
    `Pose set: ${rCount} rotations × ${numTranslations} translations = ${numPoses} poses`,
  );

  let adapter: GPUAdapter | null = null;
  if ("gpu" in navigator) {
    adapter = await navigator.gpu.requestAdapter();
  }
  if (!adapter) {
    console.log("WebGPU adapter not available");
    bar.setStatus("WebGPU required");
    bar.done();
    return;
  }

  const conformers = generateConformers(numConformers, branches.length);

  const perConformerBest: { energy: number; bestIdx: number; cIdx: number }[] = [];

  const tCoarse = performance.now();
  for (let cIdx = 0; cIdx < conformers.length; cIdx++) {
    bar.setStatus(
      `[${systemName}] ${(((cIdx + 1) / conformers.length) * 100).toFixed(0)}%`,
    );
    bar.setProgress((cIdx / conformers.length) * 100);

    const bentAtoms = applyTorsions(
      ligandAtomsBase,
      serialToIndex,
      branches,
      conformers[cIdx],
    );
    const bc = computeCenter(bentAtoms);
    console.log(
      `  Conf #${cIdx}: bent center (${bc.x.toFixed(4)}, ${bc.y.toFixed(4)}, ${bc.z.toFixed(4)}) — shift from origin: ${Math.sqrt(bc.x**2 + bc.y**2 + bc.z**2).toFixed(4)} Å`,
    );
    const ligandArr = atomsToArrayRealGrid(bentAtoms, grid.typeToLayer, ad4Table);

    const { energies } = await scoreAllPosesGPUWithGrid(
      grid.data, grid.origin, grid.dims, grid.spacing,
      grid.numLayers - 1, grid.numLayers - 1,
      ligandArr, numLigand, poses, numPoses,
    );
    const { best, bestIdx } = bestOf(energies);
    perConformerBest.push({ energy: best, bestIdx, cIdx });
  }

  const tCoarseMs = performance.now() - tCoarse;

  perConformerBest.sort((a, b) => a.energy - b.energy);
  const NUM_CANDIDATES = 3;
  const topCandidates = perConformerBest.slice(0, NUM_CANDIDATES);

  let globalBest = topCandidates.length ? topCandidates[0].energy : Infinity;
  let globalBestIdx = topCandidates.length ? topCandidates[0].bestIdx : -1;
  let globalBestConf = topCandidates.length ? topCandidates[0].cIdx : -1;

  const tBFGS = performance.now();
  let bestPoseArr = poses;
  let bestPoseIdx = globalBestIdx;
  let bestBentAtoms: LigandAtom[] = [];

  if (topCandidates.length) {
    let refinedBest = Infinity;
    let refinedPoseArr = poses;
    let refinedPoseIdx = globalBestIdx;
    let refinedBentAtoms: LigandAtom[] = [];
    let refinedConf = globalBestConf;
    let refinedSourceIdx = globalBestIdx;
    const coarseBest = topCandidates[0].energy;

    for (let k = 0; k < topCandidates.length; k++) {
      const cand = topCandidates[k];
      const pb = cand.bestIdx * 12;
      const startCenter = {
        x: poses[pb + 9],
        y: poses[pb + 10],
        z: poses[pb + 11],
      };
      const startRotation = Array.from(poses.slice(pb, pb + 9));
      const bentAtoms = applyTorsions(ligandAtomsBase, serialToIndex, branches, conformers[cand.cIdx]);
      const baseArr = atomsToArrayReal(bentAtoms, ad4Table);
      const scorer = await createSinglePoseScorer(proteinAtoms, numProtein, baseArr, numLigand);

      bar.setStatus(`[${systemName}] BFGS refine (candidate ${k + 1}/${topCandidates.length})...`);
      const bfgsResult = await bfgsRefine(scorer, startCenter, startRotation, 30, 1e-4);
      scorer.destroy();

      const candidateFinalEnergy = Math.min(bfgsResult.energy, cand.energy);
      if (candidateFinalEnergy < refinedBest) {
        refinedBest = candidateFinalEnergy;
        refinedBentAtoms = bentAtoms;
        refinedConf = cand.cIdx;
        refinedSourceIdx = cand.bestIdx;
        if (bfgsResult.energy < cand.energy) {
          refinedPoseIdx = 0;
          refinedPoseArr = new Float32Array([
            ...bfgsResult.rotation,
            bfgsResult.center.x, bfgsResult.center.y, bfgsResult.center.z,
          ]);
        } else {
          refinedPoseIdx = cand.bestIdx;
          refinedPoseArr = poses;
        }
      }
    }

    if (refinedBest < coarseBest) {
      console.log(`BFGS improved: ${coarseBest.toFixed(3)} → ${refinedBest.toFixed(3)} kcal/mol (best of ${topCandidates.length} refined candidates)`);
    }
    if (refinedConf !== globalBestConf || refinedSourceIdx !== globalBestIdx) {
      console.log(`  (winner was coarse candidate conformer #${refinedConf}, pose #${refinedSourceIdx}, not the top single coarse score)`);
    }
    globalBest = refinedBest;
    globalBestConf = refinedConf;
    globalBestIdx = refinedSourceIdx;
    bestPoseArr = refinedPoseArr;
    bestPoseIdx = refinedPoseIdx;
    bestBentAtoms = refinedBentAtoms;

    const bestArr = atomsToArrayReal(bestBentAtoms, ad4Table);
    const { clashes, ljSum, coulombSum, nearClashes } = evaluatePose(bestArr, numLigand, proteinAtoms, numProtein, bestPoseArr, bestPoseIdx);
    console.log(`  Clashes: ${clashes}  Near-clashes: ${nearClashes}  LJ: ${ljSum.toFixed(1)}  Coulomb: ${coulombSum.toFixed(1)}`);
  }

  bar.setProgress(100);
  const tBFGSms = performance.now() - tBFGS;
  const totalSec = ((performance.now() - tStart) / 1000).toFixed(1);

  const realCenter = knownCenter;
  const lcx =
    ligandAtomsBase.reduce((s, a) => s + a.x, 0) / ligandAtomsBase.length;
  const lcy =
    ligandAtomsBase.reduce((s, a) => s + a.y, 0) / ligandAtomsBase.length;
  const lcz =
    ligandAtomsBase.reduce((s, a) => s + a.z, 0) / ligandAtomsBase.length;
  const pb2 = bestPoseIdx * 12;
  const rx =
    bestPoseArr[pb2] * lcx + bestPoseArr[pb2 + 1] * lcy + bestPoseArr[pb2 + 2] * lcz + bestPoseArr[pb2 + 9];
  const ry =
    bestPoseArr[pb2 + 3] * lcx +
    bestPoseArr[pb2 + 4] * lcy +
    bestPoseArr[pb2 + 5] * lcz +
    bestPoseArr[pb2 + 10];
  const rz =
    bestPoseArr[pb2 + 6] * lcx +
    bestPoseArr[pb2 + 7] * lcy +
    bestPoseArr[pb2 + 8] * lcz +
    bestPoseArr[pb2 + 11];
  const dist = Math.sqrt(
    (rx - realCenter[0]) ** 2 +
      (ry - realCenter[1]) ** 2 +
      (rz - realCenter[2]) ** 2,
  );
  console.log(
    `\nBEST OVERALL: ${globalBest.toFixed(3)} kcal/mol (conformer #${globalBestConf}, pose #${globalBestIdx})`,
  );
  console.log(
    `Best pose center: (${rx.toFixed(2)}, ${ry.toFixed(2)}, ${rz.toFixed(2)})`,
  );
  console.log(
    `Real center (PDB ${systemName}): (${realCenter[0]}, ${realCenter[1]}, ${realCenter[2]})`,
  );
  console.log(
    `Distance from known position: ${dist.toFixed(2)} Å  ${dist < 2 ? "✓ Hit" : dist < 5 ? "○ Near" : "✗ Miss"}`,
  );

  console.log(`Timing: grid ${tGrid.toFixed(0)}ms + coarse ${tCoarseMs.toFixed(0)}ms + BFGS ${tBFGSms.toFixed(0)}ms = ${totalSec}s total`);
  bar.setStatus(
    `Best: ${globalBest.toFixed(1)} kcal/mol — ${dist.toFixed(1)}Å from known`,
  );
  bar.done();

  return {
    systemName,
    bestScore: globalBest,
    bestIdx: globalBestIdx,
    bestConf: globalBestConf,
    dist,
    bestCenter: [rx, ry, rz],
    knownCenter: realCenter,
    totalTime: parseFloat(totalSec),
  };
}

async function runBenchmark(): Promise<void> {
  const systems = [
    {
      name: "1IEP (imatinib)",
      prot: "systems/1iep/protein.pdbqt",
      lig: "systems/1iep/ligand.pdbqt",
      center: [15.614, 53.38, 15.455],
    },
    {
      name: "1HSG (indinavir)",
      prot: "systems/1hsg/protein.pdbqt",
      lig: "systems/1hsg/ligand.pdbqt",
      center: [13.073, 22.467, 5.557],
    },
    {
      name: "1STP (biotin)",
      prot: "systems/1stp/protein.pdbqt",
      lig: "systems/1stp/ligand.pdbqt",
      center: [11.118, 1.68, -10.755],
    },
    {
      name: "1AC8 (TMZ)",
      prot: "systems/1ac8/protein.pdbqt",
      lig: "systems/1ac8/ligand.pdbqt",
      center: [31.924, 93.444, 47.924],
    },
    {
      name: "3CE3 (Factor Xa)",
      prot: "systems/3ce3/protein.pdbqt",
      lig: "systems/3ce3/ligand.pdbqt",
      center: [20.402, 18.013, 56.855],
    },
    {
      name: "3TMN (thermolysin)",
      prot: "systems/3tmn/protein.pdbqt",
      lig: "systems/3tmn/ligand.pdbqt",
      center: [52.227, 15.518, -2.409],
    },
    {
      name: "7CPA (carboxypeptidase A)",
      prot: "systems/7cpa/protein.pdbqt",
      lig: "systems/7cpa/ligand.pdbqt",
      center: [49.836, 17.609, 36.272],
    },
  ];

  const tBenchStart = performance.now();
  const results: BenchResult[] = [];

  for (let i = 0; i < systems.length; i++) {
    const sys = systems[i];
    console.log(`\n${"=".repeat(60)}`);
    console.log(`  [${i + 1}/${systems.length}] ${sys.name}`);
    console.log(`${"=".repeat(60)}`);
    const r = await runDemo({
      proteinPDBQT: sys.prot,
      ligandPDBQT: sys.lig,
      systemName: sys.name,
      knownCenter: sys.center,
      numConformers: 20,
    });
    if (r) results.push(r);
  }

  const benchTotal = ((performance.now() - tBenchStart) / 1000).toFixed(1);

  console.log(`\n\n${"=".repeat(72)}`);
  console.log(`  BENCHMARK RESULTS`);
  console.log(`${"=".repeat(72)}`);
  console.log(
    `  ${"System".padEnd(24)} ${"Best Score".padEnd(14)} ${"Distance".padEnd(10)} ${"Time".padEnd(8)} ${"Status"}`,
  );
  console.log(
    `  ${"─".repeat(23)} ${"─".repeat(13)} ${"─".repeat(9)} ${"─".repeat(7)} ${"─".repeat(8)}`,
  );
  for (const r of results) {
    const status = r.dist < 2 ? "✓ Hit" : r.dist < 5 ? "○ Near" : "✗ Miss";
    console.log(
      `  ${r.systemName.padEnd(24)} ${(r.bestScore.toFixed(2) + " kcal/mol").padEnd(14)} ${(r.dist.toFixed(2) + " Å").padEnd(10)} ${(r.totalTime.toFixed(1) + "s").padEnd(8)} ${status}`,
    );
  }
  console.log(`${"=".repeat(72)}`);
  console.log(`  Total benchmark time: ${benchTotal}s`);
  console.log(`${"=".repeat(72)}`);

  bar.setStatus(
    `Benchmark: ${benchTotal}s — ${results.filter((r) => r.dist < 2).length}/${results.length} hits`,
  );
  bar.done();
}

runBenchmark();
