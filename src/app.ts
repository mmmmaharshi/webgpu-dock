import type { BenchResult, DemoOptions, AD4Params, LigandAtom } from "./types";
import { bar } from "./ui";
import { parsePDBQT, parseLigandPDBQT, loadAD4Params, atomsToArrayReal } from "./parsers";
import { buildPoses, applyTorsions, generateConformers, computeCenter, toLocalOrigin, randomSmallRotation, multiplyMatrices3 } from "./geometry";
import { scoreAllPosesGPU, bestOf, evaluatePose } from "./scoring";

declare function OpenBabelModule(): any;

let obMod: any = null;
let obInit: Promise<void> | null = null;

async function getOB(): Promise<any> {
  if (obMod) return obMod;
  if (!obInit) {
    console.log("[OB] calling OpenBabelModule()...");
    const watchdog = setInterval(() => {
      console.warn("[OB] still waiting on OpenBabelModule() init...");
    }, 3000);
    obInit = new Promise<void>((resolve, reject) => {
      try {
        const m = OpenBabelModule();
        // m.then() is not guaranteed to return a spec-compliant chainable
        // Promise (this build's shim doesn't support .catch on it), so
        // normalize via Promise.resolve() instead of chaining directly.
        Promise.resolve(m).then(
          () => {
            clearInterval(watchdog);
            console.log("[OB] OpenBabelModule() init resolved");
            // Re-enable error printing after init
            if (typeof Module !== 'undefined') {
              Module.printErr = function(text: string) { console.error(text); };
              Module.err = function(text: string) { console.error(text); };
            }
            obMod = m;
            resolve();
          },
          (e: any) => {
            clearInterval(watchdog);
            console.error("[OB] OpenBabelModule() init rejected:", e);
            reject(e);
          },
        );
      } catch (e) {
        clearInterval(watchdog);
        console.error("[OB] OpenBabelModule() threw synchronously:", e);
        reject(e);
      }
    });
  }
  await obInit;
  return obMod;
}

function extractProteinRecords(pdbText: string): string {
  const lines: string[] = [];
  for (const line of pdbText.split("\n")) {
    const rec = line.slice(0, 6).trim();
    if (rec === "ATOM" || rec === "TER" || rec === "END") lines.push(line);
  }
  return lines.join("\n");
}

function extractLigandRecords(pdbText: string, resName: string): string {
  return pdbText
    .split("\n")
    .filter(
      (line) =>
        line.startsWith("HETATM") && line.slice(17, 20).trim() === resName,
    )
    .join("\n");
}

async function convertPDBToPDBQT(
  pdbText: string,
  includeBranches: boolean,
  addHydrogens: boolean,
  gen3D: boolean,
  pH?: number,
): Promise<string> {
  console.log(`[OB] convertPDBToPDBQT start, ${pdbText.length} chars, addH=${addHydrogens} gen3D=${gen3D}`);
  try {
    console.log("[OB] awaiting getOB()...");
    const OB = await getOB();
    console.log("[OB] getOB() resolved");
    const conv = new OB.ObConversionWrapper();
    conv.setInFormat("", "pdb");
    const mol = new OB.OBMol();
    console.log("[OB] readString...");
    conv.readString(mol, pdbText);
    console.log("[OB] readString done");
    if (addHydrogens) {
      console.log("[OB] AddHydrogensWithParam...");
      mol.AddHydrogensWithParam(false, true, pH ?? 7.4);
      console.log("[OB] AddHydrogensWithParam done");
    }
    if (!includeBranches) {
      conv.addOption("r", OB.ObConversion_Option_type.OUTOPTIONS, "");
    }
    if (gen3D) {
      console.log("[OB] generate3DStructure...");
      const gen = new OB.OB3DGenWrapper();
      gen.generate3DStructure(mol, "MMFF94");
      console.log("[OB] generate3DStructure done");
    }
    conv.setOutFormat("", "pdbqt");
    console.log("[OB] writeString...");
    const outData = conv.writeString(mol, false);
    console.log(`[OB] writeString done, ${outData.length} chars`);
    mol.delete();
    conv.delete();
    return outData;
  } catch (err) {
    console.error("[OB] convertPDBToPDBQT threw:", err);
    throw err;
  }
}

function metropolisAccept(oldE: number, newE: number, temp: number): boolean {
  if (newE < oldE) return true;
  const prob = Math.exp((oldE - newE) / temp);
  return Math.random() < prob;
}

async function monteCarloSearch(
  baseAtoms: LigandAtom[],
  ad4Table: Record<string, AD4Params>,
  protein: Float32Array,
  numProtein: number,
  startCenter: { x: number; y: number; z: number },
  startRotation: number[],
  steps = 300,
  temp = 3.0,
): Promise<{ energy: number; center: { x: number; y: number; z: number }; rotation: number[] }> {
  // NOTE: this used to always test the identity rotation here, discarding
  // whatever orientation the coarse global search actually found. That made
  // this "refinement" step throw away the rotational answer and only ever
  // nudge translation around a (usually wrong) fixed orientation. It now
  // starts from, and jointly perturbs, the real best rotation + translation
  // together (simulated annealing over both), which is what actually lets
  // this step improve on the coarse grid search instead of just re-testing
  // a pose that was never a real candidate.
  let current = { center: { ...startCenter }, rotation: startRotation.slice(), energy: Infinity };
  let best = { center: { ...startCenter }, rotation: startRotation.slice(), energy: Infinity };
  const baseArr = atomsToArrayReal(baseAtoms, ad4Table);
  const numLigand = baseAtoms.length;

  for (let step = 0; step < steps; step++) {
    const scale = 2.0 * (1 - step / steps);
    const angleScale = (Math.PI / 6) * (1 - step / steps); // up to 30° early, 0° at the end

    const candidateCenter = {
      x: current.center.x + (Math.random() - 0.5) * scale,
      y: current.center.y + (Math.random() - 0.5) * scale,
      z: current.center.z + (Math.random() - 0.5) * scale,
    };
    const candidateRotation = multiplyMatrices3(randomSmallRotation(angleScale), current.rotation);

    const testPoses = new Float32Array([
      ...candidateRotation,
      candidateCenter.x, candidateCenter.y, candidateCenter.z,
    ]);
    const { energies } = await scoreAllPosesGPU(protein, numProtein, baseArr, numLigand, testPoses, 1);
    const candE = energies[0];

    if (step === 0 || metropolisAccept(current.energy, candE, temp)) {
      current = { center: candidateCenter, rotation: candidateRotation, energy: candE };
      if (candE < best.energy) best = { center: candidateCenter, rotation: candidateRotation, energy: candE };
    }
  }
  return best;
}

export async function runDemo({
  proteinPDBQT = "data/protein.pdbqt",
  ligandPDBQT = "data/ligand.pdbqt",
  proteinPDB,
  ligandResName,
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
      let pt: string, lt: string;

      if (proteinPDB && ligandResName && typeof OpenBabelModule !== "undefined") {
        const pdbText = await fetch(proteinPDB).then((r) => r.text());
        pg(30);
        const protPdbText = extractProteinRecords(pdbText);
        const ligPdbText = extractLigandRecords(pdbText, ligandResName);
        pg(40);
        pt = await convertPDBToPDBQT(protPdbText, false, false, false);
        pg(60);
        lt = await convertPDBToPDBQT(ligPdbText, true, true, false, 7.4);
      } else {
        [pt, lt] = await Promise.all([
          fetch(proteinPDBQT).then((r) => r.text()),
          fetch(ligandPDBQT).then((r) => r.text()),
        ]);
      }
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

  // Track every conformer's best coarse pose, not just the single overall
  // winner. The coarse grid score is noisy (simplified LJ+Coulomb, 1Å/
  // random-rotation resolution), so the single top scorer isn't always the
  // best *refinable* basin — a nearby runner-up can turn out to sit in a
  // deeper, more correct energy well once locally optimized. Refining
  // several candidates and keeping the best after refinement is what real
  // docking tools do (multiple seeds / clustering) instead of committing to
  // one coarse winner. This costs nothing extra during the coarse pass
  // (bestOf(energies) was already being computed per conformer) — the added
  // cost is only in re-running Monte Carlo refinement per candidate below.
  const perConformerBest: { energy: number; bestIdx: number; cIdx: number }[] = [];

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
    const ligandArr = atomsToArrayReal(bentAtoms, ad4Table);

    const { energies, gpuMs } = await scoreAllPosesGPU(
      proteinAtoms,
      numProtein,
      ligandArr,
      numLigand,
      poses,
      numPoses,
    );
    const { best, bestIdx } = bestOf(energies);
    perConformerBest.push({ energy: best, bestIdx, cIdx });
  }

  perConformerBest.sort((a, b) => a.energy - b.energy);
  const NUM_CANDIDATES = 3;
  const topCandidates = perConformerBest.slice(0, NUM_CANDIDATES);

  let globalBest = topCandidates.length ? topCandidates[0].energy : Infinity;
  let globalBestIdx = topCandidates.length ? topCandidates[0].bestIdx : -1;
  let globalBestConf = topCandidates.length ? topCandidates[0].cIdx : -1;

  // === Monte Carlo refinement: run on each of the top candidates, keep the best ===
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

      bar.setStatus(`[${systemName}] Monte Carlo (candidate ${k + 1}/${topCandidates.length})...`);
      const mcResult = await monteCarloSearch(
        bentAtoms, ad4Table, proteinAtoms, numProtein, startCenter, startRotation, 300,
      );

      // Compare against this candidate's own coarse energy, not just track
      // whether MC beat the eventual global winner — each candidate should
      // get credit for its own refinement.
      const candidateFinalEnergy = Math.min(mcResult.energy, cand.energy);
      if (candidateFinalEnergy < refinedBest) {
        refinedBest = candidateFinalEnergy;
        refinedBentAtoms = bentAtoms;
        refinedConf = cand.cIdx;
        refinedSourceIdx = cand.bestIdx;
        if (mcResult.energy < cand.energy) {
          refinedPoseIdx = 0;
          refinedPoseArr = new Float32Array([
            ...mcResult.rotation,
            mcResult.center.x, mcResult.center.y, mcResult.center.z,
          ]);
        } else {
          refinedPoseIdx = cand.bestIdx;
          refinedPoseArr = poses;
        }
      }
    }

    if (refinedBest < coarseBest) {
      console.log(`Monte Carlo improved: ${coarseBest.toFixed(3)} → ${refinedBest.toFixed(3)} kcal/mol (best of ${topCandidates.length} refined candidates)`);
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

  console.log(`Total run time: ${totalSec}s`);
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
      // pdb/resName removed: they triggered the live OpenBabel WASM
      // PDB->PDBQT conversion path, which hangs during module init in the
      // browser (never fires its ready callback — a bug in the bundled
      // openbabel.js/.wasm, not app code). A correctly pre-converted
      // protein.pdbqt/ligand.pdbqt already exists on disk for 1IEP, same as
      // the other 4 systems, so just fetch those instead.
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
      name: "3PTB (benzamidine)",
      prot: "systems/3ptb/protein.pdbqt",
      lig: "systems/3ptb/ligand.pdbqt",
      center: [-1.759, 14.461, 16.916],
    },
    {
      name: "1AC8 (TMZ)",
      prot: "systems/1ac8/protein.pdbqt",
      lig: "systems/1ac8/ligand.pdbqt",
      center: [31.924, 93.444, 47.924],
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
      proteinPDB: sys.pdb,
      ligandResName: sys.resName,
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
