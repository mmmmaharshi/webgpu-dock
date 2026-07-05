import type { BenchResult, DemoOptions, LigandAtom } from "./types";
import { bar } from "./ui";
import { parsePDBQT, parseLigandPDBQT, loadAD4Params, atomsToArrayReal } from "./parsers";
import { buildPoses, applyTorsions, generateConformers, computeCenter, recenterLigand } from "./geometry";
import { scoreAllPosesGPU, bestOf } from "./scoring";

export async function runDemo({
  proteinPDBQT = "data/protein.pdbqt",
  ligandPDBQT = "data/ligand.pdbqt",
  numRotations = 12,
  translationRange = 5,
  translationStep = 1,
  numConformers = 40,
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
    "Fetching PDBQT files...",
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
  const ligandAtomsBase = recenterLigand(rawLigandAtoms, pocketCenter);
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
    const result = buildPoses(numRotations, translationRange, translationStep);
    pg(100);
    return result;
  });
  bar.setStatus(
    `Pose set: ${numPoses} poses (${rCount} rotations × ${numTranslations} translations)`,
  );
  console.log(
    `Pose set: ${rCount} rotations × ${numTranslations} translations = ${numPoses} poses`,
  );

  if (!("gpu" in navigator)) {
    console.log("WebGPU not available — cannot run torsion search efficiently");
    bar.setStatus("WebGPU required");
    bar.done();
    return;
  }

  const conformers = generateConformers(numConformers, branches.length);

  let globalBest = Infinity;
  let globalBestIdx = -1;
  let globalBestConf = -1;

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

    if (best < globalBest) {
      globalBest = best;
      globalBestIdx = bestIdx;
      globalBestConf = cIdx;
    }
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
  const pb = globalBestIdx * 12;
  const rx =
    poses[pb] * lcx + poses[pb + 1] * lcy + poses[pb + 2] * lcz + poses[pb + 9];
  const ry =
    poses[pb + 3] * lcx +
    poses[pb + 4] * lcy +
    poses[pb + 5] * lcz +
    poses[pb + 10];
  const rz =
    poses[pb + 6] * lcx +
    poses[pb + 7] * lcy +
    poses[pb + 8] * lcz +
    poses[pb + 11];
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
      systemName: sys.name,
      knownCenter: sys.center,
      numConformers: 40,
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
