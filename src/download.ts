import { existsSync, mkdirSync, writeFileSync } from "fs";
import path from "path";

const DATA_DIR = path.resolve(import.meta.dir, "..", "data");

const ELEM_TO_AD4: Record<string, string> = {
  C: "A",
  N: "N",
  O: "OA",
  S: "SA",
  H: "HD",
};

const AD4_CHARGE: Record<string, number> = {
  A: 0.0,
  C: 0.05,
  N: -0.35,
  NA: -0.35,
  OA: -0.25,
  SA: -0.1,
  HD: 0.15,
};

function getElement(pdbLine: string): string {
  const elem = pdbLine.slice(76, 78).trim() ||
    (pdbLine.slice(12, 16).trim()[0] ?? "");
  return elem.toUpperCase();
}

function getLigandAD4Type(
  element: string,
  atomName: string,
): string {
  const name = atomName.trim().toUpperCase();
  switch (element) {
    case "C":
      return "A";
    case "N":
      if (name.startsWith("N") && name.length >= 2 && !isNaN(Number(name[1])))
        return "N";
      return "NA";
    case "O":
      return "OA";
    case "S":
      return "SA";
    case "H":
      return "HD";
    case "F":
    case "CL":
      return "C";
    default:
      return "C";
  }
}

function convertProteinPDBQT(pdbText: string): string {
  const lines: string[] = [];
  for (const line of pdbText.split("\n")) {
    if (!line.startsWith("ATOM")) continue;
    const elem = getElement(line);
    const ad4Type = ELEM_TO_AD4[elem] ?? "C";
    const charge = AD4_CHARGE[ad4Type] ?? 0;
    const newLine =
      line.slice(0, 70) +
      charge.toFixed(3).padStart(6) +
      ad4Type.padStart(2);
    lines.push(newLine);
  }
  return lines.join("\n") + "\n";
}

function convertLigandPDBQT(pdbText: string, resName: string): string {
  const lines = ["ROOT"];
  for (const line of pdbText.split("\n")) {
    if (!line.startsWith("HETATM")) continue;
    const rName = line.slice(17, 20).trim();
    if (rName !== resName) continue;
    const atomName = line.slice(12, 16).trim();
    const elem = getElement(line);
    const ad4Type = getLigandAD4Type(elem, atomName);
    const charge = AD4_CHARGE[ad4Type] ?? 0;
    const newLine =
      line.slice(0, 70) +
      charge.toFixed(3).padStart(6) +
      ad4Type.padStart(2);
    lines.push(newLine);
  }
  lines.push("ENDROOT");
  lines.push("TORSDOF 0");
  return lines.join("\n") + "\n";
}

async function fetchText(url: string): Promise<string> {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`${resp.status} ${url}`);
  return resp.text();
}

function ensureDir(dir: string): void {
  if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
}

async function ensureFile(filePath: string, url: string): Promise<boolean> {
  if (existsSync(filePath)) return false;
  console.log(`  Downloading ${path.basename(filePath)}...`);
  const text = await fetchText(url);
  writeFileSync(filePath, text);
  console.log(`    -> ${(text.length / 1024).toFixed(0)} KB`);
  return true;
}

interface SystemDef {
  name: string;
  protUrl?: string;
  ligUrl?: string;
  pdbUrl?: string;
  ligResName?: string;
}

const SYSTEMS: SystemDef[] = [
  {
    name: "1iep",
    protUrl:
      "https://raw.githubusercontent.com/ccsb-scripps/AutoDock-Vina/develop/example/python_scripting/1iep_receptor.pdbqt",
    ligUrl:
      "https://raw.githubusercontent.com/ccsb-scripps/AutoDock-Vina/develop/example/python_scripting/1iep_ligand.pdbqt",
  },
  {
    name: "1hsg",
    protUrl:
      "https://bioboot.github.io/bggn213_W19/class-material/1hsg_protein.pdbqt",
    ligUrl:
      "https://bioboot.github.io/bggn213_W19/class-material/ligand.pdbqt",
  },
  {
    name: "1stp",
    protUrl:
      "https://raw.githubusercontent.com/ccsb-scripps/AutoDock-GPU/develop/input/1stp/derived/1stp_protein.pdbqt",
    ligUrl:
      "https://raw.githubusercontent.com/ccsb-scripps/AutoDock-GPU/develop/input/1stp/derived/1stp_ligand.pdbqt",
  },
  {
    name: "1ac8",
    protUrl:
      "https://raw.githubusercontent.com/ccsb-scripps/AutoDock-GPU/develop/input/1ac8/derived/rec.pdbqt",
    ligUrl:
      "https://raw.githubusercontent.com/ccsb-scripps/AutoDock-GPU/develop/input/1ac8/derived/1ac8_ligand.pdbqt",
  },
  {
    name: "3ptb",
    pdbUrl: "https://files.rcsb.org/download/3PTB.pdb",
    ligResName: "BEN",
  },
];

export async function ensureData(): Promise<void> {
  console.log("Checking data files...");

  for (const sys of SYSTEMS) {
    const sysDir = path.join(DATA_DIR, "systems", sys.name);
    ensureDir(sysDir);

    if (sys.protUrl) {
      await ensureFile(
        path.join(sysDir, "protein.pdbqt"),
        sys.protUrl,
      );
    }
    if (sys.ligUrl) {
      await ensureFile(
        path.join(sysDir, "ligand.pdbqt"),
        sys.ligUrl,
      );
    }

    if (sys.pdbUrl && sys.ligResName) {
      const protPdbqtPath = path.join(sysDir, "protein.pdbqt");
      const ligPdbqtPath = path.join(sysDir, "ligand.pdbqt");

      if (!existsSync(protPdbqtPath) || !existsSync(ligPdbqtPath)) {
        console.log(`  Converting ${sys.name} from PDB...`);
        const pdbText = await fetchText(sys.pdbUrl);

        if (!existsSync(protPdbqtPath)) {
          const protPdbqt = convertProteinPDBQT(pdbText);
          writeFileSync(protPdbqtPath, protPdbqt);
          console.log(
            `    protein.pdbqt -> ${(protPdbqt.length / 1024).toFixed(0)} KB`,
          );
        }

        if (!existsSync(ligPdbqtPath)) {
          const ligPdbqt = convertLigandPDBQT(pdbText, sys.ligResName);
          writeFileSync(ligPdbqtPath, ligPdbqt);
          console.log(
            `    ligand.pdbqt -> ${(ligPdbqt.length / 1024).toFixed(0)} KB`,
          );
        }
      }
    }
  }

  // Also ensure root demo files (same as 1IEP)
  const rootProt = path.join(DATA_DIR, "protein.pdbqt");
  const rootLig = path.join(DATA_DIR, "ligand.pdbqt");
  const iepProt = path.join(DATA_DIR, "systems", "1iep", "protein.pdbqt");
  const iepLig = path.join(DATA_DIR, "systems", "1iep", "ligand.pdbqt");

  if (!existsSync(rootProt) && existsSync(iepProt)) {
    console.log("  Copying root protein.pdbqt from 1iep...");
    writeFileSync(rootProt, await fetchText(
      "https://raw.githubusercontent.com/ccsb-scripps/AutoDock-Vina/develop/example/python_scripting/1iep_receptor.pdbqt",
    ));
  }
  if (!existsSync(rootLig) && existsSync(iepLig)) {
    console.log("  Copying root ligand.pdbqt from 1iep...");
    writeFileSync(rootLig, await fetchText(
      "https://raw.githubusercontent.com/ccsb-scripps/AutoDock-Vina/develop/example/python_scripting/1iep_ligand.pdbqt",
    ));
  }

  console.log("Data check complete.");
}
