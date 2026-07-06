import { existsSync, mkdirSync, writeFileSync } from "fs";
import path from "path";

const DATA_DIR = path.resolve(import.meta.dir, "..", "data");

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
    name: "3ce3",
    protUrl:
      "https://raw.githubusercontent.com/ccsb-scripps/AutoDock-GPU/develop/input/3ce3/derived/3ce3_protein.pdbqt",
    ligUrl:
      "https://raw.githubusercontent.com/ccsb-scripps/AutoDock-GPU/develop/input/3ce3/derived/3ce3_ligand.pdbqt",
  },
  {
    name: "3tmn",
    protUrl:
      "https://raw.githubusercontent.com/ccsb-scripps/AutoDock-GPU/develop/input/3tmn/derived/rec.pdbqt",
    ligUrl:
      "https://raw.githubusercontent.com/ccsb-scripps/AutoDock-GPU/develop/input/3tmn/derived/3tmn_ligand.pdbqt",
  },
  {
    name: "7cpa",
    protUrl:
      "https://raw.githubusercontent.com/ccsb-scripps/AutoDock-GPU/develop/input/7cpa/derived/rec.pdbqt",
    ligUrl:
      "https://raw.githubusercontent.com/ccsb-scripps/AutoDock-GPU/develop/input/7cpa/derived/7cpa_ligand.pdbqt",
  },
];

export async function ensureData(): Promise<void> {
  console.log("Checking data files...");

  for (const sys of SYSTEMS) {
    const sysDir = path.join(DATA_DIR, "systems", sys.name);
    ensureDir(sysDir);

    if (sys.protUrl) {
      await ensureFile(path.join(sysDir, "protein.pdbqt"), sys.protUrl);
    }
    if (sys.ligUrl) {
      await ensureFile(path.join(sysDir, "ligand.pdbqt"), sys.ligUrl);
    }
  }

  const rootProt = path.join(DATA_DIR, "protein.pdbqt");
  const rootLig = path.join(DATA_DIR, "ligand.pdbqt");

  if (!existsSync(rootProt)) {
    console.log("  Downloading root protein.pdbqt from 1iep...");
    writeFileSync(
      rootProt,
      await fetchText(
        "https://raw.githubusercontent.com/ccsb-scripps/AutoDock-Vina/develop/example/python_scripting/1iep_receptor.pdbqt",
      ),
    );
  }
  if (!existsSync(rootLig)) {
    console.log("  Downloading root ligand.pdbqt from 1iep...");
    writeFileSync(
      rootLig,
      await fetchText(
        "https://raw.githubusercontent.com/ccsb-scripps/AutoDock-Vina/develop/example/python_scripting/1iep_ligand.pdbqt",
      ),
    );
  }

  console.log("Data check complete.");
}
