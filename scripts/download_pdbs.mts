import { writeFileSync } from 'fs';
import { fileURLToPath } from 'url';
import path from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, '..', '..');
const systemsDir = path.join(projectRoot, 'data', 'systems');

async function get(url: string): Promise<string> {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`${resp.status} ${url}`);
  return resp.text();
}

const pdbs: Record<string, string> = {
  '1iep': 'https://files.rcsb.org/download/1IEP.pdb',
  '1hsg': 'https://files.rcsb.org/download/1HSG.pdb',
  '1stp': 'https://files.rcsb.org/download/1STP.pdb',
  '3ptb': 'https://files.rcsb.org/download/3PTB.pdb',
  '1ac8': 'https://files.rcsb.org/download/1AC8.pdb',
};

for (const [name, url] of Object.entries(pdbs)) {
  console.log(`Downloading ${name}...`);
  const text = await get(url);
  writeFileSync(path.join(systemsDir, name, `${name}.pdb`), text);
  console.log(`  ${(text.length / 1024).toFixed(0)} KB`);
}

function extractLigandCenters(pdbText: string, resName: string): { center: number[]; numAtoms: number; resName: string } | null {
  const atoms: { x: number; y: number; z: number }[] = [];
  for (const line of pdbText.split('\n')) {
    if (!line.startsWith('HETATM')) continue;
    const rName = line.slice(17, 20).trim();
    if (rName !== resName) continue;
    const x = parseFloat(line.slice(30, 38));
    const y = parseFloat(line.slice(38, 46));
    const z = parseFloat(line.slice(46, 54));
    atoms.push({ x, y, z });
  }
  if (atoms.length === 0) return null;
  const cx = atoms.reduce((s, a) => s + a.x, 0) / atoms.length;
  const cy = atoms.reduce((s, a) => s + a.y, 0) / atoms.length;
  const cz = atoms.reduce((s, a) => s + a.z, 0) / atoms.length;
  return { center: [cx, cy, cz], numAtoms: atoms.length, resName };
}

const resNames: Record<string, string> = {
  '1iep': 'STI',
  '1hsg': 'MK1',
  '1stp': 'BTN',
  '3ptb': 'BEN',
  '1ac8': 'TMZ',
};

for (const [name] of Object.entries(pdbs)) {
  const pdbText = await get(pdbs[name]);
  const info = extractLigandCenters(pdbText, resNames[name]);
  if (info) {
    console.log(`\n${name} (${info.resName}): center = (${info.center[0].toFixed(3)}, ${info.center[1].toFixed(3)}, ${info.center[2].toFixed(3)}), ${info.numAtoms} atoms`);
  } else {
    console.log(`\n${name}: NO LIGAND FOUND for "${resNames[name]}"`);
  }
}
