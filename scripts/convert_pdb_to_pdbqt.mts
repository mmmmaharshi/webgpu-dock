import { readFileSync, writeFileSync } from 'fs';
import { fileURLToPath } from 'url';
import path from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, '..', '..');

const ELEM_TO_AD4: Record<string, string> = {
  'C': 'A',
  'N': 'N',
  'O': 'OA',
  'S': 'SA',
  'H': 'HD',
};

const AD4_CHARGE: Record<string, number> = {
  'A':  0.000,
  'C':  0.050,
  'N': -0.350,
  'NA': -0.350,
  'OA': -0.250,
  'SA': -0.100,
  'HD':  0.150,
};

function getElement(pdbLine: string): string {
  const elem = pdbLine.slice(76, 78).trim() || pdbLine.slice(12, 16).trim()[0] || '';
  return elem.toUpperCase();
}

function getLigandAD4Type(element: string, atomName: string, _resName: string): string {
  const name = atomName.trim().toUpperCase();
  switch (element) {
    case 'C':
      return 'A';
    case 'N':
      if (name.startsWith('N') && name.length >= 2 && !isNaN(Number(name[1])))
        return 'N';
      return 'NA';
    case 'O':
      return 'OA';
    case 'S':
      return 'SA';
    case 'H':
      return 'HD';
    case 'F':
    case 'CL':
      return 'C';
    default:
      return 'C';
  }
}

function convertProteinPDBQT(pdbText: string, outPath: string): void {
  const lines: string[] = [];
  for (const line of pdbText.split('\n')) {
    if (!line.startsWith('ATOM')) continue;

    const elem = getElement(line);
    const ad4Type = ELEM_TO_AD4[elem] || 'C';
    const charge = AD4_CHARGE[ad4Type] || 0;

    const newLine = line.slice(0, 70) +
      charge.toFixed(3).padStart(6) +
      ad4Type.padStart(2);

    lines.push(newLine);
  }
  writeFileSync(outPath, lines.join('\n') + '\n');
  console.log(`  Protein PDBQT: ${lines.length} atoms -> ${outPath}`);
}

function convertLigandPDBQT(pdbText: string, outPath: string, resName: string): { numAtoms: number } {
  const lines = ['ROOT'];
  let atomCount = 0;

  for (const line of pdbText.split('\n')) {
    if (!line.startsWith('HETATM')) continue;
    const rName = line.slice(17, 20).trim();
    if (resName && rName !== resName) continue;

    const atomName = line.slice(12, 16).trim();
    const elem = getElement(line);
    const ad4Type = getLigandAD4Type(elem, atomName, rName);
    const charge = AD4_CHARGE[ad4Type] || 0;

    const newLine = line.slice(0, 70) +
      charge.toFixed(3).padStart(6) +
      ad4Type.padStart(2);

    lines.push(newLine);
    atomCount++;
  }

  lines.push('ENDROOT');
  lines.push('TORSDOF 0');

  writeFileSync(outPath, lines.join('\n') + '\n');
  console.log(`  Ligand PDBQT: ${atomCount} atoms -> ${outPath}`);

  return { numAtoms: atomCount };
}

const systemDir = path.join(projectRoot, 'data', 'systems', '3ptb');
const pdbPath = path.join(systemDir, '3ptb.pdb');
const pdbText = readFileSync(pdbPath, 'utf-8');

console.log('Converting 3PTB (Trypsin + Benzamidine)...');
convertProteinPDBQT(pdbText, path.join(systemDir, 'protein.pdbqt'));
convertLigandPDBQT(pdbText, path.join(systemDir, 'ligand.pdbqt'), 'BEN');

console.log('\nDone.');
