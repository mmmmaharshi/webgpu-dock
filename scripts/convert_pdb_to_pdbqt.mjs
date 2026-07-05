// PDB → PDBQT converter (approximate — no Gasteiger charges, but AD4 types assigned)
import { readFileSync, writeFileSync } from 'fs';
import { dirname, resolve } from 'path';
import { fileURLToPath } from 'url';

// Element → AD4 atom type mapping for protein residues
const ELEM_TO_AD4 = {
  'C': 'A',   // aromatic carbon (common default in AD4 for protein C)
  'N': 'N',
  'O': 'OA',
  'S': 'SA',
  'H': 'HD',
};

// Approximate charges by AD4 type (will deviate from true Gasteiger)
const AD4_CHARGE = {
  'A':  0.000,
  'C':  0.050,
  'N': -0.350,
  'NA': -0.350,
  'OA': -0.250,
  'SA': -0.100,
  'HD':  0.150,
};

function getElement(pdbLine) {
  const elem = pdbLine.slice(76, 78).trim() || pdbLine.slice(12, 16).trim()[0] || '';
  return elem.toUpperCase();
}

// For ligands: assign more specific AD4 types
function getLigandAD4Type(element, atomName, resName) {
  // AD4 type rules from AutoDock
  const name = atomName.trim().toUpperCase();
  switch (element) {
    case 'C':
      // Check if aromatic-like (name starts with C in aromatic rings)
      return 'A';  // default aromatic
    case 'N':
      if (name.startsWith('N') && name.length >= 2 && !isNaN(name[1]))
        return 'N';  // aliphatic N
      return 'NA';  // aromatic N
    case 'O':
      if (name.startsWith('O') && name.length >= 2 && name[1] === 'H')
        return 'OA';
      // Check carbonyl
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

function convertProteinPDBQT(pdbText, outPath) {
  const lines = [];
  for (const line of pdbText.split('\n')) {
    if (!line.startsWith('ATOM')) continue;
    
    const x = line.slice(30, 38).trim();
    const y = line.slice(38, 46).trim();
    const z = line.slice(46, 54).trim();
    const elem = getElement(line);
    const ad4Type = ELEM_TO_AD4[elem] || 'C';
    const charge = AD4_CHARGE[ad4Type] || 0;
    
    // PDBQT format: keep original columns but replace charge and type
    const newLine = line.slice(0, 70) +
      charge.toFixed(3).padStart(6) +
      ad4Type.padStart(2);
    
    lines.push(newLine);
  }
  writeFileSync(outPath, lines.join('\n') + '\n');
  console.log(`  Protein PDBQT: ${lines.length} atoms -> ${outPath}`);
}

function convertLigandPDBQT(pdbText, outPath, resName) {
  const lines = ['ROOT'];
  let atomCount = 0;
  
  for (const line of pdbText.split('\n')) {
    if (!line.startsWith('HETATM')) continue;
    const rName = line.slice(17, 20).trim();
    if (resName && rName !== resName) continue;
    
    const atomName = line.slice(12, 16).trim();
    const x = line.slice(30, 38).trim();
    const y = line.slice(38, 46).trim();
    const z = line.slice(46, 54).trim();
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

// --- MAIN ---
const systemDir = 'systems/3ptb';
const pdbPath = `${systemDir}/3ptb.pdb`;
const pdbText = readFileSync(pdbPath, 'utf-8');

console.log('Converting 3PTB (Trypsin + Benzamidine)...');
convertProteinPDBQT(pdbText, `${systemDir}/protein.pdbqt`);
convertLigandPDBQT(pdbText, `${systemDir}/ligand.pdbqt`, 'BEN');

// Also verify the existing PDBQT files have valid formats
console.log('\nDone.');
