import { readFileSync } from 'fs';

function extractLigandByChain(pdbText, resName) {
  const atomsByChain = {};
  for (const line of pdbText.split('\n')) {
    if (!line.startsWith('HETATM')) continue;
    const rName = line.slice(17, 20).trim();
    if (rName !== resName) continue;
    const chain = line.slice(21, 22).trim() || '?';
    const x = parseFloat(line.slice(30, 38));
    const y = parseFloat(line.slice(38, 46));
    const z = parseFloat(line.slice(46, 54));
    if (!atomsByChain[chain]) atomsByChain[chain] = [];
    atomsByChain[chain].push({ x, y, z });
  }
  return atomsByChain;
}

const systems = ['1iep', '1hsg', '1stp', '3ptb', '1ac8'];
const resNames = { '1iep': 'STI', '1hsg': 'MK1', '1stp': 'BTN', '3ptb': 'BEN', '1ac8': 'TMZ' };

for (const name of systems) {
  const pdbText = readFileSync(`systems/${name}/${name}.pdb`, 'utf-8');
  const byChain = extractLigandByChain(pdbText, resNames[name]);
  console.log(`\n${name} (${resNames[name]}):`);
  for (const [chain, atoms] of Object.entries(byChain)) {
    const cx = atoms.reduce((s, a) => s + a.x, 0) / atoms.length;
    const cy = atoms.reduce((s, a) => s + a.y, 0) / atoms.length;
    const cz = atoms.reduce((s, a) => s + a.z, 0) / atoms.length;
    console.log(`  Chain ${chain}: (${cx.toFixed(3)}, ${cy.toFixed(3)}, ${cz.toFixed(3)}), ${atoms.length} atoms`);
  }
}
