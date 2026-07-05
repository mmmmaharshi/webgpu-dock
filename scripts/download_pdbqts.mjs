import { writeFileSync, copyFileSync } from 'fs';

async function get(url) {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`${resp.status} ${url}`);
  return resp.text();
}

// 1IEP — copy existing files
copyFileSync('protein.pdbqt', 'systems/1iep/protein.pdbqt');
copyFileSync('ligand.pdbqt', 'systems/1iep/ligand.pdbqt');
console.log('1IEP: copied');

// 1HSG
const hsgProt = await get('https://bioboot.github.io/bggn213_W19/class-material/1hsg_protein.pdbqt');
writeFileSync('systems/1hsg/protein.pdbqt', hsgProt);
const hsgLig = await get('https://bioboot.github.io/bggn213_W19/class-material/ligand.pdbqt');
writeFileSync('systems/1hsg/ligand.pdbqt', hsgLig);
console.log('1HSG: done');

// 1STP
const stpProt = await get('https://raw.githubusercontent.com/ccsb-scripps/AutoDock-GPU/develop/input/1stp/derived/1stp_protein.pdbqt');
writeFileSync('systems/1stp/protein.pdbqt', stpProt);
const stpLig = await get('https://raw.githubusercontent.com/ccsb-scripps/AutoDock-GPU/develop/input/1stp/derived/1stp_ligand.pdbqt');
writeFileSync('systems/1stp/ligand.pdbqt', stpLig);
console.log('1STP: done');

// 1AC8
const ac8Prot = await get('https://raw.githubusercontent.com/ccsb-scripps/AutoDock-GPU/develop/input/1ac8/derived/rec.pdbqt');
writeFileSync('systems/1ac8/protein.pdbqt', ac8Prot);
const ac8Lig = await get('https://raw.githubusercontent.com/ccsb-scripps/AutoDock-GPU/develop/input/1ac8/derived/1ac8_ligand.pdbqt');
writeFileSync('systems/1ac8/ligand.pdbqt', ac8Lig);
console.log('1AC8: done');

console.log('All PDBQT files downloaded.');
