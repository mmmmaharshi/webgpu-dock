import type { Atom, AD4Params, LigandAtom, Branch, LigandResult } from "./types";
import { RII_TO_SIGMA } from "./constants";

export function parsePDBQT(text: string): Atom[] {
  const atoms: Atom[] = [];
  for (const line of text.split("\n")) {
    const record = line.slice(0, 6).trim();
    if (record !== "ATOM" && record !== "HETATM") continue;

    const x = parseFloat(line.slice(30, 38));
    const y = parseFloat(line.slice(38, 46));
    const z = parseFloat(line.slice(46, 54));
    const charge = parseFloat(line.slice(70, 76));
    const atomType = line.slice(77, 79).trim();

    atoms.push({ x, y, z, charge, atomType });
  }
  return atoms;
}

export function parseLigandPDBQT(text: string): LigandResult {
  const atoms: LigandAtom[] = [];
  const serialToIndex: Record<number, number> = {};
  const branches: Branch[] = [];
  const stack: Branch[] = [];
  let openCounter = 0;

  for (const rawLine of text.split("\n")) {
    const trimmed = rawLine.trim();
    if (!trimmed) continue;
    const tag = trimmed.split(/\s+/)[0];

    if (tag === "ATOM" || tag === "HETATM") {
      const serial = parseInt(rawLine.slice(6, 11), 10);
      const x = parseFloat(rawLine.slice(30, 38));
      const y = parseFloat(rawLine.slice(38, 46));
      const z = parseFloat(rawLine.slice(46, 54));
      const charge = parseFloat(rawLine.slice(70, 76));
      const atomType = rawLine.slice(77, 79).trim();

      const idx = atoms.length;
      atoms.push({ serial, x, y, z, charge, atomType });
      serialToIndex[serial] = idx;

      for (const b of stack) b.indices.add(idx);
    } else if (tag === "BRANCH") {
      const parts = trimmed.split(/\s+/);
      const branch: Branch = {
        parentSerial: parseInt(parts[1], 10),
        childSerial: parseInt(parts[2], 10),
        indices: new Set(),
        openOrder: openCounter++,
      };
      stack.push(branch);
      branches.push(branch);
    } else if (tag === "ENDBRANCH") {
      stack.pop();
    }
  }

  branches.sort((a, b) => a.openOrder - b.openOrder);
  return { atoms, serialToIndex, branches };
}

export async function loadAD4Params(): Promise<Record<string, AD4Params>> {
  const url =
    "https://raw.githubusercontent.com/ccsb-scripps/AutoDock-Vina/develop/data/AD4_parameters.dat";
  const text = await fetch(url).then((r) => r.text());
  const table: Record<string, AD4Params> = {};
  for (const line of text.split("\n")) {
    if (!line.startsWith("atom_par")) continue;
    const parts = line.trim().split(/\s+/);
    table[parts[1]] = {
      Rii: parseFloat(parts[2]),
      epsii: parseFloat(parts[3]),
    };
  }
  return table;
}

export function atomsToArrayReal(
  atoms: Atom[],
  ad4Table: Record<string, AD4Params>,
): Float32Array {
  const arr = new Float32Array(atoms.length * 6);
  atoms.forEach((a, i) => {
    const params = ad4Table[a.atomType];
    if (!params) {
      console.warn(
        `No AD4 param for atom type "${a.atomType}", using generic carbon fallback`,
      );
    }
    const Rii = params ? params.Rii : 4.0;
    const epsii = params ? params.epsii : 0.15;

    const base = i * 6;
    arr[base] = a.x;
    arr[base + 1] = a.y;
    arr[base + 2] = a.z;
    arr[base + 3] = a.charge;
    arr[base + 4] = Rii * RII_TO_SIGMA;
    arr[base + 5] = epsii;
  });
  return arr;
}
