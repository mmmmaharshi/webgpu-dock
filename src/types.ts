export interface Atom {
  x: number;
  y: number;
  z: number;
  charge: number;
  atomType: string;
}

export interface LigandAtom {
  serial: number;
  x: number;
  y: number;
  z: number;
  charge: number;
  atomType: string;
}

export interface Branch {
  parentSerial: number;
  childSerial: number;
  indices: Set<number>;
  openOrder: number;
}

export interface LigandResult {
  atoms: LigandAtom[];
  serialToIndex: Record<number, number>;
  branches: Branch[];
}

export interface AD4Params {
  Rii: number;
  epsii: number;
}

export interface PoseSet {
  poses: Float32Array;
  numPoses: number;
  numRotations: number;
  numTranslations: number;
}

export interface BenchResult {
  systemName: string;
  bestScore: number;
  bestIdx: number;
  bestConf: number;
  dist: number;
  bestCenter: number[];
  knownCenter: number[];
  totalTime: number;
}

export interface DemoOptions {
  proteinPDBQT?: string;
  ligandPDBQT?: string;
  numRotations?: number;
  translationRange?: number;
  translationStep?: number;
  numConformers?: number;
  systemName?: string;
  knownCenter?: number[];
}

export interface ProgressBar {
  setStatus(msg: string): void;
  setProgress(pct: number): void;
  done(): void;
  run<T>(
    initialStatus: string,
    fn: (pg: (pct: number) => void, st: (msg: string) => void) => Promise<T>,
  ): Promise<T>;
}
