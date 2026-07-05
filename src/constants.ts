export const COULOMB_K = 332.0636;

export const RII_TO_SIGMA = 1 / Math.pow(2, 1 / 6);

// Mehler-Solmajer (1991) sigmoidal distance-dependent dielectric, the same
// model AutoDock4 uses by default for its electrostatics term. Without this,
// Coulomb's energy is computed with a fixed dielectric of 1 (bare 1/r),
// which blows up unrealistically at the short contact distances that occur
// between charged atoms in a real bound pose (we saw this directly: 3PTB's
// score was -4155 kcal/mol, ~15-60x more negative than every other system,
// once its ligand charges were corrected to their real protonation state).
// This model screens electrostatics much more aggressively at short range
// (approximating solvent/protein dielectric screening), so energy
// magnitudes become comparable across systems instead of dominated by
// whichever one happens to have the closest same-sign or opposite-sign
// charge contact.
export const DIELECTRIC_A = -8.5525;
export const DIELECTRIC_B = 86.9525;
export const DIELECTRIC_LAMBDA = 0.003627;
export const DIELECTRIC_K = 7.7839;
