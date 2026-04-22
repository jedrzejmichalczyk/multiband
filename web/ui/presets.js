// Presets match the in-page form structure: each entry is a filter order
// plus per-band targets in dB.
window.ZOLO_PRESETS = {
  ex1_uniform: {
    nF: 9, nP: 3,
    passbands: [
      { a: -1.0, b: -0.625, rl_db: 20 },
      { a:  0.25, b: 1.0,   rl_db: 20 },
    ],
    stopbands: [
      { a: -10.0, b: -1.188, rej_db: 0 },
      { a:  -0.5, b:  0.125, rej_db: 0 },
      { a:  1.212, b: 10.0,  rej_db: 0 },
    ],
  },

  ex1_paper: {
    nF: 9, nP: 3,
    passbands: [
      { a: -1.0, b: -0.625, rl_db: 20 },
      { a:  0.25, b: 1.0,   rl_db: 20 },
    ],
    stopbands: [
      { a: -10.0, b: -1.188, rej_db: 15 },
      { a:  -0.5, b:  0.125, rej_db: 30 },
      { a:  1.212, b: 10.0,  rej_db: 15 },
    ],
  },

  ex2_paper: {
    nF: 7, nP: 3,
    passbands: [
      { a: -1.0,   b: -0.383, rl_db: 23 },
      { a:  0.383, b:  1.0,   rl_db: 23 },
    ],
    stopbands: [
      { a: -10.0,   b: -1.987, rej_db: 10 },
      { a:  -1.987, b: -1.864, rej_db: 15 },
      { a:  -0.037, b: -0.012, rej_db: 20 },
      { a:   1.185, b: 10.0,   rej_db: 40 },
    ],
  },

  symmetric_dual: {
    nF: 8, nP: 4,
    passbands: [
      { a: -1.0, b: -0.5, rl_db: 20 },
      { a:  0.5, b:  1.0, rl_db: 20 },
    ],
    stopbands: [
      { a: -3.0, b: -1.2, rej_db: 0 },
      { a: -0.3, b:  0.3, rej_db: 0 },
      { a:  1.2, b:  3.0, rej_db: 0 },
    ],
  },
};
