// Paper examples that match what the Python reference produces.
window.ZOLO_PRESETS = {
  example1: {
    nF: 9, nP: 3,
    passbands: "-1, -0.625\n0.25, 1.0",
    stopbands: "-10, -1.188\n-0.5, 0.125\n1.212, 10",
    psi_I_db: 20,
    psi_J_default_db: 15,
    psi_J_pieces: "-0.5, 0.125, 30",
  },
  example2: {
    nF: 7, nP: 3,
    passbands: "-1, -0.383\n0.383, 1",
    stopbands: "-10, -1.987\n-1.987, -1.864\n-0.037, -0.012\n1.185, 10",
    psi_I_db: 23,
    psi_J_default_db: 40,
    psi_J_pieces:
      "-10, -1.987, 10\n-1.987, -1.864, 15\n-0.037, -0.012, 20",
  },
};
