# Training reader fixtures

These are the first 32 frames of the first train and validation views from the
genuine six-rally simulator dataset generated at source revision
07376179ae56432a574073e70fb3b8586e7f47b9. The physical train/validation identities
remain disjoint. No test-split labels are included. Source split SHA-256 values:

- train: e5f9ffe1b337e324754cfc15feaf2ea77dd5b967b67eabc9754f18233eb1f04e
- val: 135a859f32aa535bcda54434da2f1607b420c1dfa7425142672e5635b61de75c

Each numeric view was cropped without changing retained values. Event tables
retain events within the crop; segment and hit tables retain physical-rally
identity. The shared NPZ writer records the original simulation provenance and
the fixture manifests contain the new view checksum/frame count. These fragments
exercise the simulator's view reader and lifter training/evaluation; they omit physical rally
and truth JSON files, so they are not a complete dataset for audit_splits.
The unchanged resolved simulation config controls independent training draws.
