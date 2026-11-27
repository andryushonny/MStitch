# Feature search (planned, milestone 3)

The operator marks a small area on the quick map where at least one feature (crystal, flake) is promised.
MStitch finds it without ML first:

1. substrate model: robust colour statistics of the area;
2. contrast to the substrate in colour space, morphological clean-up;
3. connected components, size filter (dust and the litho labels out), ranking by area and contrast;
4. `plan_quality(feature, profile)` → stage targets for the quality scan (overlap 35–40 %, one-side approach,
   the offset between objectives from the profiles).

An ML detector (ONNX on CUDA) can be added later as a plugin if the classical one is not enough.
