# Augmentation and training budgets

The supplied paper's page 8/Table 2 was rendered and visually verified: 150 epochs,
batch size 16, Adam, learning rate 0.001, Dice loss; rotation up to 2.5 degrees,
width/height shifts up to 0.75%, noise up to 0.75%, zoom up to 0.75%, horizontal flips.
The paper does not specify the full probability law, noise distribution or interpolation.

Our explicit implementation choices (not claimed exact reproduction):

- Every training image receives a combined affine transform: angle uniform [-2.5,2.5]
  degrees, shifts independently uniform [-0.0075,0.0075] times image dimensions,
  zoom uniform [0.9925,1.0075], horizontal flip probability 0.5, around pixel-center coordinates.
- Image interpolation is bilinear (SciPy order 1); masks use nearest-neighbor (order 0).
  Both use the identical affine matrix and zero border fill. Output masks stay binary.
- Image-only Gaussian noise, standard deviation uniform [0,0.0075] on normalized [0,1]
  intensities, then clipping to [0,1]. This interprets an unspecified paper noise range
  as a standard-deviation bound; it is documented rather than assumed equivalent.
- SeedSequence([fold seed, shuffled position]) gives reproducible paired transforms.
  Position/image associations change under seeded epoch shuffling. Parallel mapping does
  not depend on shared mutable NumPy random state. No selection/test augmentation occurs.

Ranges/probabilities can be overridden through `augmentation_parameters` in JSON config;
`augmentation` toggles the transform. Epochs, batch size, optimizer, learning rate,
loss, evaluation threshold, fold count and early-stopping patience are configurable.
Checkpoint selection stays prespecified as inner hard Dice, maximum with earliest ties.

The intended paper-budget config is 150 epochs/batch 16; an explicit memory-budget
configuration uses smaller batches where necessary. Short 10-epoch experiments remain
preliminary. Any deviation is saved in each run's configuration. Smoke checks on real
512x512 data precede a representative fold; actual timings determine the affordable
budget. No expectation of improvement substitutes for measured curves.

Five augmentation tests verify exact identity/flip, deterministic sampling, alignment
under geometry, binary masks, image-only noise, and no validation transforms.

Measured CPU functionality checks passed for all four models at 512x512. A full
Modified U-Net training/inner-selection epoch took 363.70 seconds with batch size 2.
Its checkpoint was saved without evaluating the outer test set. See reports/resources.
