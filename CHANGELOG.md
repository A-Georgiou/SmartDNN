# Changelog

## Unreleased

### Added

- AdaGrad optimizer and options with cumulative squared-gradient state, positive
  learning-rate overrides and count validation, plus focused unit tests
  ([#98](https://github.com/A-Georgiou/SmartDNN/pull/98)).
- Unit-test CI for main pushes, pull requests and manual dispatch, plus scheduled
  Copilot task-branch cleanup with a manual dry-run option
  ([#91](https://github.com/A-Georgiou/SmartDNN/pull/91)).
- Swish and Mish activations, SGD and RMSProp optimizers, and associated tests
  ([#50](https://github.com/A-Georgiou/SmartDNN/pull/50)).

### Documentation

- Audited README features, optimizer usage, build order, test commands, example
  links and current limitations against `bce6ecb` on 2026-10-29.
- Marked existing performance figures as historical and unverified for the
  current head; no benchmark numbers were changed.

### Historical merge coverage

This is the first documentation audit log. Earlier merged backend work is listed
for traceability, not as a claim that those backends exist at the audited head:

- Eigen backend work: [#52](https://github.com/A-Georgiou/SmartDNN/pull/52).
- ArrayFire backend work: [#53](https://github.com/A-Georgiou/SmartDNN/pull/53).
- Example validation: [#54](https://github.com/A-Georgiou/SmartDNN/pull/54).
- ArrayFire compilation fixes: [#56](https://github.com/A-Georgiou/SmartDNN/pull/56).
- GPU template fixes: [#57](https://github.com/A-Georgiou/SmartDNN/pull/57).
- CPU backend type fixes: [#58](https://github.com/A-Georgiou/SmartDNN/pull/58).

The current tree at `bce6ecb` has CPU tensor storage only; it does not contain
the historical Eigen or ArrayFire backends.
