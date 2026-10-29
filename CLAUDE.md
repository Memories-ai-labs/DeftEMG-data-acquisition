# Working on DeftEMG-data-acquisition

This repository is public (MIT). It holds data acquisition only: the armband driver, recording
and sessions, the prompter and its tasks, the live viewer, replay, timing checks, filtering, and
the trial-loading data chain they share.

## Keep it acquisition-only

- No models, training or results, and nothing about projects built on this package: not their
  names, code, experiments, recordings or results, nor the people recorded. This covers code,
  comments, docs, tests, example data, commit messages, PR descriptions, and branch and tag
  names.
- When something built on this package needs more from it, add a general extension point,
  named and documented in acquisition terms and tested here with a stand-in, as `TaskPlugin`,
  `PrompterOverlay`, `ViewerPanel`, `ReplayPanel` and `PrompterExtension` are (README, "7.
  Extending the tools"). Say what it does, not who wants it.
- Write changes here, against this code; don't paste code in from elsewhere.
- Example and test data are synthetic (`pixi run make-synthetic`); never commit recordings.
- Anything pushed to GitHub, on any branch, is public at once and stays public: check before
  pushing, not after.

## Workflow

- `main` takes changes through pull requests with passing CI (lint, format check, and tests on
  Ubuntu and macOS).
- Before pushing: `pixi run lint`, `pixi run ruff format --check .` and `pixi run test`
  (`QT_QPA_PLATFORM=offscreen` without a display).
