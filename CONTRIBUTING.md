# Contributing to CAPE-Loop

Thank you for helping improve CAPE-Loop. Contributions can include code, tests,
documentation, experiment configurations, reproducibility checks, and careful
reports of methodological problems.

## Before you start

For a bug or a focused documentation correction, open a pull request directly.
For a new experiment, public interface, dependency, or change to the scientific
design, open an issue first so the scope and validation requirements can be
agreed before substantial work begins. Report vulnerabilities privately as
described in [SECURITY.md](SECURITY.md).

All participation in this project is governed by
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## Development environment

The reference core supports Python 3.11 and later and uses only the Python
standard library. A source checkout can be tested without installing packages:

```bash
python3 --version
PYTHONPATH=src python3 -m cape_loop doctor
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_*.py' -v
```

The same checks are available through:

```bash
make check
```

Do not add a runtime dependency without first documenting why the standard
library is insufficient, how the dependency affects offline reproducibility,
and what optional boundary can contain it.

## Scientific integrity requirements

CAPE-Loop is an evaluation repository. A change is not complete merely because
it runs. Contributions must preserve these boundaries:

- The latent user remains fixed within a trajectory unless a separately
  documented study explicitly changes that assumption.
- Visible interaction context and internal policy provenance remain distinct.
- Only evaluation code may inspect latent truth. Policies, updaters, decoders,
  prompts, and model-facing records must not receive it.
- Action-aware and action-unaware comparisons use capacity-matched models and
  frozen data splits where the protocol requires them.
- Counterfactual branches use semantic random keys rather than execution-order
  randomness.
- Generated results must identify their resolved configuration, source
  revision when available, seeds, and retained artifacts.
- Do not replace proposal placeholders, stage-gate outcomes, or result tables
  with empirical claims unless the supporting run artifacts are included and
  independently verifiable.

If a change intentionally revises one of these constraints, explain the new
estimand and update the proposal-facing documentation in the same pull request.

## Tests

Use `unittest` for the standard-library test suite. Name files `test_*.py` and
keep tests deterministic, isolated, and offline. Tests must not require API
credentials, network access, wall-clock timing assumptions, or undeclared
files outside the repository.

For behavior changes:

1. Add a test that fails without the change.
2. Cover invalid inputs and information-access boundaries, not only the happy
   path.
3. Prefer small identifying fixtures with an obvious expected result.
4. Keep generated temporary data outside tracked artifact directories.
5. Run `make check` with a supported Python version.

Changes to statistical procedures should also state the unit of analysis,
randomization unit, uncertainty procedure, and any multiple-comparison
correction.

## Documentation and records

Public components should document their responsibility, inputs, outputs,
latent-truth access, and extension point. Schema changes require an explicit
schema-version decision and migration or compatibility notes. Configuration
examples must distinguish reference defaults from preregistered or paper-run
settings.

Never commit secrets, API keys, private user data, raw human-study data, or
provider responses that cannot be redistributed. Use synthetic fixtures for
tests. Review generated artifacts for sensitive content before adding them.

## Pull requests

Keep each pull request focused. Complete the pull request template and include:

- the problem and the chosen approach;
- the scientific or user-visible impact;
- the exact verification commands run;
- documentation and schema implications;
- any generated artifacts and how they can be reproduced; and
- limitations or follow-up work that remains.

By submitting a contribution, you agree that it may be distributed under the
Apache License 2.0 in [LICENSE](LICENSE).
