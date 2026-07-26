## Summary

Describe the problem, the change, and why this scope is appropriate.

## Scientific and user-visible impact

State whether this changes an estimand, information boundary, schema, reference
default, experiment, metric, CLI behavior, or artifact format. Write "none" when
it does not.

## Verification

List the exact commands run and their outcomes.

```text
make check
```

## Reproducibility and artifacts

Describe new or changed configurations, seeds, fixtures, generated artifacts,
and the commands needed to reproduce them. Do not report a stage gate or paper
finding without retained, verifiable evidence.

## Checklist

- [ ] The change is focused and includes tests for changed behavior.
- [ ] `make check` passes with a supported Python version.
- [ ] Tests run offline and do not require credentials or private data.
- [ ] Visible context, policy provenance, and latent truth remain separated.
- [ ] Policies, updaters, decoders, and prompts cannot access latent truth.
- [ ] Random behavior uses stable semantic keys where pairing is required.
- [ ] Schema or configuration changes include compatibility notes.
- [ ] Public components and user-facing behavior are documented.
- [ ] Generated claims are backed by retained artifacts; no result was fabricated.
- [ ] No secrets, personal data, or non-redistributable content are included.
- [ ] I have read and will follow the Code of Conduct.
