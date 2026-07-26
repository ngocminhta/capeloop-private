# Ethics and limitations

CAPE-Loop studies how persistent agents interpret user evidence. Profile data,
conversation histories, and agent actions can be sensitive even when the
reference implementation uses synthetic users. This page defines interpretation
and use boundaries.

## What the benchmark does not establish

- It does not show that an agent changes a user’s underlying preference.
- It does not prove that the declared random-utility simulator is a universal
  theory of human choice.
- It does not make the exact posterior normatively correct outside the declared
  model.
- It does not establish a paper hypothesis merely because an experiment or
  metric is implemented.
- It does not show that all LLMs, prompts, memory architectures, domains, or user
  populations exhibit the same behavior.
- It does not establish that a provenance-aware prompt is a complete safety
  solution.
- It does not license deployment decisions based only on synthetic results.

## Fixed latent preference

The primary design intentionally fixes \(\theta\). Presentation changes response
probability but not preference or welfare. Real people may learn, adapt, change
their goals, strategically respond, or interpret recommendations as information.
Those processes are outside the core causal estimand.

Calling policy-conditioned observation “preference change” would misstate the
study.

## Simulator dependence

Exact action-aware inference is optimal under a finite declared response model.
Misspecification can affect both the reference and the apparent size/order of
evidence effects. The fitted-aware comparison, alternative response models,
parameter sweeps, natural sampling, and human pragmatic judgments reduce but do
not eliminate this limitation.

Results should identify the response model and sensitivity region. A finding
that occurs only under extreme acquiescence or forced restriction should be
reported as such.

## Human judgments

Human evidence-strength ratings validate pragmatic intuitions about what an
interaction supports. Annotators do not have privileged access to a person’s
“true” preference. Agreement with humans is neither necessary nor sufficient for
causal correctness.

The repository’s packet generator is only a material aid. Before recruitment,
researchers must address:

- institutional or jurisdiction-specific ethics review;
- informed consent and withdrawal;
- fair compensation;
- accessibility and language;
- exposure to potentially manipulative scenarios;
- collection minimization;
- retention and deletion;
- cross-border processing;
- publication and reuse terms.

No participant data should be committed merely because the packet schema exists.

## Privacy

Real interaction histories can reveal preferences, location, finances, health,
identity, or vulnerabilities. Pseudonymous IDs do not guarantee anonymity when
free text remains linkable.

When adapting CAPE-Loop to real data:

- collect only fields necessary for the registered analysis;
- separate identity/contact information from research records;
- remove direct and indirect identifiers;
- restrict raw text access;
- document external model processing and provider retention;
- define deletion and incident-response procedures;
- review whether released combinations enable re-identification;
- publish aggregates or controlled-access artifacts when raw release is unsafe.

Never store credentials or participant identifiers in configs, prompts, run
manifests, issue reports, or public artifacts.

## Manipulation and user autonomy

The benchmark deliberately models rankings, defaults, and suggestions that can
shape choices. These mechanisms can be used manipulatively in real systems.
Research deployments should not use profile-conditioned framing to steer users
toward outcomes that benefit the system operator, especially in high-stakes
domains.

Provenance logging can improve auditability, but logging more user data also
increases privacy risk. Collect the minimum causal metadata required, apply
retention limits, and make meaningful user correction/deletion mechanisms
available.

## High-stakes domains

Travel and writing are controlled research domains, not proof of safety in
health, employment, housing, credit, legal, education, or political contexts.
Those domains involve different welfare, consent, regulatory, and harm
structures. Do not transfer thresholds or conclusions without new validation
and governance.

## External LLMs

Provider models may change behind stable aliases, behave nondeterministically,
filter inputs, retain content, or restrict output redistribution. Record
available version metadata and timestamps, retain replay records where allowed,
and disclose missing raw data.

The core intentionally has no live-provider integration. An external harness is
responsible for credentials, rate limits, provider terms, and data processing.

## Bias and representativeness

The finite preference support, English-language templates, and two domains omit
many users and communication norms. Acquiescence and interpretation of defaults
can vary by culture, language, disability, expertise, and power relationship.
Presentation susceptibility should not be treated as a stable personal deficit
or used to target persuasive interventions.

## False profiles and correction

An intentionally wrong seed is an experimental intervention, not an acceptable
production practice. Real systems should expose profile state, evidence
provenance, correction, and deletion where feasible.

Correction-debt analysis is stage-gated. Slow simulated recovery does not
establish that users should repeatedly correct an agent or bear responsibility
for system-generated profile errors.

## Artifact and claim integrity

- Label smoke, pilot, and paper evidence.
- Retain failed and missing cells.
- Never copy placeholder values into paper tables.
- Do not declare a gate passed without complete retained analysis.
- Report negative or null results.
- Separate code correctness from hypothesis support.
- Respect third-party licenses and provider output terms.

Security or privacy issues should be reported through [SECURITY.md](../SECURITY.md),
not a public issue containing sensitive records.
