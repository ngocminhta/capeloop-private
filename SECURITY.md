# Security policy

## Supported versions

Before the first tagged release, the default branch is the only supported
version. After releases begin, security fixes target the default branch and the
latest tagged release. Older releases are supported only when maintainers
explicitly say so in release notes.

## Reporting a vulnerability

Please do not disclose a suspected vulnerability in a public issue, discussion,
pull request, dataset, or experiment artifact.

Use the repository host's private vulnerability-reporting feature when it is
available. Otherwise, contact a maintainer through a private channel listed by
the repository host and ask for a secure reporting method. If no private
channel is visible, a public issue may ask for private contact, but it must not
include vulnerability details.

Include only the information needed to reproduce and assess the problem:

- affected revision or release;
- affected component and configuration;
- impact and plausible attack scenario;
- minimal reproduction steps or proof of concept;
- known mitigations; and
- whether any credentials, personal data, or unreleased artifacts may have
  been exposed.

Maintainers will acknowledge reports as soon as practical, investigate them,
and coordinate a fix and disclosure with the reporter. Do not test against
systems, accounts, or data you do not own or have explicit permission to use.

## Security-sensitive areas

Reports are especially helpful for:

- command or path injection in configuration and artifact handling;
- unsafe deserialization or archive extraction;
- checksum or provenance verification bypasses;
- accidental network access in offline workflows;
- credential exposure in logs, prompts, fixtures, or run manifests;
- private or human-subject data included in published artifacts; and
- model-provider integrations that transmit more information than documented.

Scientific correctness problems without a security or privacy impact should use
the scientific issue template instead.

## Secrets and sensitive data

The standard-library core does not require credentials. Optional integrations
must read secrets from the environment, never from committed configuration.
Synthetic data should be used in tests. Human-study data must follow the
applicable consent, ethics-review, retention, and release plan; this repository
does not itself grant permission to redistribute such data.
