# Data splits and leakage controls

CAPE-Loop treats a split as a collection of complete semantic groups, not a
random partition of rows. Training, development, and test are disjoint on all
six axes declared by the paper:

| Axis | Executable boundary |
| --- | --- |
| Latent preference | The complete three-attribute `theta` tuple is assigned to one split |
| Susceptibility | The complete ranking/default/suggestion tuple is assigned to one split |
| Option template | Feature-matched atlas, beacon, and cedar option families are used for train, development, and test |
| Dialogue template | Each split receives a distinct opaque visible wording-template ID |
| Scenario family | Each split receives a distinct scenario-family prefix |
| Paraphrase template | Content-addressed families belong to exactly one split |

The names atlas, beacon, and cedar are intentionally opaque. They prevent the
words “train” and “test” from entering an LLM-visible option while still making
surface identity auditable. Option features remain matched across the three
families, so the scientific response-model contrast does not change merely
because leakage is blocked.

## How the runner uses the split

`runner._prepare_study` creates three feature-matched `DomainSpec` variants:

```text
train        -> atlas option/dialogue/scenario families
development  -> beacon option/dialogue/scenario families
test         -> cedar option/dialogue/scenario families
```

The fitted likelihoods consume only atlas interactions. Temperature
calibration and held-out response diagnostics consume only beacon
interactions. Experiments A–C consume cedar interaction assets. The terminal-v2
battery uses an additional test-only family with novel IDs, feature values,
scenario families, and wording.

Natural-language response templates are managed by
`build_default_paraphrase_suite()`. Before Experiment A evaluates its test
families, the runner calls the suite’s no-test-leakage guard against every
train/development template ID, content hash, and literal pattern.

## Retained proof

Every A–C run writes:

```text
splits.json
metrics/split-leakage-audit.json
events/fitted-model-training.jsonl       # when events are retained
events/fitted-model-development.jsonl    # when events are retained
```

The audit records concrete asset counts, pairwise overlap sets, every
manifest-to-generator binding checked, the paraphrase suite digest, and a
status. A nonempty overlap or a manifest assignment inconsistent with the
generator aborts the run; it is never converted into a warning.

This proves software-level split execution. It does not prove that a researcher
avoided looking at test results while tuning a prompt or threshold. That
procedural boundary still requires preregistration and release review.
