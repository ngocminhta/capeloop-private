"""Minimal structured exact update.

Run with:

    PYTHONPATH=src python examples/structured_update.py
"""

import json

from cape_loop.beliefs import JointThetaPsiBelief
from cape_loop.domains import get_domain
from cape_loop.elicitation import build_matched_anchor_set
from cape_loop.inference import exact_aware_update
from cape_loop.response import RandomUtilityModel
from cape_loop.schemas import Observation, Susceptibility


domain = get_domain("travel")
context = build_matched_anchor_set(
    domain,
    scenario_id="example-budget",
    target_attribute=0,
).context("balanced")
model = RandomUtilityModel()
prior = JointThetaPsiBelief.uniform(
    susceptibilities=(
        Susceptibility(0.15, 0.15, 0.15),
        Susceptibility(0.45, 0.45, 0.45),
        Susceptibility(0.85, 0.85, 0.85),
    )
)
observation = Observation(
    selected_option_id=context.ranking[0],
    surface_response="The first option works.",
    choice_noise_key="example",
)
posterior = exact_aware_update(prior, context, observation, model)
print(json.dumps(posterior.theta_belief().to_dict(), indent=2))
