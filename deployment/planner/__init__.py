"""Deployment planner architecture layer.

The planner consumes capability profiles, measured artifacts, and policy
candidates to produce deployment recommendations. Importing this package does
not run benchmarks or modify runtime behavior.
"""

from deployment.planner.constraint_solver import evaluate_constraints
from deployment.planner.deployment_plan_schema import TRUTH_BOUNDARY, build_deployment_plan
from deployment.planner.planner import DeploymentPlanner, plan_deployment
from deployment.planner.recommendation_engine import recommend_candidate

__all__ = [
    "DeploymentPlanner",
    "TRUTH_BOUNDARY",
    "build_deployment_plan",
    "evaluate_constraints",
    "plan_deployment",
    "recommend_candidate",
]
