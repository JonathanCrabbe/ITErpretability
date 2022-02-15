import argparse
import sys
from typing import Any

import iterpretability.logger as log
from iterpretability.experiment import PrognosticSensitivity, PropensitySensitivity


def init_arg() -> Any:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", default="prop_sensitivity", type=str)

    # Arguments for Prognostic Sensitivity Experiment
    parser.add_argument(
        "--prog_masks", nargs="+", default=[1e-3, 1e-2, 1e-1, 1], type=float
    )
    # Arguments for Propensity Sensitivity Experiment

    parser.add_argument("--treatment_assgn", default="top_pred", type=str)
    parser.add_argument(
        "--prop_scales", nargs="+", default=[0, 0.1, 0.5, 1], type=float
    )
    parser.add_argument("--prognostic_mask", default=0.1, type=float)
    parser.add_argument(
        "--seed_list", nargs="+", default=[42, 666, 25, 77, 55], type=int
    )
    parser.add_argument(
        "--explainer_list",
        nargs="+",
        type=str,
        default=["integrated_gradients", "shapley_value_sampling"],
    )
    parser.add_argument(
        "--learner_list",
        nargs="+",
        type=str,
        default=["TARNet", "CFRNet", "SNet", "SNet_noprop", "TLearner", "SLearner"],
    )
    parser.add_argument("--run_name", type=str, default="results")
    parser.add_argument("--explainer_limit", type=int, default=100)
    parser.add_argument("--n_layers_r", type=int, default=1)
    parser.add_argument("--ortho_reg_type", type=str, default="abs")
    parser.add_argument("--penalty_orthogonal", type=float, default=0.01)
    parser.add_argument("--addvar_scale", type=float, default=0.1)
    return parser.parse_args()


if __name__ == "__main__":
    log.add(sink=sys.stderr, level="INFO")
    args = init_arg()
    for seed in args.seed_list:
        log.info(f"Experiment {args.experiment_name} with seed {seed}")
        if args.experiment_name == "prop_sensitivity":
            exp = PropensitySensitivity(
                prognostic_mask=args.prognostic_mask,
                prop_scales=args.prop_scales,
                treatment_assgn=args.treatment_assgn,
                seed=seed,
                explainer_limit=args.explainer_limit,
                addvar_scale=args.addvar_scale,
            )
            exp.run(
                explainer_list=args.explainer_list,
                learner_list=args.learner_list,
                run_name=args.run_name,
                ortho_reg_type=args.ortho_reg_type,
                penalty_orthogonal=args.penalty_orthogonal,
            )
        elif args.experiment_name == "prog_sensitivity":
            exp = PrognosticSensitivity(
                seed=seed,
                explainer_limit=args.explainer_limit,
                prognostic_masks=args.prog_masks,
            )
            exp.run(
                explainer_list=args.explainer_list,
            )
