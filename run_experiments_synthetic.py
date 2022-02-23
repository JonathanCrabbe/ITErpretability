import argparse
import sys
from typing import Any

import src.iterpretability.logger as log
from src.iterpretability.synthetic_experiment import PredictiveSensitivity


def init_arg() -> Any:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", default="predictive_sensitivity", type=str)
    parser.add_argument("--dataset", default="tcga_10", type=str)
    parser.add_argument("--num_important_features", default=2, type=int)
    parser.add_argument("--random_feature_selection", default=True, type=bool)
    parser.add_argument("--binary_outcome", default=False, type=bool)

    # Arguments for Propensity Sensitivity Experiment
    parser.add_argument("--treatment_assgn", default="top_pred", type=str)
    parser.add_argument(
        "--prop_scales", nargs="+", default=[0, 0.1, 0.5, 1], type=float
    )
    parser.add_argument("--predictive_scale", default=1.0, type=float)
    parser.add_argument(
        "--seed_list", nargs="+", default=[42, 666, 25, 77, 55], type=int
    )
    parser.add_argument(
        "--explainer_list",
        nargs="+",
        type=str,
        default=["feature_ablation",
            "feature_permutation",
            "integrated_gradients",
            "shapley_value_sampling",
            "lime"],
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
        if args.experiment_name == "predictive_sensitivity":
            exp = PredictiveSensitivity(
                seed=seed,
                explainer_limit=args.explainer_limit,
                binary_outcome=args.binary_outcome,
            )
            exp.run(
                dataset=args.dataset,
                num_important_features=args.num_important_features,
                random_feature_selection=args.random_feature_selection,
                explainer_list=args.explainer_list,
            )
