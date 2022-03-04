import argparse
import sys
from typing import Any

import src.iterpretability.logger as log
from src.iterpretability.synthetic_experiment import PredictiveSensitivity


def init_arg() -> Any:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", default="predictive_sensitivity", type=str)
    parser.add_argument("--train_ratio", default=0.8, type=float)

    # Arguments for Predictive Sensitivity Experiment
    parser.add_argument("--synthetic_simulator_type", default='linear_most_correlated', type=str)
    parser.add_argument("--random_feature_selection", default=True, type=bool)

    parser.add_argument(
        "--dataset_list",
        nargs="+",
        type=str,
        default=["tcga_20",
                 "tcga_20",
                 "tcga_100",
                 "tcga_100"],
    )

    parser.add_argument(
        "--num_important_features_list",
        nargs="+",
        type=int,
        default=[4,
                 4,
                 20,
                 20],
    )

    parser.add_argument(
        "--binary_outcome_list",
        nargs="+",
        type=bool,
        default=[False,
                 True,
                 False,
                 True],
    )

    # Arguments for Propensity Sensitivity Experiment
    parser.add_argument("--treatment_assgn", default="top_pred", type=str)
    parser.add_argument(
        "--prop_scales", nargs="+", default=[0, 0.1, 0.5, 1], type=float
    )
    parser.add_argument("--predictive_scale", default=1.0, type=float)
    parser.add_argument(
        "--seed_list", nargs="+", default=[42, 666, 25, 77, 55, 88, 99, 10, 2, 50], type=int
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
        log.info(f"Experiment {args.experiment_name} with simulator {args.synthetic_simulator_type} and seed {seed}.")
        if args.experiment_name == "predictive_sensitivity":
            exp = PredictiveSensitivity(
                seed=seed,
                explainer_limit=args.explainer_limit,
                synthetic_simulator_type=args.synthetic_simulator_type,
            )
            for experiment_id in range(len(args.dataset_list)):
                log.info(
                    f"Running experiment for {args.dataset_list[experiment_id]}, {args.num_important_features_list[experiment_id]} with binary outcome {args.binary_outcome_list[experiment_id]}")

                exp.run(
                    dataset=args.dataset_list[experiment_id],
                    train_ratio=args.train_ratio,
                    num_important_features=args.num_important_features_list[experiment_id],
                    random_feature_selection=args.random_feature_selection,
                    binary_outcome=args.binary_outcome_list[experiment_id],
                    explainer_list=args.explainer_list,
                )
