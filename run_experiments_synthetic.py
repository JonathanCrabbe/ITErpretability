import argparse
import sys
from typing import Any

import src.iterpretability.logger as log
from src.iterpretability.synthetic_experiment import (PredictiveSensitivity, PairwiseInteractionSensitivity,
                                                      NonLinearitySensitivity, PropensitySensitivity)


def init_arg() -> Any:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", default="propensity_sensitivity", type=str)
    parser.add_argument("--train_ratio", default=0.8, type=float)

    # Arguments for Predictive Sensitivity Experiment
    parser.add_argument("--synthetic_simulator_type", default='linear', type=str)
    parser.add_argument("--random_feature_selection", default=True, type=bool)

    parser.add_argument(
        "--dataset_list",
        nargs="+",
        type=str,
        default=["tcga_100"]
        # "tcga_20",
        # "tcga_100",
        # "tcga_100"],
    )

    parser.add_argument(
        "--num_important_features_list",
        nargs="+",
        type=int,
        default=[20]
        # 4,
        # 20,
        # 20],
    )

    parser.add_argument(
        "--binary_outcome_list",
        nargs="+",
        type=bool,
        default=[False],
    )

    # Arguments for Propensity Sensitivity Experiment
    parser.add_argument("--propensity_types", default=["top_pred", "top_prog", #"pred", "prog", #
                                                       "irrelevant_var"],
                        type=str, nargs="+")
    parser.add_argument(
        "--propensity_scales", nargs="+", default=[0, 1, 10], type=float
    )
    parser.add_argument("--predictive_scale", default=1, type=float)
    parser.add_argument(
        "--seed_list", nargs="+", default=[42, 666, 25, 77, 55
                                           # 42, 666, 25, 77, 55, 88, 99, 10, 2, 50
                                           ], type=int
    )
    parser.add_argument(
        "--explainer_list",
        nargs="+",
        type=str,
        default=[#"feature_ablation",
                 #"feature_permutation",
                 "integrated_gradients",
                 #"shapley_value_sampling",
                 #"lime"
        ],
    )
    parser.add_argument("--explainer_limit", type=int, default=100)

    return parser.parse_args()


if __name__ == "__main__":
    log.add(sink=sys.stderr, level="INFO")
    args = init_arg()
    for seed in args.seed_list:
        log.info(
            f"Experiment {args.experiment_name} with simulator {args.synthetic_simulator_type} and seed {seed}.")
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
        elif args.experiment_name == "interaction_sensitivity":
            exp = PairwiseInteractionSensitivity(
                seed=seed,
                explainer_limit=args.explainer_limit,
                synthetic_simulator_type=args.synthetic_simulator_type
            )
            for experiment_id in range(len(args.dataset_list)):
                log.info(
                    f"Running experiment for {args.dataset_list[experiment_id]}, "
                    f"{args.num_important_features_list[experiment_id]} important features "
                    f"with binary outcome {args.binary_outcome_list[experiment_id]}")

                exp.run(
                    dataset=args.dataset_list[experiment_id],
                    train_ratio=args.train_ratio,
                    num_important_features=args.num_important_features_list[experiment_id],
                    binary_outcome=args.binary_outcome_list[experiment_id],
                    explainer_list=args.explainer_list,
                )
        elif args.experiment_name == "nonlinearity_sensitivity":
            exp = NonLinearitySensitivity(
                seed=seed,
                explainer_limit=args.explainer_limit,
                synthetic_simulator_type=args.synthetic_simulator_type
            )
            for experiment_id in range(len(args.dataset_list)):
                log.info(
                    f"Running experiment for {args.dataset_list[experiment_id]}, "
                    f"{args.num_important_features_list[experiment_id]} important features "
                    f"with binary outcome {args.binary_outcome_list[experiment_id]}")

                exp.run(
                    dataset=args.dataset_list[experiment_id],
                    train_ratio=args.train_ratio,
                    num_important_features=args.num_important_features_list[experiment_id],
                    binary_outcome=args.binary_outcome_list[experiment_id],
                    explainer_list=args.explainer_list,
                )
        elif args.experiment_name == "propensity_sensitivity":
            for propensity_type in args.propensity_types:
                exp = PropensitySensitivity(
                    seed=seed,
                    explainer_limit=args.explainer_limit,
                    synthetic_simulator_type=args.synthetic_simulator_type,
                    propensity_type=propensity_type,
                    propensity_scales=args.propensity_scales
                )
                for experiment_id in range(len(args.dataset_list)):
                    log.info(
                        f"Running experiment for {args.dataset_list[experiment_id]}, "
                        f"{args.num_important_features_list[experiment_id]},"
                        f"propensity type {propensity_type}, with "
                        f"binary outcome {args.binary_outcome_list[experiment_id]}")

                    exp.run(
                        dataset=args.dataset_list[experiment_id],
                        train_ratio=args.train_ratio,
                        num_important_features=args.num_important_features_list[experiment_id],
                        random_feature_selection=args.random_feature_selection,
                        binary_outcome=args.binary_outcome_list[experiment_id],
                        explainer_list=args.explainer_list,
                        predictive_scale=args.predictive_scale
                    )
        else:
            raise ValueError("The experiment name is invalid.")
