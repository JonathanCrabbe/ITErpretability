# third party
import sys
import warnings

import optuna

#from . import experiment  # noqa: F401
#from . import explain  # noqa: F401
#from . import hyperparam_search  # noqa: F401
#from . import logger  # noqa: F401
#from . import simulate  # noqa: F401

optuna.logging.disable_propagation()
optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.


#logger.add(sink=sys.stderr, level="CRITICAL")

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings(
    "ignore", category=optuna.exceptions.ExperimentalWarning, module="optuna"
)
