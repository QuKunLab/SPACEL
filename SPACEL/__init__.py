import warnings
import pandas as pd
import logging
from ._version import __version__

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

# level = logging.INFO
# spoint_logger = logging.getLogger("spoint")
# spoint_logger.propagate = False
# spoint_logger.setLevel(level)
# if len(spoint_logger.handlers) == 0:
#     console = Console(force_terminal=True)
#     if console.is_jupyter is True:
#         console.is_jupyter = False
#     ch = RichHandler(
#         level=level, show_path=False, console=console, show_time=False
#     )
#     formatter = logging.Formatter("%(message)s")
#     ch.setFormatter(formatter)
#     spoint_logger.addHandler(ch)
