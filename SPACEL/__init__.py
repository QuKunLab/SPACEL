import warnings
import pandas as pd
import logging
from ._version import __version__

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)