import os
import numpy
import pandas as pd
from typing import Union, Hashable, Iterable

# PathLike: The PathLike type is used for defining a file path.
PathLike = Union[str, os.PathLike]
ArrayLike = Union[numpy.ndarray, pd.DataFrame, Iterable]
KeyType = Hashable