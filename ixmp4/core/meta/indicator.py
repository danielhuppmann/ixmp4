from collections import UserDict

import numpy as np
import pandas as pd

from ixmp4.data.abstract import Run as RunModel

from ..base import BaseFacade


class RunMetaFacade(BaseFacade, UserDict):
    run: RunModel

    def __init__(self, run: RunModel, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.run = run
        self.df, self.data = self._get()

    def _get(self) -> tuple[pd.DataFrame, dict]:
        df = self.backend.meta.tabulate(run_id=self.run.id, run={"default_only": False})
        if df.empty:
            return df, {}
        return df, dict(zip(df["key"], df["value"]))

    def _set(self, meta: dict):
        df = pd.DataFrame({"key": self.data.keys()})
        df["run__id"] = self.run.id
        self.backend.meta.bulk_delete(df)
        df = pd.DataFrame(
            {"key": meta.keys(), "value": [numpy_to_pytype(v) for v in meta.values()]}
        )
        df.dropna(axis=0, inplace=True)
        df["run__id"] = self.run.id
        self.backend.meta.bulk_upsert(df)
        self.df, self.data = self._get()

    def __setitem__(self, key, value: int | float | str | bool):
        try:
            del self[key]
        except KeyError:
            pass

        value = numpy_to_pytype(value)
        if value is not None:
            self.backend.meta.create(self.run.id, key, value)
        self.df, self.data = self._get()

    def __delitem__(self, key):
        id = dict(zip(self.df["key"], self.df["id"]))[key]
        self.backend.meta.delete(id)
        self.df, self.data = self._get()


def numpy_to_pytype(value):
    """Cast numpy-types to basic Python types"""
    if value is np.nan:  # np.nan is cast to 'float', not None
        return None
    elif isinstance(value, np.generic):
        return value.item()
    else:
        return value
