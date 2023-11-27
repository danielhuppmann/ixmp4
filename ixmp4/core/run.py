from typing import ClassVar, Iterable

import pandas as pd

from ixmp4.data.abstract import Run as RunModel

from .base import BaseFacade, BaseModelFacade
from .iamc import IamcData
from .meta.indicator import RunMetaFacade
from .optimization import OptimizationData


class Run(BaseModelFacade):
    _model: RunModel
    _meta: RunMetaFacade
    NoDefaultVersion: ClassVar = RunModel.NoDefaultVersion
    NotFound: ClassVar = RunModel.NotFound
    NotUnique: ClassVar = RunModel.NotUnique

    def __init__(
        self,
        model: str | None = None,
        scenario: str | None = None,
        version: int | str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if getattr(self, "_model", None) is None:
            if (model is None) or (scenario is None):
                raise TypeError("`Run` requires `model` and `scenario`")

            if version is None:
                self._model = self.backend.runs.get_default_version(model, scenario)
            elif version == "new":
                self._model = self.backend.runs.create(model, scenario)
            elif isinstance(version, int):
                self._model = self.backend.runs.get(model, scenario, version)
            else:
                raise ValueError(
                    "Invalid value for `version`, must be 'new' or integer."
                )
            self.version = self._model.version

        self.iamc = IamcData(_backend=self.backend, run=self._model)
        self._meta = RunMetaFacade(_backend=self.backend, run=self._model)
        self.optimization = OptimizationData(_backend=self.backend, run=self._model)

    @property
    def model(self):
        """Associated model."""
        return self._model.model

    @property
    def scenario(self):
        """Associated scenario."""
        return self._model.scenario

    @property
    def id(self):
        """Unique id."""
        return self._model.id

    @property
    def meta(self):
        "Meta indicator data (`dict`-like)."
        return self._meta

    @meta.setter
    def meta(self, meta):
        self._meta._set(meta)

    def set_as_default(self):
        """Sets this run as default version for this `model - scenario` combination."""
        self.backend.runs.set_as_default_version(self._model.id)

    def unset_as_default(self):
        """Unsets this run as the default version."""
        self.backend.runs.unset_as_default_version(self._model.id)


class RunRepository(BaseFacade):
    def list(self, default_only: bool = True, **kwargs) -> Iterable[Run]:
        return [
            Run(_backend=self.backend, _model=r)
            for r in self.backend.runs.list(default_only=default_only, **kwargs)
        ]

    def tabulate(self, default_only: bool = True, **kwargs) -> pd.DataFrame:
        runs = self.backend.runs.tabulate(default_only=default_only, **kwargs)
        runs["model"] = runs["model__id"].map(self.backend.models.map())
        runs["scenario"] = runs["scenario__id"].map(self.backend.scenarios.map())
        return runs[["id", "model", "scenario", "version", "is_default"]]
