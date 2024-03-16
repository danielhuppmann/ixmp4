from typing import ClassVar

from ixmp4.data.db import filters as base
from ixmp4.data.db.model import Model
from ixmp4.data.db.run import Run
from ixmp4.data.db.scenario import Scenario
from ixmp4.db import filters, utils

from .model import RunMetaEntry


class ModelFilter(base.RunFilter, metaclass=filters.FilterMeta):
    name: filters.String

    sqla_model: ClassVar[type] = Model


class ScenarioFilter(base.RunFilter, metaclass=filters.FilterMeta):
    name: filters.String

    sqla_model: ClassVar[type] = Scenario


class RunFilter(base.RunFilter, metaclass=filters.FilterMeta):
    def join(self, exc, **kwargs):
        if not utils.is_joined(exc, Run):
            exc = exc.join(Run, onclause=RunMetaEntry.run__id == Run.id)
        return exc


class RunMetaEntryFilter(base.RunMetaEntryFilter, metaclass=filters.FilterMeta):
    model: ModelFilter
    scenario: ScenarioFilter
    run: RunFilter = filters.Field(
        default=RunFilter(id=None, version=None, is_default=True)
    )
