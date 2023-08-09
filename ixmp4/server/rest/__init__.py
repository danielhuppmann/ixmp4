from datetime import datetime

from fastapi import FastAPI, Request, Depends, Path
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware

from ixmp4.conf import settings
from ixmp4.core.exceptions import IxmpError

from . import unit, scenario, region, meta, run, model, docs, deps
from .base import BaseModel
from .iamc import (
    datapoint,
    timeseries,
    model as iamc_model,
    scenario as iamc_scenario,
    region as iamc_region,
    variable as iamc_variable,
    unit as iamc_unit,
)


v1 = FastAPI()

v1.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

v1.include_router(run.router)
v1.include_router(meta.router)
v1.include_router(datapoint.router, prefix="/iamc")
v1.include_router(timeseries.router, prefix="/iamc")
v1.include_router(iamc_model.router, prefix="/iamc")
v1.include_router(iamc_scenario.router, prefix="/iamc")
v1.include_router(iamc_region.router, prefix="/iamc")
v1.include_router(iamc_unit.router, prefix="/iamc")
v1.include_router(iamc_variable.router, prefix="/iamc")
v1.include_router(region.router)
v1.include_router(scenario.router)
v1.include_router(model.router)
v1.include_router(unit.router)
v1.include_router(docs.router)


class APIInfo(BaseModel):
    name: str
    version: str
    is_managed: bool
    manager_url: None | str
    utcnow: datetime


@v1.get("/", response_model=APIInfo)
def root(
    platform: str = Path("default"),
    version: str = Depends(deps.get_version),
):
    return APIInfo(
        name=platform,
        version=version,
        is_managed=settings.managed,
        manager_url=settings.manager_url,
        utcnow=datetime.utcnow(),
    )


@v1.exception_handler(IxmpError)
async def http_exception_handler(request: Request, exc: IxmpError):
    return JSONResponse(
        content=jsonable_encoder(
            {
                "message": exc._message,
                "kwargs": exc.kwargs,
                "error_name": exc.http_error_name,
            }
        ),
        status_code=exc.http_status_code,
    )
