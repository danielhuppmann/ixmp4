# type: ignore
"""Initial Migration

Revision ID: c71efc396d2b
Revises:
Create Date: 2023-04-26 15:37:46.677955

"""

import sqlalchemy as sa
from alembic import op

# Revision identifiers, used by Alembic.
revision = "c71efc396d2b"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table(
        "iamc_variable",
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("created_by", sa.String(length=255), nullable=True),
        sa.Column(
            "id",
            sa.Integer(),
            sa.Identity(always=False, on_null=True, start=1, increment=1),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_iamc_variable")),
        sa.UniqueConstraint("name", name=op.f("uq_iamc_variable_name")),
    )
    op.create_table(
        "model",
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("created_by", sa.String(length=255), nullable=True),
        sa.Column(
            "id",
            sa.Integer(),
            sa.Identity(always=False, on_null=True, start=1, increment=1),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_model")),
        sa.UniqueConstraint("name", name=op.f("uq_model_name")),
    )
    op.create_table(
        "region",
        sa.Column("name", sa.String(length=1023), nullable=False),
        sa.Column("hierarchy", sa.String(length=1023), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("created_by", sa.String(length=255), nullable=True),
        sa.Column(
            "id",
            sa.Integer(),
            sa.Identity(always=False, on_null=True, start=1, increment=1),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_region")),
        sa.UniqueConstraint("name", name=op.f("uq_region_name")),
    )
    op.create_table(
        "scenario",
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("created_by", sa.String(length=255), nullable=True),
        sa.Column(
            "id",
            sa.Integer(),
            sa.Identity(always=False, on_null=True, start=1, increment=1),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_scenario")),
        sa.UniqueConstraint("name", name=op.f("uq_scenario_name")),
    )
    op.create_table(
        "unit",
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("created_by", sa.String(length=255), nullable=True),
        sa.Column(
            "id",
            sa.Integer(),
            sa.Identity(always=False, on_null=True, start=1, increment=1),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_unit")),
        sa.UniqueConstraint("name", name=op.f("uq_unit_name")),
    )
    op.create_table(
        "iamc_measurand",
        sa.Column("variable__id", sa.Integer(), nullable=False),
        sa.Column("unit__id", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("created_by", sa.String(length=255), nullable=True),
        sa.Column(
            "id",
            sa.Integer(),
            sa.Identity(always=False, on_null=True, start=1, increment=1),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["unit__id"],
            ["unit.id"],
            name=op.f("fk_iamc_measurand_unit__id_unit"),
        ),
        sa.ForeignKeyConstraint(
            ["variable__id"],
            ["iamc_variable.id"],
            name=op.f("fk_iamc_measurand_variable__id_iamc_variable"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_iamc_measurand")),
        sa.UniqueConstraint(
            "variable__id",
            "unit__id",
            name=op.f("uq_iamc_measurand_variable__id_unit__id"),
        ),
    )
    with op.batch_alter_table("iamc_measurand", schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f("ix_iamc_measurand_unit__id"),
            ["unit__id"],
            unique=False,
        )
        batch_op.create_index(
            batch_op.f("ix_iamc_measurand_variable__id"),
            ["variable__id"],
            unique=False,
        )

    op.create_table(
        "iamc_variable_docs",
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("dimension__id", sa.Integer(), nullable=True),
        sa.Column(
            "id",
            sa.Integer(),
            sa.Identity(always=False, on_null=True, start=1, increment=1),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["dimension__id"],
            ["iamc_variable.id"],
            name=op.f("fk_iamc_variable_docs_dimension__id_iamc_variable"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_iamc_variable_docs")),
        sa.UniqueConstraint(
            "dimension__id", name=op.f("uq_iamc_variable_docs_dimension__id")
        ),
    )
    op.create_table(
        "model_docs",
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("dimension__id", sa.Integer(), nullable=True),
        sa.Column(
            "id",
            sa.Integer(),
            sa.Identity(always=False, on_null=True, start=1, increment=1),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["dimension__id"],
            ["model.id"],
            name=op.f("fk_model_docs_dimension__id_model"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_model_docs")),
        sa.UniqueConstraint("dimension__id", name=op.f("uq_model_docs_dimension__id")),
    )
    op.create_table(
        "region_docs",
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("dimension__id", sa.Integer(), nullable=True),
        sa.Column(
            "id",
            sa.Integer(),
            sa.Identity(always=False, on_null=True, start=1, increment=1),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["dimension__id"],
            ["region.id"],
            name=op.f("fk_region_docs_dimension__id_region"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_region_docs")),
        sa.UniqueConstraint("dimension__id", name=op.f("uq_region_docs_dimension__id")),
    )
    op.create_table(
        "run",
        sa.Column("model__id", sa.Integer(), nullable=False),
        sa.Column("scenario__id", sa.Integer(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("is_default", sa.Boolean(), nullable=False),
        sa.Column(
            "id",
            sa.Integer(),
            sa.Identity(always=False, on_null=True, start=1, increment=1),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["model__id"], ["model.id"], name=op.f("fk_run_model__id_model")
        ),
        sa.ForeignKeyConstraint(
            ["scenario__id"],
            ["scenario.id"],
            name=op.f("fk_run_scenario__id_scenario"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_run")),
        sa.UniqueConstraint(
            "model__id",
            "scenario__id",
            "version",
            name=op.f("uq_run_model__id_scenario__id_version"),
        ),
    )
    with op.batch_alter_table("run", schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f("ix_run_model__id"), ["model__id"], unique=False
        )
        batch_op.create_index(
            batch_op.f("ix_run_scenario__id"), ["scenario__id"], unique=False
        )

    op.create_table(
        "scenario_docs",
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("dimension__id", sa.Integer(), nullable=True),
        sa.Column(
            "id",
            sa.Integer(),
            sa.Identity(always=False, on_null=True, start=1, increment=1),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["dimension__id"],
            ["scenario.id"],
            name=op.f("fk_scenario_docs_dimension__id_scenario"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_scenario_docs")),
        sa.UniqueConstraint(
            "dimension__id", name=op.f("uq_scenario_docs_dimension__id")
        ),
    )
    op.create_table(
        "unit_docs",
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("dimension__id", sa.Integer(), nullable=True),
        sa.Column(
            "id",
            sa.Integer(),
            sa.Identity(always=False, on_null=True, start=1, increment=1),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["dimension__id"],
            ["unit.id"],
            name=op.f("fk_unit_docs_dimension__id_unit"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_unit_docs")),
        sa.UniqueConstraint("dimension__id", name=op.f("uq_unit_docs_dimension__id")),
    )
    op.create_table(
        "iamc_timeseries",
        sa.Column("region__id", sa.Integer(), nullable=False),
        sa.Column("measurand__id", sa.Integer(), nullable=False),
        sa.Column("run__id", sa.Integer(), nullable=False),
        sa.Column(
            "id",
            sa.Integer(),
            sa.Identity(always=False, on_null=True, start=1, increment=1),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["measurand__id"],
            ["iamc_measurand.id"],
            name=op.f("fk_iamc_timeseries_measurand__id_iamc_measurand"),
        ),
        sa.ForeignKeyConstraint(
            ["region__id"],
            ["region.id"],
            name=op.f("fk_iamc_timeseries_region__id_region"),
        ),
        sa.ForeignKeyConstraint(
            ["run__id"],
            ["run.id"],
            name=op.f("fk_iamc_timeseries_run__id_run"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_iamc_timeseries")),
        sa.UniqueConstraint(
            "run__id",
            "region__id",
            "measurand__id",
            name=op.f("uq_iamc_timeseries_run__id_region__id_measurand__id"),
        ),
    )
    with op.batch_alter_table("iamc_timeseries", schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f("ix_iamc_timeseries_measurand__id"),
            ["measurand__id"],
            unique=False,
        )
        batch_op.create_index(
            batch_op.f("ix_iamc_timeseries_region__id"),
            ["region__id"],
            unique=False,
        )
        batch_op.create_index(
            batch_op.f("ix_iamc_timeseries_run__id"), ["run__id"], unique=False
        )

    op.create_table(
        "runmetaentry",
        sa.Column("run__id", sa.Integer(), nullable=False),
        sa.Column("key", sa.String(length=1023), nullable=False),
        sa.Column("type", sa.String(length=20), nullable=False),
        sa.Column("value_int", sa.Integer(), nullable=True),
        sa.Column("value_str", sa.String(length=1023), nullable=True),
        sa.Column("value_float", sa.Float(), nullable=True),
        sa.Column("value_bool", sa.Boolean(), nullable=True),
        sa.Column(
            "id",
            sa.Integer(),
            sa.Identity(always=False, on_null=True, start=1, increment=1),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["run__id"], ["run.id"], name=op.f("fk_runmetaentry_run__id_run")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_runmetaentry")),
        sa.UniqueConstraint("run__id", "key", name=op.f("uq_runmetaentry_run__id_key")),
    )
    with op.batch_alter_table("runmetaentry", schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f("ix_runmetaentry_run__id"), ["run__id"], unique=False
        )

    op.create_table(
        "iamc_datapoint_oracle",
        sa.Column("value", sa.Float(), nullable=True),
        sa.Column("type", sa.String(length=255), nullable=False),
        sa.Column("step_category", sa.String(length=1023), nullable=True),
        sa.Column("step_year", sa.Integer(), nullable=True),
        sa.Column("step_datetime", sa.DateTime(), nullable=True),
        sa.Column("time_series__id", sa.Integer(), nullable=False),
        sa.Column(
            "id",
            sa.Integer(),
            sa.Identity(always=False, on_null=True, start=1, increment=1),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["time_series__id"],
            ["iamc_timeseries.id"],
            name=op.f("fk_iamc_datapoint_oracle_time_series__id_iamc_timeseries"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_iamc_datapoint_oracle")),
        sa.UniqueConstraint(
            "time_series__id",
            "step_year",
            "step_category",
            "step_datetime",
            name=op.f(
                "uq_iamc_datapoint_oracle_time_series__id_step_year_step_category_step_datetime"
            ),
        ),
    )
    with op.batch_alter_table("iamc_datapoint_oracle", schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f("ix_iamc_datapoint_oracle_step_category"),
            ["step_category"],
            unique=False,
        )
        batch_op.create_index(
            batch_op.f("ix_iamc_datapoint_oracle_step_datetime"),
            ["step_datetime"],
            unique=False,
        )
        batch_op.create_index(
            batch_op.f("ix_iamc_datapoint_oracle_step_year"),
            ["step_year"],
            unique=False,
        )
        batch_op.create_index(
            batch_op.f("ix_iamc_datapoint_oracle_time_series__id"),
            ["time_series__id"],
            unique=False,
        )
        batch_op.create_index(
            batch_op.f("ix_iamc_datapoint_oracle_type"), ["type"], unique=False
        )

    op.create_table(
        "iamc_datapoint_universal",
        sa.Column("value", sa.Float(), nullable=True),
        sa.Column("type", sa.String(length=255), nullable=False),
        sa.Column("step_category", sa.String(length=1023), nullable=True),
        sa.Column("step_year", sa.Integer(), nullable=True),
        sa.Column("step_datetime", sa.DateTime(), nullable=True),
        sa.Column("time_series__id", sa.Integer(), nullable=False),
        sa.Column(
            "id",
            sa.Integer(),
            sa.Identity(always=False, on_null=True, start=1, increment=1),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["time_series__id"],
            ["iamc_timeseries.id"],
            name=op.f("fk_iamc_datapoint_universal_time_series__id_iamc_timeseries"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_iamc_datapoint_universal")),
        sa.UniqueConstraint(
            "time_series__id",
            "step_datetime",
            name=op.f("uq_iamc_datapoint_universal_time_series__id_step_datetime"),
        ),
        sa.UniqueConstraint(
            "time_series__id",
            "step_year",
            "step_category",
            name=op.f(
                "uq_iamc_datapoint_universal_time_series__id_step_year_step_category"
            ),
        ),
    )
    with op.batch_alter_table("iamc_datapoint_universal", schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f("ix_iamc_datapoint_universal_step_category"),
            ["step_category"],
            unique=False,
        )
        batch_op.create_index(
            batch_op.f("ix_iamc_datapoint_universal_step_datetime"),
            ["step_datetime"],
            unique=False,
        )
        batch_op.create_index(
            batch_op.f("ix_iamc_datapoint_universal_step_year"),
            ["step_year"],
            unique=False,
        )
        batch_op.create_index(
            batch_op.f("ix_iamc_datapoint_universal_time_series__id"),
            ["time_series__id"],
            unique=False,
        )
        batch_op.create_index(
            batch_op.f("ix_iamc_datapoint_universal_type"),
            ["type"],
            unique=False,
        )

    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table("iamc_datapoint_universal", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_iamc_datapoint_universal_type"))
        batch_op.drop_index(batch_op.f("ix_iamc_datapoint_universal_time_series__id"))
        batch_op.drop_index(batch_op.f("ix_iamc_datapoint_universal_step_year"))
        batch_op.drop_index(batch_op.f("ix_iamc_datapoint_universal_step_datetime"))
        batch_op.drop_index(batch_op.f("ix_iamc_datapoint_universal_step_category"))

    op.drop_table("iamc_datapoint_universal")
    with op.batch_alter_table("iamc_datapoint_oracle", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_iamc_datapoint_oracle_type"))
        batch_op.drop_index(batch_op.f("ix_iamc_datapoint_oracle_time_series__id"))
        batch_op.drop_index(batch_op.f("ix_iamc_datapoint_oracle_step_year"))
        batch_op.drop_index(batch_op.f("ix_iamc_datapoint_oracle_step_datetime"))
        batch_op.drop_index(batch_op.f("ix_iamc_datapoint_oracle_step_category"))

    op.drop_table("iamc_datapoint_oracle")
    with op.batch_alter_table("runmetaentry", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_runmetaentry_run__id"))

    op.drop_table("runmetaentry")
    with op.batch_alter_table("iamc_timeseries", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_iamc_timeseries_run__id"))
        batch_op.drop_index(batch_op.f("ix_iamc_timeseries_region__id"))
        batch_op.drop_index(batch_op.f("ix_iamc_timeseries_measurand__id"))

    op.drop_table("iamc_timeseries")
    op.drop_table("unit_docs")
    op.drop_table("scenario_docs")
    with op.batch_alter_table("run", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_run_scenario__id"))
        batch_op.drop_index(batch_op.f("ix_run_model__id"))

    op.drop_table("run")
    op.drop_table("region_docs")
    op.drop_table("model_docs")
    op.drop_table("iamc_variable_docs")
    with op.batch_alter_table("iamc_measurand", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_iamc_measurand_variable__id"))
        batch_op.drop_index(batch_op.f("ix_iamc_measurand_unit__id"))

    op.drop_table("iamc_measurand")
    op.drop_table("unit")
    op.drop_table("scenario")
    op.drop_table("region")
    op.drop_table("model")
    op.drop_table("iamc_variable")
    # ### end Alembic commands ###
