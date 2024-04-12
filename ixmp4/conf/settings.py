import logging
import logging.config
import os
from pathlib import Path
from typing import Literal

from httpx import ConnectError
from pydantic import Field, HttpUrl, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ixmp4.core.exceptions import InvalidCredentials

from .auth import AnonymousAuth, ManagerAuth
from .credentials import Credentials
from .manager import ManagerConfig
from .toml import TomlConfig
from .user import local_user

logger = logging.getLogger(__name__)

here = Path(__file__).parent


class Settings(BaseSettings):
    mode: Literal["production"] | Literal["development"] | Literal[
        "debug"
    ] = "production"
    storage_directory: Path = Field(
        "~/.local/share/ixmp4/", json_schema_extra={"env": "ixmp4_dir"}
    )
    secret_hs256: str = "default_secret_hs256"
    migration_db_uri: str = "sqlite:///./run/db.sqlite"
    manager_url: HttpUrl = Field("https://api.manager.ece.iiasa.ac.at/v1")
    managed: bool = True
    max_page_size: int = 10_000
    default_page_size: int = 5_000
    client_default_upload_chunk_size: int = 10_000
    client_max_concurrent_requests: int = Field(2, lt=4)
    client_max_request_retries: int = Field(3)
    client_backoff_factor: int = Field(5)
    client_timeout: int = Field(30)
    model_config = SettingsConfigDict(env_prefix="ixmp4_", extra="allow")

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.storage_directory.mkdir(parents=True, exist_ok=True)

        self.database_dir = self.storage_directory / "databases"
        self.database_dir.mkdir(exist_ok=True)

        self.log_dir = self.storage_directory / "log"
        self.log_dir.mkdir(exist_ok=True)

        self.configure_logging(self.mode)

        self._credentials = None
        self._toml = None
        self._default_auth = None
        self._manager = None

        logger.debug(f"Settings loaded: {self}")

    @property
    def credentials(self):
        if self._credentials is None:
            self.load_credentials()
        return self._credentials

    @property
    def default_credentials(self):
        try:
            return self.credentials.get("default")
        except KeyError:
            pass

    @property
    def toml(self):
        if self._toml is None:
            self.load_toml_config()
        return self._toml

    @property
    def default_auth(self):
        if self._default_auth is None:
            self.get_auth()
        return self._default_auth

    @property
    def manager(self):
        if self._manager is None:
            self.load_manager_config()
        return self._manager

    def load_credentials(self):
        credentials_config = self.storage_directory / "credentials.toml"
        credentials_config.touch()
        self._credentials = Credentials(credentials_config)

    def get_auth(self):
        if self.default_credentials is not None:
            try:
                self._default_auth = ManagerAuth(
                    *self.default_credentials, str(self.manager_url)
                )
                logger.info(
                    f"Connecting as user '{self._default_auth.get_user().username}'."
                )
            except InvalidCredentials:
                logger.warning(f"Invalid credentials for {self.manager_url}.")
            except ConnectError:
                logger.warning(f"Unable to connect to {self.manager_url}.")

        else:
            self._default_auth = AnonymousAuth()

    def load_manager_config(self):
        self._manager = ManagerConfig(
            str(self.manager_url), self.default_auth, remote=True
        )

    def load_toml_config(self):
        if self.default_auth is not None:
            toml_user = self.default_auth.get_user()
            if not toml_user.is_authenticated:
                toml_user = local_user
        else:  # if no connection to manager
            toml_user = local_user

        toml_config = self.storage_directory / "platforms.toml"
        toml_config.touch()
        self._toml = TomlConfig(toml_config, toml_user)

    @field_validator("storage_directory")
    def expand_user(cls, v):
        # translate ~/asdf into /home/user/asdf
        return Path.expanduser(v)

    def configure_logging(self, config: str):
        access_file = self.log_dir / "access.log"
        debug_file = self.log_dir / "debug.log"
        error_file = self.log_dir / "error.log"
        os.environ.setdefault("IXMP4_ACCESS_LOG", str(access_file.absolute()))
        os.environ.setdefault("IXMP4_DEBUG_LOG", str(debug_file.absolute()))
        os.environ.setdefault("IXMP4_ERROR_LOG", str(error_file.absolute()))

        logging_config = here / f"logging/{config}.conf"
        logging.config.fileConfig(logging_config, disable_existing_loggers=False)

    def check_credentials(self):
        if self.default_credentials is not None:
            username, password = self.default_credentials
            ManagerAuth(username, password, str(self.manager_url))
