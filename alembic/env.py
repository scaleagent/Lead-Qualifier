import os
import sys
from logging.config import fileConfig
from dotenv import load_dotenv

from sqlalchemy import engine_from_config, pool
from alembic import context

# 1) Load environment variables from .env (locally) or Heroku’s env
load_dotenv()

# 2) Ensure your project root is on Python path so we can import your models
sys.path.insert(0, os.getcwd())

# 3) Tell Alembic where to find its own config and override the DB URL
config = context.config
config.set_main_option("sqlalchemy.url", os.getenv("DATABASE_URL"))

# 4) Set up logging per alembic.ini
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# 5) Point Alembic at your SQLAlchemy models’ metadata
from repos.models import Base

target_metadata = Base.metadata


def run_migrations_offline():
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection,
                          target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
