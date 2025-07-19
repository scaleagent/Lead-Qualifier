import os, sys
from logging.config import fileConfig
from dotenv import load_dotenv
from sqlalchemy import engine_from_config, pool
from alembic import context

# 0) Ensure our app’s folder is on the path so we can import repos.models
#    (needed if Alembic isn’t already running with your project root on PYTHONPATH)
sys.path.insert(0, os.getcwd())

# 1) Load .env (local) or Heroku env vars
load_dotenv()

# 2) Grab DATABASE_URL and patch legacy scheme
raw_url = os.getenv("DATABASE_URL", "")
if raw_url.startswith("postgres://"):
    fixed_url = raw_url.replace("postgres://", "postgresql://", 1)
else:
    fixed_url = raw_url

# 3) Override the URL in Alembic’s config
config = context.config
config.set_main_option("sqlalchemy.url", fixed_url)

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
