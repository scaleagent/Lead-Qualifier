"""Add digest_config JSON column to contractors

Revision ID: 3f8a7b9c4d2e
Revises: 2b7f3e4d5c6a
Create Date: 2025-08-06 16:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '3f8a7b9c4d2e'
down_revision: Union[str, Sequence[str], None] = '2b7f3e4d5c6a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema: add digest_config to contractors"""
    op.add_column(
        'contractors',
        sa.Column('digest_config',
                  sa.JSON(),
                  nullable=False,
                  server_default=sa.text("'{}'::json")))


def downgrade() -> None:
    """Downgrade schema: remove digest_config"""
    op.drop_column('contractors', 'digest_config')
