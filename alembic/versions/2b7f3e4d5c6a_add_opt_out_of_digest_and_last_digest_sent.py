"""Add opt_out_of_digest and last_digest_sent to conversation_data

Revision ID: 2b7f3e4d5c6a
Revises: 1dad34815637
Create Date: 2025-08-06 15:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '2b7f3e4d5c6a'
down_revision: Union[str, Sequence[str], None] = '1dad34815637'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema: add opt_out_of_digest and last_digest_sent"""
    op.add_column(
        'conversation_data',
        sa.Column('opt_out_of_digest',
                  sa.Boolean(),
                  nullable=False,
                  server_default=sa.text('false')))
    op.add_column('conversation_data',
                  sa.Column('last_digest_sent', sa.DateTime(), nullable=True))


def downgrade() -> None:
    """Downgrade schema: remove opt_out_of_digest and last_digest_sent"""
    op.drop_column('conversation_data', 'last_digest_sent')
    op.drop_column('conversation_data', 'opt_out_of_digest')
