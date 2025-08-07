"""Add assistant_phone to contractors and backfill from phone

Revision ID: 4ac9de123f45
Revises: 3f8a7b9c4d2e
Create Date: 2025-08-07 09:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '4ac9de123f45'
down_revision: Union[str, Sequence[str], None] = '3f8a7b9c4d2e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1) Add as nullable so existing rows aren’t broken
    op.add_column(
        'contractors',
        sa.Column('assistant_phone', sa.String(), nullable=True),
    )
    # 2) Backfill from the existing `phone` field
    op.execute("""
        UPDATE contractors
        SET assistant_phone = phone
    """)
    # 3) Now that every row has a value, make it non-nullable
    op.alter_column(
        'contractors',
        'assistant_phone',
        nullable=False,
    )


def downgrade() -> None:
    # Simply drop the column
    op.drop_column('contractors', 'assistant_phone')
