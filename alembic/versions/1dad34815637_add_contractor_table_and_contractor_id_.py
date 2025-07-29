"""Add contractor table and contractor_id FK

Revision ID: 1dad34815637
Revises: 6969ab
Create Date: 2025-07-29 12:41:36.274992

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '1dad34815637'
down_revision: Union[str, Sequence[str], None] = '6969ab'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # 1) Create the contractors table
    op.create_table(
        'contractors',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('phone', sa.String(), nullable=False),
        sa.Column('address', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'), sa.UniqueConstraint('phone'))

    # 2) Add contractor_id columns as NULLABLE
    op.add_column('conversation_data',
                  sa.Column('contractor_id', sa.Integer(), nullable=True))
    op.add_column('conversations',
                  sa.Column('contractor_id', sa.Integer(), nullable=True))

    # 3) Create foreign key constraints
    op.create_foreign_key(None, 'conversation_data', 'contractors',
                          ['contractor_id'], ['id'])
    op.create_foreign_key(None, 'conversations', 'contractors',
                          ['contractor_id'], ['id'])

    # 4) Backfill contractor_id from old contractor_phone
    op.execute("""
        UPDATE conversation_data AS cd
        SET contractor_id = c.id
        FROM contractors AS c
        WHERE cd.contractor_phone = c.phone
    """)
    op.execute("""
        UPDATE conversations AS conv
        SET contractor_id = c.id
        FROM contractors AS c
        WHERE conv.contractor_phone = c.phone
    """)

    # 5) Alter contractor_id to be NOT NULL now that data is backfilled
    op.alter_column('conversation_data', 'contractor_id', nullable=False)
    op.alter_column('conversations', 'contractor_id', nullable=False)

    # 6) Drop the old contractor_phone columns
    op.drop_column('conversation_data', 'contractor_phone')
    op.drop_column('conversations', 'contractor_phone')


def downgrade() -> None:
    """Downgrade schema."""
    # Recreate the old phone columns
    op.add_column(
        'conversations',
        sa.Column('contractor_phone',
                  sa.VARCHAR(),
                  autoincrement=False,
                  nullable=False))
    op.add_column(
        'conversation_data',
        sa.Column('contractor_phone',
                  sa.VARCHAR(),
                  autoincrement=False,
                  nullable=False))

    # Drop foreign keys and new columns
    op.drop_constraint(None, 'conversations', type_='foreignkey')
    op.drop_column('conversations', 'contractor_id')

    op.drop_constraint(None, 'conversation_data', type_='foreignkey')
    op.drop_column('conversation_data', 'contractor_id')

    # Drop the contractors table
    op.drop_table('contractors')
