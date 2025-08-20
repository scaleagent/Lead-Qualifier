
"""Add contractor_profile field to contractors table

Revision ID: 7a8b9c0d1e2f
Revises: 4ac9de123f45
Create Date: 2024-01-01 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '7a8b9c0d1e2f'
down_revision = '9f6f2b9c3b1a'
branch_labels = None
depends_on = None

def upgrade():
    # Add contractor_profile column with default value
    op.add_column('contractors', sa.Column('contractor_profile', sa.String(), nullable=False, server_default='general_contractor'))

def downgrade():
    # Remove contractor_profile column
    op.drop_column('contractors', 'contractor_profile')
