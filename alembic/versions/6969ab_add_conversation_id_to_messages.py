"""add conversation_id to messages

Revision ID: 1234567890ab
Revises: <previous_rev>
Create Date: 2025-07-24 15:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '6969ab'
down_revision = '<previous_rev>'
branch_labels = None
depends_on = None


def upgrade():
    # 1) Add the nullable FK column
    op.add_column('messages',
                  sa.Column('conversation_id', sa.String(), nullable=True))
    # 2) Create the FK constraint
    op.create_foreign_key('fk_messages_conversation',
                          'messages',
                          'conversations', ['conversation_id'], ['id'],
                          ondelete='CASCADE')


def downgrade():
    # reverse the above
    op.drop_constraint('fk_messages_conversation',
                       'messages',
                       type_='foreignkey')
    op.drop_column('messages', 'conversation_id')
