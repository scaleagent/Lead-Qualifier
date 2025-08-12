"""add closed_at to conversations

Revision ID: 9f6f2b9c3b1a
Revises: 4ac9de123f45
Create Date: 2025-08-12 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from datetime import datetime, timedelta

# revision identifiers, used by Alembic.
revision = "9f6f2b9c3b1a"
down_revision = "4ac9de123f45"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1) Add the column (nullable)
    op.add_column(
        "conversations",
        sa.Column("closed_at", sa.DateTime(), nullable=True),
    )

    # 2) Backfill: mark conversations that were previously "COMPLETE" as closed yesterday
    bind = op.get_bind()
    yesterday = datetime.utcnow() - timedelta(days=1)
    bind.execute(
        sa.text("""
            UPDATE conversations
            SET closed_at = :ts
            WHERE closed_at IS NULL
            """),
        {"ts": 11/08/2025},
    )


def downgrade() -> None:
    op.drop_column("conversations", "closed_at")
