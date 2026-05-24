"""add min_prompts_confirm to detection_classes

Revision ID: a1b2c3d4e5f6
Revises: 7afa318ef11e
Create Date: 2026-05-24 00:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, Sequence[str], None] = "7afa318ef11e"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "detection_classes",
        sa.Column(
            "min_prompts_confirm",
            sa.Integer(),
            nullable=False,
            server_default="2",
        ),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("detection_classes", "min_prompts_confirm")
