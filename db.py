"""
Database layer for CalorAI Logging Agent.

Design decisions:
- SQLite chosen for zero-setup local persistence (no server needed).
- Meals are never hard-deleted on correction/deletion — instead marked
  `is_active=False`. This preserves an audit trail ("actually that was
  3 rotis not 2" becomes: old row deactivated, new row inserted) and
  keeps daily-totals math simple: SUM over is_active=True rows only.
- user_memory is a simple key-value table, not a vector store. For a
  handful of durable facts per user (diet preference, "usual" meals,
  targets), a flat table is easier to reason about, cheaper to query,
  and just as effective as embeddings at this scale.
"""

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Boolean,
    DateTime, ForeignKey, Text
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime, timezone

DATABASE_URL = "sqlite:///calorai.db"

engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class User(Base):
    """Minimal user record. Supports multi-user / session isolation."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    external_id = Column(String, unique=True, nullable=False)  # e.g. phone number / CLI session name
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    meals = relationship("Meal", back_populates="user")
    memories = relationship("UserMemory", back_populates="user")


class Meal(Base):
    """
    A single logged meal (or meal correction).

    is_active=False rows are corrections/deletions that have been
    superseded — kept for history, excluded from totals.
    """
    __tablename__ = "meals"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    raw_text = Column(Text, nullable=True)        # original user message, if text
    description = Column(Text, nullable=False)    # normalized description, e.g. "2 rotis, chai"
    source = Column(String, default="text")        # "text" | "image"

    calories = Column(Float, default=0.0)
    protein_g = Column(Float, default=0.0)
    carbs_g = Column(Float, default=0.0)
    fat_g = Column(Float, default=0.0)

    is_active = Column(Boolean, default=True)      # False = corrected/deleted, excluded from totals
    supersedes_id = Column(Integer, ForeignKey("meals.id"), nullable=True)  # link correction -> original

    logged_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))  # when the meal was eaten (best guess)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))  # when the row was written

    user = relationship("User", back_populates="meals")


class UserMemory(Base):
    """
    Durable facts worth remembering across sessions.
    e.g. key="diet_preference", value="vegetarian"
         key="usual_breakfast", value="2 parathas and chai"
         key="protein_target_g", value="140"
    """
    __tablename__ = "user_memory"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    key = Column(String, nullable=False)
    value = Column(Text, nullable=False)

    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    user = relationship("User", back_populates="memories")


def init_db():
    """Create all tables if they don't exist yet."""
    Base.metadata.create_all(engine)


def get_session():
    """Return a new DB session. Caller is responsible for closing it."""
    return SessionLocal()


def get_or_create_user(session, external_id: str) -> User:
    """Fetch a user by external_id (e.g. CLI name / phone number), or create one."""
    user = session.query(User).filter_by(external_id=external_id).first()
    if user is None:
        user = User(external_id=external_id)
        session.add(user)
        session.commit()
        session.refresh(user)
    return user


if __name__ == "__main__":
    # Quick manual check: run `python db.py` to create the DB file and tables.
    init_db()
    print("✅ Database initialized -> calorai.db")

    session = get_session()
    test_user = get_or_create_user(session, "test_user_cli")
    print(f"✅ Test user ready: id={test_user.id}, external_id={test_user.external_id}")
    session.close()