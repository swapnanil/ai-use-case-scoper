import os
from datetime import datetime
from uuid import uuid4

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy import JSON, select
from sqlalchemy.orm import DeclarativeBase, Session, relationship

from agent.models import AIRoadmap, CompanyProfile, PlanOutcome

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./scoper.db")

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})


class Base(DeclarativeBase):
    pass


class CompanyRow(Base):
    __tablename__ = "companies"
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_active = Column(DateTime, default=datetime.utcnow)
    profile_json = Column(JSON)
    sessions = relationship("SessionRow", back_populates="company", cascade="all, delete-orphan")


class SessionRow(Base):
    __tablename__ = "sessions"
    id = Column(String, primary_key=True)
    company_id = Column(String, ForeignKey("companies.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    input_profile_json = Column(JSON)
    roadmap_json = Column(JSON)
    session_type = Column(String, nullable=False)
    checkin_notes = Column(Text)
    parent_session_id = Column(String, ForeignKey("sessions.id"), nullable=True)
    company = relationship("CompanyRow", back_populates="sessions")
    outcomes = relationship("OutcomeRow", back_populates="session", cascade="all, delete-orphan")


class OutcomeRow(Base):
    __tablename__ = "outcomes"
    id = Column(String, primary_key=True)
    session_id = Column(String, ForeignKey("sessions.id"), nullable=False)
    use_case_title = Column(String)
    status = Column(String)
    blocker = Column(Text)
    actual_timeline = Column(String)
    notes = Column(Text)
    recorded_at = Column(DateTime, default=datetime.utcnow)
    session = relationship("SessionRow", back_populates="outcomes")


def init_db() -> None:
    Base.metadata.create_all(engine)


# ---------------------------------------------------------------------------
# Company CRUD
# ---------------------------------------------------------------------------

def create_company(name: str) -> dict:
    company_id = str(uuid4())
    with Session(engine) as session:
        row = CompanyRow(id=company_id, name=name, created_at=datetime.utcnow(), last_active=datetime.utcnow())
        session.add(row)
        session.commit()
    return {"id": company_id, "name": name, "created_at": datetime.utcnow().isoformat()}


def get_company(company_id: str) -> dict | None:
    with Session(engine) as session:
        row = session.get(CompanyRow, company_id)
        if not row:
            return None
        return {
            "id": row.id,
            "name": row.name,
            "created_at": row.created_at.isoformat(),
            "last_active": row.last_active.isoformat() if row.last_active else None,
            "profile": row.profile_json,
        }


# ---------------------------------------------------------------------------
# Session CRUD
# ---------------------------------------------------------------------------

def save_session(
    company_id: str,
    profile: CompanyProfile,
    roadmap: AIRoadmap,
    session_type: str,
    checkin_notes: str | None = None,
    parent_session_id: str | None = None,
) -> str:
    session_id = str(uuid4())
    with Session(engine) as db:
        row = SessionRow(
            id=session_id,
            company_id=company_id,
            created_at=datetime.utcnow(),
            input_profile_json=profile.model_dump(mode="json"),
            roadmap_json=roadmap.model_dump(mode="json"),
            session_type=session_type,
            checkin_notes=checkin_notes,
            parent_session_id=parent_session_id,
        )
        company = db.get(CompanyRow, company_id)
        if company:
            company.last_active = datetime.utcnow()
            company.profile_json = profile.model_dump(mode="json")
        db.add(row)
        db.commit()
    return session_id


def get_session(session_id: str) -> dict | None:
    with Session(engine) as db:
        row = db.get(SessionRow, session_id)
        if not row:
            return None
        return {
            "id": row.id,
            "company_id": row.company_id,
            "created_at": row.created_at.isoformat(),
            "input_profile": row.input_profile_json,
            "roadmap": row.roadmap_json,
            "session_type": row.session_type,
            "checkin_notes": row.checkin_notes,
            "parent_session_id": row.parent_session_id,
        }


def get_company_sessions(company_id: str) -> list[dict]:
    with Session(engine) as db:
        rows = db.execute(
            select(SessionRow)
            .where(SessionRow.company_id == company_id)
            .order_by(SessionRow.created_at.desc())
        ).scalars().all()
        return [
            {
                "id": r.id,
                "created_at": r.created_at.isoformat(),
                "session_type": r.session_type,
                "parent_session_id": r.parent_session_id,
                "readiness_score": (r.roadmap_json or {}).get("readiness_score"),
                "recommended_first_project": (r.roadmap_json or {}).get("recommended_first_project"),
            }
            for r in rows
        ]


# ---------------------------------------------------------------------------
# Outcome recording
# ---------------------------------------------------------------------------

def save_outcome(outcome: PlanOutcome) -> None:
    with Session(engine) as db:
        row = OutcomeRow(
            id=str(uuid4()),
            session_id=outcome.session_id,
            use_case_title=outcome.use_case_title,
            status=outcome.status,
            blocker=outcome.blocker,
            actual_timeline=outcome.actual_timeline,
            notes=outcome.notes,
            recorded_at=datetime.utcnow(),
        )
        db.add(row)
        db.commit()


# ---------------------------------------------------------------------------
# Memory context for prompt injection
# ---------------------------------------------------------------------------

def build_memory_context(company_id: str) -> str:
    sessions = get_company_sessions(company_id)
    if not sessions:
        return ""

    latest = sessions[0]
    lines = [
        "COMPANY MEMORY (from previous scoping sessions):",
        f"- Sessions on record: {len(sessions)}",
        f"- Previous readiness score: {latest.get('readiness_score', 'N/A')} "
        f"(session {latest['created_at'][:10]})",
        f"- Previous recommended first project: {latest.get('recommended_first_project', 'N/A')}",
        f"- Last session type: {latest.get('session_type', 'N/A')}",
    ]
    return "\n".join(lines)
