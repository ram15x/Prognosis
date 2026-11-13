import argparse
import getpass

from sqlalchemy import select

from .config import get_settings
from .db.models import Clinic, User
from .db.session import make_engine, make_session_factory
from .services import security


def create_admin(a):
    s = get_settings()
    factory = make_session_factory(make_engine(s.db_url))
    pw = a.password or getpass.getpass("password (min 10 chars): ")
    if len(pw) < 10:
        raise SystemExit("password too short")
    with factory() as db:
        clinic = db.scalar(select(Clinic).where(Clinic.name == a.clinic))
        if not clinic:
            clinic = Clinic(name=a.clinic)
            db.add(clinic)
            db.flush()
        db.add(User(username=a.username, password_hash=security.hash_password(pw), role="admin", clinic_id=clinic.id))
        db.commit()
    print(f"admin '{a.username}' created in clinic '{a.clinic}'")


def main():
    ap = argparse.ArgumentParser(prog="prognosis")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create-admin")
    c.add_argument("--username", required=True)
    c.add_argument("--clinic", default="Main Clinic")
    c.add_argument("--password", help="leave out to be prompted, passing it here ends up in shell history")
    c.set_defaults(fn=create_admin)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
