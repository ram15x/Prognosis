import datetime as dt
import time
from collections import defaultdict

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError

_hasher = PasswordHasher()
ALGO = "HS256"


def hash_password(pw: str) -> str:
    return _hasher.hash(pw)


def verify_password(pw: str, hashed: str) -> bool:
    try:
        return _hasher.verify(hashed, pw)
    except (VerificationError, InvalidHashError):
        return False


def make_token(secret: str, user_id: int, minutes: int) -> str:
    exp = dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=minutes)
    return jwt.encode({"sub": str(user_id), "exp": exp}, secret, algorithm=ALGO)


def read_token(secret: str, token: str) -> int | None:
    try:
        return int(jwt.decode(token, secret, algorithms=[ALGO])["sub"])
    except (jwt.PyJWTError, KeyError, ValueError):
        return None


class LoginThrottle:
    """Lock a username out for a while after repeated failures. In memory, so it resets on restart
    and is per process. Fine for one container, a shared store would be needed behind a load balancer."""

    def __init__(self, max_fails=5, window=900):
        self.max_fails, self.window = max_fails, window
        self._fails = defaultdict(list)

    def _recent(self, name):
        cutoff = time.time() - self.window
        self._fails[name] = [t for t in self._fails[name] if t > cutoff]
        return self._fails[name]

    def blocked(self, name: str) -> bool:
        return len(self._recent(name)) >= self.max_fails

    def fail(self, name: str):
        self._recent(name).append(time.time())

    def clear(self, name: str):
        self._fails.pop(name, None)


# used to burn the same time on a login for a username that does not exist
DUMMY_HASH = _hasher.hash("not-a-real-password")
