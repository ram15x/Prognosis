import httpx


class ApiError(Exception):
    def __init__(self, status, detail):
        self.status, self.detail = status, detail
        super().__init__(f"{status}: {detail}")

    def messages(self):
        d = self.detail
        if isinstance(d, dict) and "problems" in d:
            return d["problems"]
        return [str(d)]


class Api:
    # thin wrapper so the UI never builds urls or headers by hand
    def __init__(self, base_url: str, token: str | None = None, transport=None):
        self.http = httpx.Client(base_url=base_url, timeout=30, transport=transport)
        self.token = token

    def _req(self, method, path, **kw):
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        r = self.http.request(method, path, headers=headers, **kw)
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text)
            except ValueError:
                detail = r.text
            raise ApiError(r.status_code, detail)
        return r

    def login(self, username, password):
        r = self._req("POST", "/auth/login", data={"username": username, "password": password})
        self.token = r.json()["access_token"]

    def me(self): return self._req("GET", "/me").json()
    def models(self): return self._req("GET", "/models").json()
    def model(self, key): return self._req("GET", f"/models/{key}").json()
    def patients(self, q=""): return self._req("GET", "/patients", params={"q": q}).json()
    def add_patient(self, body): return self._req("POST", "/patients", json=body).json()
    def predict(self, pid, body): return self._req("POST", f"/patients/{pid}/predictions", json=body).json()
    def trend(self, pid, cond): return self._req("GET", f"/patients/{pid}/trend/{cond}").json()
    def dashboard(self): return self._req("GET", "/dashboard").json()
    def report(self, prid): return self._req("GET", f"/predictions/{prid}/report.pdf").content
    def audit(self): return self._req("GET", "/audit").json()
    def add_user(self, body): return self._req("POST", "/users", json=body).json()
