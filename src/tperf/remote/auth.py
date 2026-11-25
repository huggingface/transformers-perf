"""Sign in to the transformers-ci dashboard, for the commands that queue work on the bench host (tperf bench).

Usage: tperf auth login [--no-browser]   # device login: confirm the code in the browser, as a Hugging Face member
       tperf auth status
       tperf auth logout

The access token (eight hours) is kept in ~/.config/tperf/dashboard-auth.json (chmod 600), per dashboard.
TPERF_DASHBOARD (default https://transformers-ci.huggingface.cool) selects the dashboard.
"""

import argparse
import json
import os
import sys
import tempfile
import time
import webbrowser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

DEFAULT_DASHBOARD = "https://transformers-ci.huggingface.cool"


class DashboardError(ValueError):
    def __init__(self, error: str):
        self.error = error
        super().__init__(error)


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise DashboardError("the dashboard redirected an authenticated request; check TPERF_DASHBOARD")


def dashboard() -> str:
    value = os.environ.get("TPERF_DASHBOARD") or DEFAULT_DASHBOARD
    parsed = urlsplit(value)
    if parsed.scheme != "https" or not parsed.hostname or parsed.path not in ("", "/") or parsed.query:
        raise ValueError(f"TPERF_DASHBOARD must be an HTTPS origin, e.g. {DEFAULT_DASHBOARD}")
    return value.rstrip("/")


def request(path: str, *, body: dict | None = None, token: str | None = None) -> dict:
    headers = {"Accept": "application/json", "User-Agent": "tperf"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = "Bearer " + token
    req = Request(dashboard() + path, data=json.dumps(body).encode() if body is not None else None, headers=headers)
    try:
        with build_opener(NoRedirect()).open(req, timeout=20) as response:
            payload = response.read(1_000_001)
            return json.loads(payload) if payload else {}
    except HTTPError as error:
        try:
            reason = json.loads(error.read(4096)).get("error", "request_failed")
        except (ValueError, AttributeError):
            reason = "request_failed"
        # only protocol error names are printed, never a body that could carry a credential
        if not isinstance(reason, str) or not reason.replace("_", "").isalnum():
            reason = "request_failed"
        raise DashboardError(reason) from None
    except (URLError, TimeoutError):
        raise DashboardError("could not reach the dashboard") from None


def credential_path() -> Path:
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "tperf" / "dashboard-auth.json"


def credentials() -> dict:
    path = credential_path()
    if not path.exists():
        return {}
    if path.is_symlink() or path.stat().st_mode & 0o077:
        raise ValueError(f"the credential file must be private (chmod 600): {path}")
    data = json.loads(path.read_text())
    return data if isinstance(data, dict) else {}


def save(data: dict) -> None:
    path = credential_path()
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".dashboard-auth-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(data, stream)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def access_token() -> str:
    record = credentials().get(dashboard(), {})
    if not record or record.get("expires", 0) <= time.time():
        raise ValueError("not signed in to the dashboard (or the session expired): run tperf auth login")
    return record["access_token"]


def login(open_browser: bool) -> None:
    grant = request("/auth/device", body={"client_id": "tperf"})
    verification = grant["verification_uri_complete"]
    if urlsplit(verification).scheme != "https" or urlsplit(verification).netloc != urlsplit(dashboard()).netloc:
        raise ValueError("the dashboard returned an unexpected authorization URL")
    print(f"Open {verification}\nConfirm code: {grant['user_code']}", flush=True)
    if open_browser:
        webbrowser.open(verification)
    interval = max(5, int(grant["interval"]))
    deadline = time.monotonic() + min(int(grant["expires_in"]), 900)
    while time.monotonic() < deadline:
        time.sleep(interval)
        try:
            result = request(
                "/auth/token",
                body={
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                    "device_code": grant["device_code"],
                },
            )
        except DashboardError as error:
            if error.error == "authorization_pending":
                continue
            if error.error == "slow_down":
                interval += 5
                continue
            raise
        token = result["access_token"]
        user = request("/auth/me", token=token)
        data = credentials()
        previous = data.get(dashboard(), {}).get("access_token")
        data[dashboard()] = {"access_token": token, "expires": time.time() + int(result["expires_in"]),
                             "login": user["login"]}  # fmt: skip
        save(data)
        if previous:
            try:
                request("/auth/revoke", body={}, token=previous)
            except DashboardError:
                pass  # expired or already revoked
        print(f"Signed in as @{user['login']} on {dashboard()}. Access expires in eight hours.")
        return
    raise ValueError("the authorization expired: run tperf auth login again")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["login", "status", "logout"])
    parser.add_argument("--no-browser", action="store_true", help="print the URL, do not open a browser")
    args = parser.parse_args(argv)
    try:
        if args.action == "login":
            login(not args.no_browser)
        elif args.action == "status":
            user = request("/auth/me", token=access_token())
            print(json.dumps({"dashboard": dashboard(), "login": user["login"], "scope": user["scope"]}, indent=2))
        else:
            data = credentials()
            if record := data.get(dashboard()):
                try:
                    request("/auth/revoke", body={}, token=record["access_token"])
                except DashboardError as error:
                    if error.error != "unauthorized":
                        raise  # keep the credential so logout can retry
                data.pop(dashboard())
                save(data)
            print(f"Signed out of {dashboard()}.")
    except (ValueError, OSError) as e:
        print(f"tperf auth: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
