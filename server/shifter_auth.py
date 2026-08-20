"""CERN SSO (OIDC) login gate for the shifter-only /shift page.

Standalone auth module, kept separate from api_server.py's already-large
scope -- matches how request_planner.py/agent.py/llm_client.py are already
split out. Fails closed: if CERN SSO isn't configured, or a signed-in
account isn't on the allowlist, every route this protects refuses access
rather than silently exposing the page. The allowlist is re-checked on
every request (not just at login), so removing a name takes effect on that
person's very next request instead of only at their next login.

Requires CERN_SSO_CLIENT_ID/CERN_SSO_CLIENT_SECRET from CERN's Application
Portal (https://application-portal.web.cern.ch/) -- register this app there
first; the redirect URI registered must exactly match CERN_SSO_REDIRECT_URI.
CERN_SSO_DISCOVERY_URL is CERN's standard OIDC discovery document; verify
it's still current in CERN's SSO docs before relying on it in production, as
exact endpoint paths can change over time.
"""
import os

from authlib.integrations.starlette_client import OAuth
from fastapi import APIRouter, HTTPException, Request
from starlette.responses import RedirectResponse

CERN_SSO_DISCOVERY_URL = os.environ.get(
    "CERN_SSO_DISCOVERY_URL",
    "https://auth.cern.ch/auth/realms/cern/.well-known/openid-configuration",
)
CERN_SSO_CLIENT_ID = os.environ.get("CERN_SSO_CLIENT_ID")
CERN_SSO_CLIENT_SECRET = os.environ.get("CERN_SSO_CLIENT_SECRET")
CERN_SSO_REDIRECT_URI = os.environ.get("CERN_SSO_REDIRECT_URI")
SSO_CONFIGURED = bool(CERN_SSO_CLIENT_ID and CERN_SSO_CLIENT_SECRET and CERN_SSO_REDIRECT_URI)

# Local-only stand-in for the real CERN SSO flow, for testing this page on a laptop before
# the Application Portal registration exists. Off by default; must be turned on explicitly.
# It does NOT bypass the allowlist check -- it only fakes "proving who you are", so
# SHIFTER_CERN_USERNAMES still has to list the test username, exactly like production. That
# means require_shifter's logic is never actually different between dev and real CERN SSO;
# swapping this out for the real flow later is a config change (register the app, unset this
# var), never a code change.
DEV_SHIFTER_LOGIN = os.environ.get("DEV_SHIFTER_LOGIN", "false").strip().lower() == "true"
if DEV_SHIFTER_LOGIN:
    print(
        "WARNING: DEV_SHIFTER_LOGIN=true -- /shift/dev-login bypasses CERN SSO entirely. "
        "This must never be set in a real deployment."
    )


def shifter_allowlist() -> frozenset[str]:
    """Comma-separated CERN usernames in SHIFTER_CERN_USERNAMES, re-read on
    every call (not cached) so an env change takes effect on process env
    reload without a code change -- matches this project's other env-driven
    config (CONCEPT_TTL_DAYS etc.) in spirit, though those are read once at
    import time; this one deliberately isn't, since the allowlist is exactly
    the kind of thing expected to change between deploys of the same process
    less often than restarts, but access-control lists are worth re-reading
    defensively."""
    raw = os.environ.get("SHIFTER_CERN_USERNAMES", "")
    return frozenset(name.strip().lower() for name in raw.split(",") if name.strip())


oauth = OAuth()
if SSO_CONFIGURED:
    oauth.register(
        name="cern",
        server_metadata_url=CERN_SSO_DISCOVERY_URL,
        client_id=CERN_SSO_CLIENT_ID,
        client_secret=CERN_SSO_CLIENT_SECRET,
        client_kwargs={"scope": "openid profile email"},
    )

router = APIRouter()


@router.get("/shift/login", include_in_schema=False)
async def shift_login(request: Request):
    if not SSO_CONFIGURED:
        raise HTTPException(503, "CERN SSO is not configured on this deployment (missing client credentials).")
    return await oauth.cern.authorize_redirect(request, CERN_SSO_REDIRECT_URI)


@router.get("/shift/auth/callback", include_in_schema=False)
async def shift_auth_callback(request: Request):
    if not SSO_CONFIGURED:
        raise HTTPException(503, "CERN SSO is not configured on this deployment (missing client credentials).")
    token = await oauth.cern.authorize_access_token(request)
    userinfo = token.get("userinfo") or await oauth.cern.userinfo(token=token)
    username = (userinfo.get("preferred_username") or userinfo.get("cern_upn") or userinfo.get("sub") or "").lower()
    if not username or username not in shifter_allowlist():
        request.session.pop("shifter_username", None)
        raise HTTPException(403, "This CERN account is not on the shifter allowlist.")
    request.session["shifter_username"] = username
    return RedirectResponse(url="/shift")


@router.get("/shift/dev-login", include_in_schema=False)
async def shift_dev_login(request: Request, username: str):
    """Local-testing-only login: sets the session directly, no CERN SSO round
    trip. Returns 404 (pretend the route doesn't exist) unless
    DEV_SHIFTER_LOGIN=true, so it's not even discoverable in a real
    deployment that forgot to unset it. Still subject to the allowlist --
    require_shifter re-checks it on the very next request either way."""
    if not DEV_SHIFTER_LOGIN:
        raise HTTPException(404)
    request.session["shifter_username"] = username.strip().lower()
    return RedirectResponse(url="/shift")


@router.get("/shift/logout", include_in_schema=False)
async def shift_logout(request: Request):
    request.session.pop("shifter_username", None)
    return RedirectResponse(url="/shift/login")


def require_shifter(request: Request) -> str:
    """FastAPI dependency for every protected shifter route. 503 if there's
    no session yet AND no way to start one (neither real SSO nor the dev
    bypass is configured) -- this is the only place that combination is
    checked, so /api/shift-whoami and friends can report it directly instead
    of only surfacing on /shift/login, a full-page navigation target the
    frontend's fetch-based logic can't otherwise inspect. 401 if not signed
    in but a login path exists; 403 if signed in but not (or no longer) on
    the allowlist."""
    username = request.session.get("shifter_username")
    if not username:
        if not SSO_CONFIGURED and not DEV_SHIFTER_LOGIN:
            raise HTTPException(503, "CERN SSO is not configured on this deployment (missing client credentials).")
        raise HTTPException(401, "Not signed in. Visit /shift/login first.")
    if username not in shifter_allowlist():
        raise HTTPException(403, "This CERN account is not on the shifter allowlist.")
    return username
