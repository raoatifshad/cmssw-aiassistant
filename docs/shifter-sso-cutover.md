# Shift Console: switching from local dev login to real CERN SSO

The `/shift` shifter-only page (shift summary digest + live alerts) is fully
built and tested. Today it runs with `DEV_SHIFTER_LOGIN=true`, a local-only
bypass that skips the real CERN SSO round trip so it can be tested on a
laptop before the CERN-side registration exists. This doc is the checklist
for flipping it over to real CERN SSO once the idea is ready to go live —
nothing else about the feature changes at that point.

See also: `server/shifter_auth.py` (the auth module itself), the
`/api/*` endpoint reference and integration options published as the
"Shift Console API" artifact from this session (ask for the link again if
it's needed, or regenerate via `Artifact({action: "list"})`), and
`.env.example` for the annotated list of every variable below.

## Why this is a config change, not a code change

`require_shifter` (the one function every protected route depends on) always
does the exact same two checks, regardless of how the session got set:

1. Is there a `shifter_username` in the session?
2. Is that username in `SHIFTER_CERN_USERNAMES`?

`DEV_SHIFTER_LOGIN`'s `/shift/dev-login?username=...` route only fakes step
1 (it sets the session directly, no CERN involved) — step 2, the actual
access-control decision, is identical between dev and production. Real CERN
SSO (`/shift/login` → `/shift/auth/callback`) sets that same session key
after actually verifying identity with CERN. So going live is entirely about
provisioning real credentials and turning the bypass off — `shifter_auth.py`
itself needs no edits.

## Checklist

- [ ] **Register this app on CERN's Application Portal**
  (<https://application-portal.web.cern.ch/>). This is the one step that
  cannot be automated or done by an assistant — it needs your own CERN admin
  access. When it asks for a redirect URI, register exactly the value you're
  about to put in `CERN_SSO_REDIRECT_URI` below (e.g.
  `https://<your-production-host>/shift/auth/callback` — not `localhost`
  once this is a real deployment).

- [ ] **Verify `CERN_SSO_DISCOVERY_URL` is still current.** The default in
  `.env.example` (`https://auth.cern.ch/auth/realms/cern/.well-known/openid-configuration`)
  was not independently confirmed against CERN's current SSO documentation
  at the time this was built — check it before relying on it.

- [ ] **Set the real credentials in `.env`** (production's `.env`, not just
  your laptop's):
  ```
  CERN_SSO_CLIENT_ID=<from the Application Portal>
  CERN_SSO_CLIENT_SECRET=<from the Application Portal>
  CERN_SSO_REDIRECT_URI=<the exact URI you registered above>
  ```

- [ ] **Generate a real session-signing secret** — never reuse the
  `dev-only-insecure-key` fallback that `server/api_server.py` uses only so
  a plain `uvicorn` boot doesn't crash when `SESSION_SECRET_KEY` is unset:
  ```
  SESSION_SECRET_KEY=$(openssl rand -hex 32)
  ```

- [ ] **Set the real allowlist**, if it's changed since laptop testing:
  ```
  SHIFTER_CERN_USERNAMES=<cern-username-1>,<cern-username-2>
  ```

- [ ] **Turn the dev bypass off**:
  ```
  DEV_SHIFTER_LOGIN=false
  ```
  Confirm the startup warning (`DEV_SHIFTER_LOGIN=true -- /shift/dev-login
  bypasses CERN SSO entirely`) no longer prints when the service starts.

- [ ] **Confirm the dev route is gone.** `GET /shift/dev-login?username=anything`
  must now return `404` (not a redirect) — if it still logs you in, the env
  var didn't actually take effect on the running process (stale env, wrong
  `.env` file, or the process needs a real restart, not just a reload).

- [ ] **Walk the real login flow yourself first**, before telling the
  shifters it's live: visit `/shift`, confirm it redirects into an actual
  CERN SSO prompt (or straight through, if you're already CERN-authenticated
  elsewhere in the browser), confirm `/api/shift-whoami` reports your real
  CERN username afterward, and confirm someone *not* on the allowlist gets a
  `403`, not a `200`.

- [ ] **Redeploy/restart the service** so every change above is actually
  loaded — `python-dotenv` reads `.env` at process start, not on a live
  reload.

## Rollback

If something's wrong with the real CERN SSO wiring and the page needs to
keep working for a demo in the meantime, set `DEV_SHIFTER_LOGIN=true` again
and restart — this reverts to laptop-testing behavior without touching any
other config. Don't leave it set once the real issue is fixed.
