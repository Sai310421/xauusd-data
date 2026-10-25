"""Publish CI evidence with authenticated preflight, deduplication and artifact fallback.

Read preflight does not prove write permission. Never switch credentials after a
403. GitHub App installation authentication is not configured by this module.
"""
import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path("results/amos-abc")
FILES = ("run-manifest.json", "rawtick-kpi.json", "rawtick-input-quality.json",
         "native-gate-summary.json", "native-matching-A01.json",
         "native-matching-B01.json", "native-matching-C01.json")
TOKEN_NAMES = ("GH_RESULTS_TOKEN", "GITHUB_TOKEN", "GH_TOKEN")
REPO = "Sai310421/xauusd-data"
PR = 30


class PublishError(Exception):
    def __init__(self, category, **details):
        self.details = {"category": category, **details}


def select_token(env):
    explicit = env.get("EVIDENCE_TOKEN_ENV")
    if explicit and explicit not in TOKEN_NAMES:
        raise PublishError("INVALID_AUTH_SOURCE")
    name = explicit or next((n for n in TOKEN_NAMES if env.get(n)), None)
    if not name or not env.get(name):
        raise PublishError("MISSING_TOKEN", token_source=name)
    return name, env[name]


def safe(value, env):
    """Remove all candidate credential values, including unselected ones."""
    if isinstance(value, dict):
        return {safe(k, env): safe(v, env) for k, v in value.items()}
    if isinstance(value, list):
        return [safe(v, env) for v in value]
    if isinstance(value, str):
        for name in TOKEN_NAMES:
            if env.get(name):
                value = value.replace(env[name], "[REDACTED]")
        return re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", value)
    return value


def request(token, path, method="GET", data=None):
    req = urllib.request.Request(
        "https://api.github.com" + path, method=method,
        data=None if data is None else json.dumps(data).encode(),
        headers={"Authorization": "Bearer " + token,
                 "Accept": "application/vnd.github+json",
                 "X-GitHub-Api-Version": "2022-11-28",
                 "Content-Type": "application/json",
                 "User-Agent": "AMOS-CircleCI-Evidence"})
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        try:
            message = json.loads(exc.read(4096)).get("message", "")
        except (ValueError, AttributeError):
            message = ""
        headers = exc.headers or {}
        category = {401: "AUTHENTICATION_FAILED", 403: "ACCESS_FORBIDDEN",
                    404: "RESOURCE_NOT_FOUND_OR_HIDDEN", 422: "VALIDATION_OR_SPAM"}.get(
                        exc.code, "GITHUB_HTTP_ERROR")
        if exc.code in (403, 429) and (headers.get("X-RateLimit-Remaining") == "0"
                                     or headers.get("Retry-After")
                                     or "rate limit" in str(message).lower()):
            category = "RATE_LIMITED"
        known = ("Resource not accessible by personal access token",
                 "Resource not accessible by integration", "Bad credentials",
                 "Validation Failed", "Not Found")
        # Never log arbitrary response bodies, exception text or all headers.
        raise PublishError(
            category, http_status=exc.code, method=method, endpoint=path,
            reason=message if message in known else "UNCLASSIFIED",
            accepted_permissions=str(headers.get("X-Accepted-GitHub-Permissions", "not supplied")),
            request_id=str(headers.get("X-GitHub-Request-Id", "not supplied")),
            sso_required=str(headers.get("X-GitHub-SSO", "")).startswith("required"),
            retry_after=str(headers.get("Retry-After", ""))) from None
    except (urllib.error.URLError, TimeoutError, OSError):
        raise PublishError("NETWORK_ERROR", method=method, endpoint=path) from None
    except ValueError:
        raise PublishError("INVALID_API_RESPONSE", method=method, endpoint=path) from None


def preflight(token):
    identity = request(token, "/user")
    request(token, f"/repos/{REPO}")
    target = request(token, f"/repos/{REPO}/issues/{PR}")
    if not identity.get("id") or "pull_request" not in target:
        raise PublishError("INVALID_TARGET_OR_IDENTITY")
    return identity["id"]


def collect(root, env):
    evidence, problems = {}, {}
    for name in FILES:
        try:
            data = json.loads((root / name).read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError()
            evidence[name] = data
        except FileNotFoundError:
            problems[name] = "MISSING"
        except (ValueError, UnicodeError, OSError):
            problems[name] = "INVALID_JSON_OR_UNREADABLE"
    return safe({"schema_version": 2,
                 "commit_sha": env.get("CIRCLE_SHA1", "unknown"),
                 "circleci_job": env.get("CIRCLE_BUILD_URL", "unknown"),
                 "circleci_workflow": env.get("CIRCLE_WORKFLOW_ID", "unknown"),
                 "circleci_build_number": env.get("CIRCLE_BUILD_NUM", "unknown"),
                 "bt_status": env.get("EVIDENCE_BT_STATUS", "UNKNOWN"),
                 "evidence_complete": not problems,
                 "input_problems": problems, "evidence": evidence}, env)


def publish(token, author_id, body, marker):
    # Fail closed if the bounded scan cannot establish absence of a duplicate.
    for page in range(1, 101):
        comments = request(token, f"/repos/{REPO}/issues/{PR}/comments?per_page=100&page={page}")
        if not isinstance(comments, list):
            raise PublishError("INVALID_API_RESPONSE")
        for comment in comments:
            if marker in comment.get("body", "") and comment.get("user", {}).get("id") == author_id:
                if comment["body"] == body:
                    return "UNCHANGED", comment
                return "UPDATED", request(
                    token, f"/repos/{REPO}/issues/comments/{int(comment['id'])}",
                    "PATCH", {"body": body})
        if len(comments) < 100:
            return "CREATED", request(token, f"/repos/{REPO}/issues/{PR}/comments",
                                      "POST", {"body": body})
    raise PublishError("COMMENT_SCAN_LIMIT")


def main(argv=None, *, root=ROOT, env=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    options = parser.parse_args(argv)
    env = os.environ if env is None else env
    root.mkdir(parents=True, exist_ok=True)
    status = {"publication": "NOT_ATTEMPTED", "write_access": "UNVERIFIED",
              "auth_selection": "explicit" if env.get("EVIDENCE_TOKEN_ENV") else "priority",
              "available_sources": [n for n in TOKEN_NAMES if env.get(n)]}

    def save(name, data):
        (root / name).write_text(json.dumps(safe(data, env), indent=2, ensure_ascii=True) + "\n",
                                 encoding="utf-8")

    payload = None if options.preflight_only else collect(root, env)
    if payload is not None:
        # Save before auth: missing credentials must not discard the results.
        save("evidence-bundle.json", payload)
        status.update({k: payload[k] for k in
                       ("commit_sha", "circleci_job", "bt_status", "evidence_complete")})
    code = 1
    try:
        name, token = select_token(env)
        status["token_source"] = name
        print("EVIDENCE_AUTH_SOURCE: " + name, flush=True)
        author_id = preflight(token)
        status["read_preflight"] = "PASS"
        if options.preflight_only:
            code = 0
        else:
            sha = env.get("CIRCLE_SHA1", "")
            if not re.fullmatch(r"[0-9a-fA-F]{40}", sha):
                raise PublishError("INVALID_COMMIT_SHA")
            marker = f"<!-- AMOS_NATIVE_EVIDENCE_V2:{sha} -->"
            body = (marker + "\n### AMOS Raw Tick CI evidence\n"
                    "BT and publication are independent.\n\n```json\n"
                    + json.dumps(payload, indent=2, ensure_ascii=True) + "\n```")
            save("evidence-comment.json", {"body": body})
            if len(body) > 60000:
                raise PublishError("PAYLOAD_TOO_LARGE")
            action, result = publish(token, author_id, body, marker)
            url = result.get("html_url", "")
            if not re.fullmatch(
                r"https://github\.com/Sai310421/xauusd-data/(?:issues|pull)/30#issuecomment-\d+", url
            ):
                raise PublishError("INVALID_COMMENT_RESPONSE")
            stored = request(token, f"/repos/{REPO}/issues/comments/{int(result['id'])}")
            if stored.get("body") != body:
                raise PublishError("COMMENT_VERIFICATION_FAILED")
            status.update(publication=action, comment_url=url, write_access="VERIFIED")
            code = 0 if payload["evidence_complete"] else 2
    except PublishError as exc:
        status.update(publication="BLOCKED" if options.preflight_only else "FAILED", **exc.details)
    except (KeyError, TypeError, ValueError, AttributeError):
        status.update(publication="FAILED", category="INVALID_API_RESPONSE")
    status["exit_code"] = code
    # Redact before truncating: truncation must not expose part of a credential.
    status = safe(status, env)
    for key in ("accepted_permissions", "request_id", "retry_after"):
        if key in status:
            status[key] = status[key][:200]
    filename = "evidence-preflight.json" if options.preflight_only else "evidence-publish-status.json"
    save(filename, status)
    print("EVIDENCE_STATUS: " + json.dumps(safe(status, env), ensure_ascii=True), flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
