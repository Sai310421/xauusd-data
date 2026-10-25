"""Offline contract tests. Fixtures are simulated evidence, never measured BT KPIs."""
import contextlib
import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

import publish_evidence as p


class Response(io.BytesIO):
    def __init__(self, value):
        super().__init__(json.dumps(value).encode())


class PublisherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in p.FILES:
            (self.root / name).write_text('{"fixture": true}', encoding="utf-8")
        self.env = {"GH_RESULTS_TOKEN": "secret-primary-value", "GH_TOKEN": "secret-other-value",
                    "CIRCLE_SHA1": "a" * 40, "CIRCLE_BUILD_URL": "https://app.circleci.com/jobs/123",
                    "EVIDENCE_BT_STATUS": "PASS"}
        self.comments = []
        self.calls = []
        self.failure = None

    def server(self, req, **kwargs):
        path = req.full_url.removeprefix("https://api.github.com")
        method = req.get_method()
        self.calls.append((method, path))
        self.assertEqual(req.get_header("Authorization"), "Bearer " + self.env["GH_RESULTS_TOKEN"])
        if self.failure and self.failure[0](method, path):
            code, headers, message = self.failure[1:]
            raise urllib.error.HTTPError(req.full_url, code, "error", headers,
                                         io.BytesIO(json.dumps({"message": message}).encode()))
        if path == "/user":
            return Response({"id": 7})
        if path == f"/repos/{p.REPO}":
            return Response({"full_name": p.REPO})
        if path == f"/repos/{p.REPO}/issues/30":
            return Response({"pull_request": {}})
        if "?per_page=" in path:
            page = int(path.rsplit("=", 1)[1])
            return Response(self.comments[(page-1)*100:page*100])
        if method in ("POST", "PATCH"):
            body = json.loads(req.data)["body"]
            if method == "POST":
                comment = {"id": len(self.comments) + 1, "user": {"id": 7}}
                self.comments.append(comment)
            else:
                comment = next(c for c in self.comments if c["id"] == int(path.rsplit("/", 1)[1]))
            comment.update(body=body, html_url=f"https://github.com/{p.REPO}/pull/30#issuecomment-{comment['id']}")
            return Response(comment)
        return Response(next(c for c in self.comments if c["id"] == int(path.rsplit("/", 1)[1])))

    def run_main(self, args=None):
        out = io.StringIO()
        with patch.object(p.urllib.request, "urlopen", side_effect=self.server), contextlib.redirect_stdout(out):
            code = p.main(args or [], root=self.root, env=self.env)
        self.output = out.getvalue()
        for name in p.TOKEN_NAMES:
            if self.env.get(name):
                self.assertNotIn(self.env[name], self.output)
                for path in self.root.glob("evidence-*.json"):
                    self.assertNotIn(self.env[name], path.read_text(encoding="utf-8"))
        filename = "evidence-preflight.json" if args else "evidence-publish-status.json"
        return code, json.loads((self.root / filename).read_text())

    def test_missing_token_saves_fallback_without_api(self):
        self.env = {"CIRCLE_SHA1": "a" * 40}
        code, status = self.run_main()
        self.assertEqual((code, status["category"]), (1, "MISSING_TOKEN"))
        self.assertTrue((self.root / "evidence-bundle.json").is_file())
        self.assertFalse(self.calls)

    def test_explicit_source_and_precedence(self):
        self.assertEqual(p.select_token(self.env)[0], "GH_RESULTS_TOKEN")
        self.env["GITHUB_TOKEN"] = "another-secret"
        self.env["EVIDENCE_TOKEN_ENV"] = "GITHUB_TOKEN"
        self.assertEqual(p.select_token(self.env)[0], "GITHUB_TOKEN")

    def test_explicit_missing_source_never_falls_back(self):
        self.env["EVIDENCE_TOKEN_ENV"] = "GITHUB_TOKEN"
        self.assertEqual(self.run_main()[1]["category"], "MISSING_TOKEN")
        self.assertFalse(self.calls)

    def test_invalid_source(self):
        self.env["EVIDENCE_TOKEN_ENV"] = "PATH"
        self.assertEqual(self.run_main()[1]["category"], "INVALID_AUTH_SOURCE")

    def test_invalid_token_401(self):
        self.failure = (lambda m, x: x == "/user", 401, {}, "Bad credentials")
        code, status = self.run_main()
        self.assertEqual((code, status["category"]), (1, "AUTHENTICATION_FAILED"))
        self.assertEqual(len(self.calls), 1)

    def test_403_preserves_permissions_no_credential_switch(self):
        self.failure = (lambda m, x: m == "POST", 403,
                        {"X-Accepted-GitHub-Permissions": "issues=write; pull_requests=write",
                         "X-GitHub-Request-Id": "test-request"},
                        "Resource not accessible by personal access token")
        code, status = self.run_main()
        self.assertEqual((code, status["category"]), (1, "ACCESS_FORBIDDEN"))
        self.assertEqual(status["token_source"], "GH_RESULTS_TOKEN")
        self.assertEqual(status["accepted_permissions"], "issues=write; pull_requests=write")
        self.assertEqual(status["request_id"], "test-request")
        self.assertEqual(status["write_access"], "UNVERIFIED")
        self.assertTrue((self.root / "evidence-comment.json").is_file())

    def test_422(self):
        self.failure = (lambda m, x: m == "POST", 422, {}, "Validation Failed")
        self.assertEqual(self.run_main()[1]["category"], "VALIDATION_OR_SPAM")

    def test_rate_limit_distinct_from_permissions(self):
        self.failure = (lambda m, x: m == "POST", 403,
                        {"X-RateLimit-Remaining": "0"}, "API rate limit exceeded")
        self.assertEqual(self.run_main()[1]["category"], "RATE_LIMITED")

    def test_hidden_repository(self):
        self.failure = (lambda m, x: x == f"/repos/{p.REPO}", 404, {}, "Not Found")
        self.assertEqual(self.run_main()[1]["category"], "RESOURCE_NOT_FOUND_OR_HIDDEN")
        self.assertFalse(any(m == "POST" for m, _ in self.calls))

    def test_sso_diagnostic(self):
        self.failure = (lambda m, x: m == "POST", 403,
                        {"X-GitHub-SSO": "required; url=https://example.invalid/secret"}, "SSO error")
        status = self.run_main()[1]
        self.assertTrue(status["sso_required"])
        self.assertNotIn("example.invalid", self.output)

    def test_success_and_readback(self):
        code, status = self.run_main()
        self.assertEqual((code, status["publication"]), (0, "CREATED"))
        self.assertEqual(status["write_access"], "VERIFIED")
        self.assertEqual(self.calls[-1], ("GET", f"/repos/{p.REPO}/issues/comments/1"))
        body = self.comments[0]["body"]
        self.assertIn(self.env["CIRCLE_BUILD_URL"], body)
        self.assertIn('"rawtick-kpi.json"', body)

    def test_duplicate_unchanged_no_second_post(self):
        self.run_main()
        self.calls.clear()
        status = self.run_main()[1]
        self.assertEqual(status["publication"], "UNCHANGED")
        self.assertEqual(status["write_access"], "NOT_TESTED_UNCHANGED")
        self.assertFalse(any(m in ("POST", "PATCH") for m, _ in self.calls))
        self.assertEqual(len(self.comments), 1)

    def test_rerun_updates_same_commit(self):
        self.run_main()
        self.env["CIRCLE_BUILD_URL"] += "-rerun"
        self.assertEqual(self.run_main()[1]["publication"], "UPDATED")
        self.assertEqual(len(self.comments), 1)

    def test_other_author_not_overwritten(self):
        self.run_main()
        self.comments[0]["user"]["id"] = 999
        self.assertEqual(self.run_main()[1]["publication"], "CREATED")
        self.assertEqual(len(self.comments), 2)

    def test_paginated_duplicate(self):
        self.run_main()
        saved = self.comments[0]
        self.comments = [{"id": 100+i, "body": "unrelated"} for i in range(100)] + [saved]
        self.assertEqual(self.run_main()[1]["publication"], "UNCHANGED")
        self.assertTrue(any("page=2" in x for _, x in self.calls))

    def test_missing_json_publishes_partial_and_fails_gate(self):
        (self.root / p.FILES[0]).unlink()
        code, status = self.run_main()
        self.assertEqual(code, 2)
        self.assertFalse(status["evidence_complete"])
        self.assertEqual(status["publication"], "CREATED")
        self.assertIn('"MISSING"', self.comments[0]["body"])

    def test_malformed_json_preserved_as_invalid(self):
        (self.root / p.FILES[0]).write_text("not JSON")
        self.assertEqual(self.run_main()[0], 2)
        self.assertIn("INVALID_JSON_OR_UNREADABLE", self.comments[0]["body"])

    def test_oversized_payload_saved_without_post(self):
        (self.root / p.FILES[0]).write_text(json.dumps({"large": "x" * 60000}))
        self.assertEqual(self.run_main()[1]["category"], "PAYLOAD_TOO_LARGE")
        self.assertFalse(any(m == "POST" for m, _ in self.calls))
        self.assertTrue((self.root / "evidence-bundle.json").is_file())

    def test_preflight_never_writes_comment(self):
        code, status = self.run_main(["--preflight-only"])
        self.assertEqual(code, 0)
        self.assertEqual(status["write_access"], "UNVERIFIED")
        self.assertTrue(all(m == "GET" for m, _ in self.calls))

    def test_network_failure_preserves_fallback(self):
        with patch.object(p, "preflight", side_effect=p.PublishError("NETWORK_ERROR")):
            self.assertEqual(self.run_main()[1]["category"], "NETWORK_ERROR")
        self.assertTrue((self.root / "evidence-bundle.json").is_file())

    def test_secrets_redacted_in_evidence_and_diagnostics(self):
        (self.root / p.FILES[0]).write_text(json.dumps({"oops": self.env["GH_TOKEN"]}))
        self.failure = (lambda m, x: m == "POST", 403,
                        {"X-Accepted-GitHub-Permissions": self.env["GH_RESULTS_TOKEN"]},
                        self.env["GH_TOKEN"])
        self.assertEqual(self.run_main()[1]["reason"], "UNCLASSIFIED")

    def test_invalid_sha_no_write(self):
        self.env["CIRCLE_SHA1"] = "unknown"
        self.assertEqual(self.run_main()[1]["category"], "INVALID_COMMIT_SHA")
        self.assertFalse(any(m == "POST" for m, _ in self.calls))

    def test_readback_mismatch_not_success(self):
        original = self.server

        def wrong_readback(req, **kwargs):
            if req.get_method() == "GET" and req.full_url.endswith("/issues/comments/1"):
                return Response({"body": "wrong"})
            return original(req, **kwargs)

        self.server = wrong_readback
        self.assertEqual(self.run_main()[1]["category"], "COMMENT_VERIFICATION_FAILED")

    def test_update_forbidden_preserves_original_comment(self):
        self.run_main()
        original_body = self.comments[0]["body"]
        self.env["CIRCLE_BUILD_URL"] += "-rerun"
        self.failure = (lambda m, x: m == "PATCH", 403, {},
                        "Resource not accessible by personal access token")
        self.assertEqual(self.run_main()[1]["category"], "ACCESS_FORBIDDEN")
        self.assertEqual(self.comments[0]["body"], original_body)
        self.assertEqual(len(self.comments), 1)

    def test_long_header_redacts_before_truncation(self):
        self.failure = (lambda m, x: m == "POST", 403,
                        {"X-Accepted-GitHub-Permissions": "x" * 190 + self.env["GH_RESULTS_TOKEN"]},
                        "Resource not accessible by personal access token")
        status = self.run_main()[1]
        self.assertNotIn("secret-pri", status["accepted_permissions"])
        self.assertLessEqual(len(status["accepted_permissions"]), 200)

    def test_transport_error_classification(self):
        with patch.object(p.urllib.request, "urlopen", side_effect=urllib.error.URLError("private details")):
            with self.assertRaises(p.PublishError) as caught:
                p.request("test-secret", "/user")
        self.assertEqual(caught.exception.details["category"], "NETWORK_ERROR")
        self.assertNotIn("private details", str(caught.exception.details))

    def test_comment_scan_limit_never_posts(self):
        with patch.object(p, "request", return_value=[{"body": "unrelated"}] * 100) as api:
            with self.assertRaises(p.PublishError) as caught:
                p.publish("test-secret", 7, "body", "marker")
        self.assertEqual(caught.exception.details["category"], "COMMENT_SCAN_LIMIT")
        self.assertEqual(api.call_count, 100)
        self.assertTrue(all(len(c.args) == 2 for c in api.call_args_list))


if __name__ == "__main__":
    unittest.main()
