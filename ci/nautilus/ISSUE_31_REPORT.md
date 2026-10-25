# Issue #31: CircleCI evidence publication investigation

Status: **incomplete; original 403 root cause unconfirmed**. No token or permission
changes have been made. This report does not claim a successful real CI comment.

## CHAT_SPEC / ISSUE_DAG / DEPENDENCY

Read [Issue #31](https://github.com/Sai310421/xauusd-data/issues/31), PR #30 and the
existing publisher/config at diagnostic commit
`39f71e2325632c733a9813e28d184758b2b28176`.

Dependency order: inspect latest diagnostics -> establish effective credential and
access -> repair publisher -> offline tests -> Draft PR -> real CircleCI run ->
verify PR #30 comment and artifact contents. The real-credential investigation
remains an external dependency; offline development can proceed independently.

## Evidence and limits

- GitHub commit status API reports `ci/circleci: abc-nautilus-rawtick` = `failure`,
  updated `2026-10-25T04:20:38Z` for diagnostic commit `39f71e2`.
- [Original CircleCI job](https://app.circleci.com/workflow/a9677845-acfb-4592-b7fb-7a0b450e1cb3/job/ea0a175c-31df-495a-8c89-baa1f53d72e2).
- Opening that job in the available browser redirects to CircleCI login with
  "Your session has expired". GitHub sign-in also needs an interactive login.
- An unauthenticated CircleCI v1.1 project-build-list request returned HTTP 404;
  that does not prove the project is absent. No CircleCI process credential is
  available. GitHub CLI has a working *local OAuth* session, which is a different
  credential from the unknown CircleCI PAT and has not been copied into CI.
- PR #30 had zero timeline comments when queried during this investigation.
- The original 403 reason is recorded in Issue #31. Its actual selected variable,
  `X-Accepted-GitHub-Permissions`, request ID, PAT repository grants and effective
  Issues/Pull requests permissions have **not** yet been observed from CircleCI.

The endpoint is correct: GitHub documents PR timeline comments under the issue
comment endpoint. The documented permission alternatives are Issues write OR
Pull requests write; this alone does not justify demanding both. The accepted
permissions header describes endpoint requirements, **not the token's grants**.
Public repository read success also does not prove fine-grained write access.

Sources:
- [GitHub issue comments API](https://docs.github.com/en/rest/issues/comments#create-an-issue-comment)
- [GitHub REST troubleshooting](https://docs.github.com/en/rest/using-the-rest-api/troubleshooting-the-rest-api#resource-not-accessible)
- [CircleCI configuration reference](https://circleci.com/docs/reference/configuration-reference/)

## EXECUTION / REPAIR

Proven code deficiencies: the old preflight only checked variable presence, a
missing token prevented the independent BT, every run posted a new comment,
`rawtick-kpi.json` was omitted from the comment, malformed JSON could crash the
publisher, and no consolidated fallback/status was written before authentication.
These are repaired; none proves the cause of the original GitHub authorization
denial.

- Preserve default `GH_RESULTS_TOKEN > GITHUB_TOKEN > GH_TOKEN` precedence; log
  only source names and selection policy. Optional `EVIDENCE_TOKEN_ENV` pins one
  of these names, fails closed when absent, and never switches after API denial.
- Perform authenticated GET `/user`, repository and PR checks before BT and again
  before publication. These establish read access only; successful comment write
  and readback are required for `write_access=VERIFIED`.
- Classify missing auth, 401, 403, rate limits, hidden/not-found resources, 422,
  malformed payloads and network failures. Allowlist response reasons; record only
  bounded diagnostic headers. Never log token values, hashes or fingerprints.
- Save `evidence-bundle.json` before authentication; save outgoing
  `evidence-comment.json` when available and `evidence-publish-status.json` with
  separate BT/publication fields. Missing/invalid JSON is explicit partial evidence
  and keeps the publication gate nonzero, even if that partial comment is posted.
- Include the original seven JSON results, commit SHA, CircleCI job URL, workflow
  and build number. No raw ticks are posted.
- Scan paginated comments by commit marker and authenticated author; create,
  update on rerun or skip unchanged content, then read back the stored body.
  Fail closed at the scan bound. CircleCI serializes the configured jobs in one
  group to prevent competing POSTs; standalone concurrent invocations are not an
  atomic GitHub upsert.
- Keep auth preflight non-blocking for BT, preserve the original Docker/BT commands,
  record the Docker exit as BT status, retain a separate failing publication gate,
  and retain artifacts on failure. Enable the repair branch in CI filters.

Artifacts under `results/amos-abc/` are stored at CircleCI destination
`abc-nautilus-rawtick`. When commenting works, PR #30 contains the evidence JSON
inline, without an artifact download. When auth fails, artifact fallback alone
does **not** meet the PR-visible automatic publication acceptance criterion.

## VALIDATION

Executed locally with Python 3.11:

```
python -m unittest discover -s ci/nautilus -p test_publish_evidence.py -v
Ran 27 tests ... OK
```

Tests cover missing/invalid credentials, selection precedence and explicit source,
401, 403 with permission/request headers, 422, rate limits, SSO diagnostics, hidden
repository, creation/readback, unchanged duplicate, rerun PATCH, other-author
protection, paginated duplicate, missing/invalid JSON, oversized fallback, network
failure, secret redaction before truncation, invalid SHA, failed readback, denied
PATCH and scan-limit behavior. All API and BT evidence fixtures are simulated.
This is not a measured Raw Tick backtest or a real GitHub-write test.

PyYAML 6.0.3 parsed `.circleci/config.yml`; workflow structure assertions and
`git diff --check` passed. CircleCI CLI/server validation has not been performed
locally. No trading, data, strategy, KPI or live-order code is changed.

## REPORT / NEXT

The Draft PR and Issue #31 comments carry the repair commit SHA and real CI status
once submitted. Do not close Issue #31 on offline tests alone.

Next required evidence: read the original and repair jobs' sanitized auth source,
HTTP category, accepted-permissions/request headers and BT output; verify the
selected PAT's actual repository/owner/grants or access restriction based on those
observations; repair only the demonstrated issue; rerun and verify an inline
comment on PR #30. Browser login to the existing CircleCI account is sufficient
to inspect logs; no token reissue or manual log download is requested.

Authentication options, if a credential problem is established:

| Option | Existing support / constraints |
| --- | --- |
| Current fine-grained PAT | Supported; least change. First prove the actual selected credential and effective target access. |
| Another existing CI variable | Explicit selection supported; do not silently try unrelated credentials after 403. |
| GitHub App installation token | Short-lived and repo-scoped, but requires an installed App, grant verification, key/token issuance and an installation-compatible identity preflight. Not provisioned or enabled here. |
| Local GitHub CLI OAuth | Works for development/reporting; says nothing about the CI PAT. Not transferred into CircleCI. |

Remaining acceptance items: original root-cause proof; repaired live CircleCI
result, real saved Raw Tick JSON and verified automatic inline publication on
PR #30. These remain pending until their actual evidence can be read.
