"""Publish small CI evidence as a GitHub PR comment (no manual artifact download).

Requires CircleCI environment variable GH_RESULTS_TOKEN with Issues:write
permission on Sai310421/xauusd-data. Never publish credentials or raw ticks.
"""
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT=Path("results/amos-abc")
FILES=("run-manifest.json","rawtick-input-quality.json","native-gate-summary.json",
       "native-matching-A01.json","native-matching-B01.json","native-matching-C01.json")
REPO="Sai310421/xauusd-data"
PR=30

def main():
    token=os.environ.get("GH_RESULTS_TOKEN","")
    if not token:
        print("EVIDENCE_PUBLISH_BLOCKED: GH_RESULTS_TOKEN missing (configure once in CircleCI)",flush=True)
        return 2
    evidence={}
    for name in FILES:
        path=ROOT/name
        if path.exists():
            evidence[name]=json.loads(path.read_text())
        else:
            evidence[name]={"status":"MISSING"}
    sha=os.environ.get("CIRCLE_SHA1","unknown")
    job=os.environ.get("CIRCLE_BUILD_URL","unknown")
    payload={"schema_version":1,"commit_sha":sha,"circleci_job":job,
             "evidence":evidence}
    body="<!-- AMOS_NATIVE_EVIDENCE_V1 -->\n### AMOS Raw Tick CI evidence\n" + \
         "Commit: `"+sha+"`\n\n```json\n"+json.dumps(payload,indent=2)+"\n```"
    if len(body)>60000:
        print("EVIDENCE_PUBLISH_FAILED: payload exceeds 60000 chars",flush=True)
        return 2
    req=urllib.request.Request(
        f"https://api.github.com/repos/{REPO}/issues/{PR}/comments",
        data=json.dumps({"body":body}).encode(),
        headers={"Authorization":"Bearer "+token,"Accept":"application/vnd.github+json",
                 "X-GitHub-Api-Version":"2022-11-28","Content-Type":"application/json",
                 "User-Agent":"AMOS-CircleCI-Evidence"},
        method="POST")
    try:
        with urllib.request.urlopen(req,timeout=30) as response:
            result=json.load(response)
    except urllib.error.HTTPError as exc:
        print(f"EVIDENCE_PUBLISH_FAILED: GitHub HTTP {exc.code}",flush=True)
        return 1
    except Exception as exc:
        print(f"EVIDENCE_PUBLISH_FAILED: {type(exc).__name__}",flush=True)
        return 1
    print("EVIDENCE_PUBLISHED "+result.get("html_url",""),flush=True)
    return 0

if __name__=="__main__":
    sys.exit(main())
