"""Show the status of an ollama-poc Jenkins build (default: the last one).

Signs in with the token that ci/jenkins-login.sh saved in ~/.netrc.

Usage: python3 ci/jenkins_status.py [BUILD_NUMBER]
"""

import base64
import datetime
import json
import netrc
import sys
import urllib.error
import urllib.request

HOST = "127.0.0.1"
JOB_URL = f"http://{HOST}:8080/job/ollama-poc"
BUILD_TREE = ("number,result,building,duration,timestamp,"
              "actions[causes[shortDescription]],changeSets[items[commitId,msg]]")


def get_json(path):
    try:
        login, _, token = netrc.netrc().authenticators(HOST) or (None, None, None)
    except FileNotFoundError:
        login = None
    if not login:
        sys.exit(f"No Jenkins token for {HOST} in ~/.netrc: run `bash ci/jenkins-login.sh` first.")
    request = urllib.request.Request(f"{JOB_URL}/{path}")
    request.add_header("Authorization", "Basic " + base64.b64encode(f"{login}:{token}".encode()).decode())
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.load(response)


def main(build="lastBuild"):
    try:
        b = get_json(f"{build}/api/json?tree={BUILD_TREE}")
    except urllib.error.HTTPError as exc:
        sys.exit(f"Jenkins answered {exc.code} for build {build}.")

    causes = [c["shortDescription"] for a in b["actions"] if a for c in a.get("causes", [])]
    commits = [f"{i['commitId'][:7]} {i['msg']}" for cs in b["changeSets"] for i in cs["items"]]
    started = datetime.datetime.fromtimestamp(b["timestamp"] / 1000).strftime("%Y-%m-%d %H:%M:%S")

    print(f"Build:    #{b['number']}")
    print(f"Status:   {'RUNNING' if b['building'] else b['result']}")
    print(f"Started:  {started} ({', '.join(causes) or 'unknown cause'})")
    if not b["building"]:
        print(f"Duration: {b['duration'] / 1000:.0f} s")
    print(f"Commits:  {'; '.join(commits) or '(none new)'}")

    try:
        t = get_json(f"{build}/testReport/api/json?tree=passCount,failCount,skipCount")
        print(f"Tests:    {t['passCount']} passed, {t['failCount']} failed, {t['skipCount']} skipped")
    except urllib.error.HTTPError:
        print("Tests:    no test report (yet)")

    try:
        stats = get_json(f"{build}/coverage/api/json")["projectStatistics"]
        print(f"Coverage: {stats['line']} of lines, {stats['branch']} of branches")
    except urllib.error.HTTPError:
        print("Coverage: no coverage report (yet)")


if __name__ == "__main__":
    main(*sys.argv[1:2])
