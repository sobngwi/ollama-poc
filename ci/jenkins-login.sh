#!/usr/bin/env bash
# Save a Jenkins API token in ~/.netrc (mode 600) so `curl --netrc` can call the local Jenkins API.
# Create the token in Jenkins: your name (top right) > Security > API Token > Add new token.
#
# Usage: bash ci/jenkins-login.sh
set -euo pipefail

HOST=127.0.0.1
JENKINS_URL="http://$HOST:8080"
NETRC="$HOME/.netrc"

read -rp "Jenkins username: " user
read -rsp "Jenkins API token (hidden): " token
echo

touch "$NETRC"
chmod 600 "$NETRC"
# Replace any earlier entry for this host, keep everything else.
grep -v "^machine $HOST " "$NETRC" > "$NETRC.tmp" || true
echo "machine $HOST login $user password $token" >> "$NETRC.tmp"
mv "$NETRC.tmp" "$NETRC"
chmod 600 "$NETRC"

who=$(curl -s --netrc "$JENKINS_URL/whoAmI/api/json" | python3 -c 'import sys, json; print(json.load(sys.stdin)["name"])')
if [[ "$who" == "anonymous" ]]; then
  echo "Jenkins did not accept that username/token." >&2
  exit 1
fi
echo "OK: the Jenkins API accepts you as '$who'."
