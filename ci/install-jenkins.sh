#!/usr/bin/env bash
# Install Jenkins LTS on Ubuntu from the official signed apt repository.
# Jenkins listens on 127.0.0.1:8080 only and runs as a systemd service.
# Safe to run again: every step checks or overwrites its own files.
#
# Usage: sudo bash ci/install-jenkins.sh
set -euo pipefail

JENKINS_KEY_URL="https://pkg.jenkins.io/debian-stable/jenkins.io-2026.key"
KEYRING=/etc/apt/keyrings/jenkins-keyring.asc
LISTEN_ADDRESS=127.0.0.1
PORT=8080

if [[ $EUID -ne 0 ]]; then
  echo "Run this script with sudo: sudo bash $0" >&2
  exit 1
fi

echo "==> Installing Java 21 and build tools"
apt-get update -q
apt-get install -y -q fontconfig openjdk-21-jre-headless git python3-venv curl

echo "==> Adding the Jenkins LTS apt repository"
install -d -m 0755 /etc/apt/keyrings
curl -fsSL "$JENKINS_KEY_URL" -o "$KEYRING"
echo "deb [signed-by=$KEYRING] https://pkg.jenkins.io/debian-stable binary/" \
  > /etc/apt/sources.list.d/jenkins.list

echo "==> Installing Jenkins"
apt-get update -q
apt-get install -y -q jenkins

echo "==> Restricting Jenkins to $LISTEN_ADDRESS:$PORT"
install -d /etc/systemd/system/jenkins.service.d
cat > /etc/systemd/system/jenkins.service.d/override.conf <<EOF
[Service]
Environment="JENKINS_LISTEN_ADDRESS=$LISTEN_ADDRESS"
Environment="JENKINS_PORT=$PORT"
EOF
systemctl daemon-reload
systemctl enable jenkins
systemctl restart jenkins

echo "==> Waiting for Jenkins to start"
for _ in $(seq 1 60); do
  if curl -s -o /dev/null "http://$LISTEN_ADDRESS:$PORT/login"; then
    break
  fi
  sleep 2
done

systemctl --no-pager --lines=0 status jenkins
java -version 2>&1 | head -n 1
echo "Jenkins $(dpkg-query -W -f='${Version}' jenkins)"
echo
echo "Jenkins is running at http://localhost:$PORT"
if [[ -f /var/lib/jenkins/secrets/initialAdminPassword ]]; then
  echo "Unlock it with the password shown by:"
  echo "  sudo cat /var/lib/jenkins/secrets/initialAdminPassword"
fi
