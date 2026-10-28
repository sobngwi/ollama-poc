#!/usr/bin/env bash
# Share one pip cache between you and the Jenkins service user, so a package
# downloaded by either of you is never downloaded again by the other.
#
#   /var/cache/pip-shared          root:pipcache, group-writable (setgid + default ACL)
#   ~YOU/.cache/pip              -> /var/cache/pip-shared
#   /var/lib/jenkins/.cache/pip  -> /var/cache/pip-shared
#
# Existing caches are merged into the shared folder with hard links (instant, no extra
# disk space when on the same filesystem); the originals are kept as pip.bak-<date>.
# Jenkins is NOT restarted: the script prints the command to run when no build is running.
# Safe to run again.
#
# Usage: sudo bash ci/share-pip-cache.sh
set -euo pipefail

SHARED=/var/cache/pip-shared
GROUP=pipcache
STAMP=$(date +%Y%m%d-%H%M%S)

die() { echo "Error: $*" >&2; exit 1; }
step() { echo "==> $*"; }

[[ $EUID -eq 0 ]] || die "run this script with sudo: sudo bash $0"
[[ -n "${SUDO_USER:-}" && "$SUDO_USER" != root ]] || die "run it with sudo from your own account, not as root"
id jenkins >/dev/null 2>&1 || die "user 'jenkins' not found (is Jenkins installed?)"
USERS=("$SUDO_USER" jenkins)

if pgrep -af 'pip[0-9.]* (install|download|wheel)' >/dev/null; then
  pgrep -af 'pip[0-9.]* (install|download|wheel)' >&2
  die "pip is running (see above); wait until it finishes, e.g. the Jenkins build, then run again"
fi

if ! command -v setfacl >/dev/null; then
  step "Installing the acl tools (setfacl)"
  apt-get install -y -q acl
fi

step "Creating group '$GROUP' with members: ${USERS[*]}"
groupadd -f "$GROUP"
for user in "${USERS[@]}"; do
  usermod -aG "$GROUP" "$user"
done

step "Creating $SHARED"
install -d -o root -g "$GROUP" -m 2775 "$SHARED"

for user in "${USERS[@]}"; do
  home=$(getent passwd "$user" | cut -d: -f6)
  cache="$home/.cache/pip"
  if [[ -L "$cache" ]]; then
    [[ "$(readlink -f "$cache")" == "$SHARED" ]] || die "$cache is a link to $(readlink "$cache"), not $SHARED"
    step "$user: $cache already links to $SHARED"
    continue
  fi
  if [[ -d "$cache" ]]; then
    step "$user: merging $cache ($(du -sh "$cache" | cut -f1)) into $SHARED"
    if [[ "$(stat -c %d "$cache")" == "$(stat -c %d "$SHARED")" ]]; then
      cp -a -l -n "$cache/." "$SHARED/"   # hard links: instant, no extra space
    else
      cp -a -n "$cache/." "$SHARED/"
    fi
    mv "$cache" "$cache.bak-$STAMP"
    echo "    original kept as $cache.bak-$STAMP"
  fi
  install -d -o "$user" -g "$(id -gn "$user")" "$home/.cache"
  ln -s "$SHARED" "$cache"
  chown -h "$user:$(id -gn "$user")" "$cache"
  step "$user: $cache -> $SHARED"
done

step "Making everything in $SHARED readable and writable by group '$GROUP'"
chgrp -R "$GROUP" "$SHARED"
chmod -R g+rwX "$SHARED"
find "$SHARED" -type d -exec chmod g+s {} +
setfacl -R -m "g:$GROUP:rwX" "$SHARED"
setfacl -R -d -m "g:$GROUP:rwX" "$SHARED"   # default ACL: new files stay group-writable

step "Checking that each user can write to the shared cache"
for user in "${USERS[@]}"; do
  probe="$SHARED/.write-test-$user"
  sudo -u "$user" touch "$probe" && rm -f "$probe" || die "$user cannot write to $SHARED"
  echo "    $user: OK"
done

cat <<EOF

Done. Shared pip cache: $SHARED ($(du -sh "$SHARED" | cut -f1))

To finish:
  1. When no Jenkins build is running, restart Jenkins so it joins group '$GROUP':
       sudo systemctl restart jenkins
  2. Log out and back in (or run: newgrp $GROUP) so your own account joins it too.
  3. When everything works, delete the backups:
       rm -rf ~$SUDO_USER/.cache/pip.bak-$STAMP
       sudo rm -rf /var/lib/jenkins/.cache/pip.bak-$STAMP

To undo:
  for each user: rm ~/.cache/pip && mv ~/.cache/pip.bak-$STAMP ~/.cache/pip
  sudo rm -rf $SHARED && sudo groupdel $GROUP
EOF
