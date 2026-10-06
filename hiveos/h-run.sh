#!/usr/bin/env bash
set -o pipefail

MINER_DIR="/hive/miners/custom/prl0"
CONF="$MINER_DIR/prl0.conf"
LOG="/var/log/miner/custom/prl0.log"

if [[ ! -f "$CONF" ]]; then
  echo "PRL0: falta $CONF; executa h-config.sh/Flight Sheet primeiro." >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$CONF"
export PRL_POOL PRL_WALLET PRL_WORKER PRL_PASS PRL_SHAPE PEARL_DEVICES PEARL_FATBIN MAX_ITERS

mkdir -p "$(dirname "$LOG")"
cd "$MINER_DIR" || exit 1
exec ./prl0-kryptex 2>&1 | tee -a "$LOG"
