#!/usr/bin/env bash
# HiveOS custom miner config generator for PRL0.

miner_ver() {
  echo "0.1.8"
}

miner_config_echo() {
  local cfg="${CUSTOM_CONFIG_FILENAME:-/hive/miners/custom/prl0/prl0.conf}"
  [[ -f "$cfg" ]] && cat "$cfg"
}

miner_config_gen() {
  local dir="/hive/miners/custom/prl0"
  local cfg="${CUSTOM_CONFIG_FILENAME:-$dir/prl0.conf}"
  mkdir -p "$dir"

  local pool="${CUSTOM_URL:-prl-eu.kryptex.network:7048}"
  pool="${pool#stratum+tcp://}"
  pool="${pool#tcp://}"

  local login="${CUSTOM_TEMPLATE:-${CUSTOM_WALLET:-}}"
  local worker="${WORKER_NAME:-hive}"
  local pass="${CUSTOM_PASS:-x}"
  local shape="small"
  local devices=""
  local graphs="0"

  # Hive often expands %WAL%.%WORKER_NAME% before h-config.sh is called.
  # PRL0 sends wallet and worker as separate authorize fields, so strip the
  # worker suffix when it is present in the template.
  if [[ -n "$worker" ]]; then
    [[ "$login" == *".$worker" ]] && login="${login%.$worker}"
    [[ "$login" == *"/$worker" ]] && login="${login%/$worker}"
  fi

  # Extra config examples:
  #   shape=small
  #   shape=big_m
  #   shape=huge_m devices=0,1,2,3
  if [[ -n "${CUSTOM_USER_CONFIG:-}" ]]; then
    local token
    for token in ${CUSTOM_USER_CONFIG}; do
      case "$token" in
        shape=*) shape="${token#shape=}" ;;
        devices=*) devices="${token#devices=}" ;;
        graphs=*) graphs="${token#graphs=}" ;;
      esac
    done
  fi

  case "$shape" in
    small|big_m|huge_m) ;;
    *) shape="small" ;;
  esac

  if [[ -z "$login" ]]; then
    echo "PRL0: wallet/login vazio. Define Wallet/Worker Template no Flight Sheet." >&2
    return 1
  fi

  printf 'PRL_POOL=%q\n' "$pool" > "$cfg"
  printf 'PRL_WALLET=%q\n' "$login" >> "$cfg"
  printf 'PRL_WORKER=%q\n' "$worker" >> "$cfg"
  printf 'PRL_PASS=%q\n' "$pass" >> "$cfg"
  printf 'PRL_SHAPE=%q\n' "$shape" >> "$cfg"
  printf 'PEARL_DEVICES=%q\n' "$devices" >> "$cfg"
  printf 'PEARL_FATBIN=%q\n' "$dir/pearl_gemm.fatbin" >> "$cfg"
  printf 'PRL_GRAPHS=%q\n' "$graphs" >> "$cfg"
  printf 'MAX_ITERS=%q\n' "0" >> "$cfg"
}

# Older Custom Miner launchers may only source h-config.sh; generation here is
# idempotent, and newer HiveOS will safely call miner_config_gen again.
if [[ -n "${CUSTOM_URL:-}" && -n "${CUSTOM_TEMPLATE:-${CUSTOM_WALLET:-}}" ]]; then
  miner_config_gen
fi
