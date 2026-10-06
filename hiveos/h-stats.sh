#!/usr/bin/env bash
# HiveOS expects variables named khs and stats.
LOG="/var/log/miner/custom/prl0.log"
VER="0.1.0"

khs=0
stats="null"

[[ -f "$LOG" ]] || return 0 2>/dev/null || exit 0
line="$(grep '\[hive\]' "$LOG" | tail -n 1)"
[[ -n "$line" ]] || return 0 2>/dev/null || exit 0

gpu_csv="$(sed -n 's/.*gpu_ths=\([^ ]*\).*/\1/p' <<<"$line")"
total_ths="$(sed -n 's/.*total_ths=\([^ ]*\).*/\1/p' <<<"$line")"
accepted="$(sed -n 's/.*accepted=\([0-9]*\).*/\1/p' <<<"$line")"
rejected="$(sed -n 's/.*rejected=\([0-9]*\).*/\1/p' <<<"$line")"
uptime="$(sed -n 's/.*uptime=\([0-9]*\).*/\1/p' <<<"$line")"

[[ -n "$total_ths" ]] || total_ths=0
[[ -n "$accepted" ]] || accepted=0
[[ -n "$rejected" ]] || rejected=0
[[ -n "$uptime" ]] || uptime=0

# Hive's internal base is kH/s; 1 TH/s = 1e9 kH/s.
khs="$(awk -v x="$total_ths" 'BEGIN { printf "%.0f", x * 1000000000 }')"

if [[ -n "$gpu_csv" ]]; then
  hs_json="[$gpu_csv]"
else
  hs_json="[]"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  temp_json="[$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | paste -sd, -)]"
  fan_json="[$(nvidia-smi --query-gpu=fan.speed --format=csv,noheader,nounits 2>/dev/null | tr -d ' %' | paste -sd, -)]"
else
  temp_json="[]"
  fan_json="[]"
fi

if command -v jq >/dev/null 2>&1; then
  stats="$(jq -cn \
    --argjson hs "$hs_json" \
    --argjson temp "$temp_json" \
    --argjson fan "$fan_json" \
    --argjson uptime "$uptime" \
    --argjson accepted "$accepted" \
    --argjson rejected "$rejected" \
    --arg ver "$VER" \
    '{hs:$hs,hs_units:"ths",temp:$temp,fan:$fan,uptime:$uptime,ver:$ver,ar:[$accepted,$rejected],algo:"pearlhash"}')"
else
  stats="{\"hs\":$hs_json,\"hs_units\":\"ths\",\"temp\":$temp_json,\"fan\":$fan_json,\"uptime\":$uptime,\"ver\":\"$VER\",\"ar\":[$accepted,$rejected],\"algo\":\"pearlhash\"}"
fi
