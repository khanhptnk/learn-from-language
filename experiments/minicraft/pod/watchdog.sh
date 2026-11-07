#!/usr/bin/env bash
# Run on the pod (nohup): stop (ACTION=stop, default) or terminate (ACTION=terminate) this pod after IDLE_MIN minutes
# with no job (.venv python process) running.
# Stopping keeps /workspace; GPU billing ends (a stopped pod still pays for its volume, ~$0.20/GB/month).
IDLE_MIN=${IDLE_MIN:-20}
ACTION=${ACTION:-stop}
eval "$(tr '\0' '\n' < /proc/1/environ | grep -E '^RUNPOD_(API_KEY|POD_ID)=' | sed 's/^/export /')"
idle=0
while true; do
  # a job = a process whose executable (first word of its command line) is the project's venv python; matching
  # anywhere in the command line would also count shells whose command text merely mentions it
  if pgrep -f "^[^ ]*\.venv/bin/python " >/dev/null; then idle=0; else idle=$((idle + 1)); fi
  if [ "$idle" -ge "$IDLE_MIN" ]; then
    echo "$(date) idle ${IDLE_MIN} min: ${ACTION} pod $RUNPOD_POD_ID"
    if [ "$ACTION" = terminate ]; then q="mutation { podTerminate(input: {podId: \\\"$RUNPOD_POD_ID\\\"}) }"
    else q="mutation { podStop(input: {podId: \\\"$RUNPOD_POD_ID\\\"}) { id desiredStatus } }"; fi
    curl -s https://api.runpod.io/graphql -H "Content-Type: application/json" -H "Authorization: Bearer $RUNPOD_API_KEY" \
      -d "{\"query\": \"$q\"}"
    exit 0
  fi
  sleep 60
done
