#!/usr/bin/env bash
set -e
cd /workspace/vectoradd-trad

# Source env for Modal/Anthropic credentials
if [[ -f .env ]]; then
    set -a; source .env; set +a
fi

echo "Checking GPU..."
OUTPUT=$(uv run python vectoradd_kernel/run_eval.py vectoradd_kernel/submission.py -o /tmp/gpu_check.json --mode test 2>&1)
echo "$OUTPUT"

GPU_LINE=$(echo "$OUTPUT" | grep "GPU:" || true)
echo ""
echo "Detected: $GPU_LINE"

if echo "$OUTPUT" | grep -q "NVIDIA H100"; then
    echo ""
    echo "--- GPU is H100 — launching agent in tmux ---"
    echo ""
    SESSION="vectoradd-agent"
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        tmux kill-session -t "$SESSION"
    fi
    tmux new-session -d -s "$SESSION" -c "/workspace/vectoradd-trad" \
        "bash -c 'set -a && source /workspace/vectoradd-trad/.env && set +a && uv run vectoradd_kernel/agent.py --baseline vectoradd_kernel/starting_point.py --iterations 25 2>&1 | tee /tmp/agent_run.log; echo; echo \"--- agent finished, press any key to exit ---\"; read -n1'"
    tmux attach-session -t "$SESSION"
else
    echo ""
    echo "--- GPU is NOT H100 — aborting ---"
    exit 1
fi
