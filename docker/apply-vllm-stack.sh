#!/usr/bin/env bash
set -uo pipefail

LOG="${VLLM_STACK_LOG:-/opt/r9700-vllm/build-info/apply-vllm-stack.log}"
BASE_COMMIT="${VLLM_BASE_COMMIT:-c3284c31f52c005bde02cf7899959c2539b01d2f}"

GGZ_GDN_REF="refs/remotes/ggz14/fix/gdn-kkt-rdna4-tp2"
GGZ_GDN_PIN="55f952aa6ce88b8f0e0a51266852d4008929061e"
GGZ_AITER_REF="refs/remotes/ggz14/feat/aiter-unified-attn-gfx1201"
GGZ_AITER_PIN="184e06dadb69a803f3189708bc0aaa1244c0a358"
GGZ_FP8_REF="refs/remotes/ggz14/perf/rocm-fp8-kv-decode-dequant"
GGZ_FP8_PIN="6de8e1d05b3107b1fc3f27f243113b323455b097"
AR_FUSED_REF="refs/remotes/ar/feat/fused-rope-fp8-kvcache"
AR_FUSED_PIN="21c7e1352a71f716ad9e23480a905604367e2050"
FEI_SPLITKV_REF="refs/remotes/feiyehua/feiyehua/rocm-gfx12xx-splitkv"
FEI_SPLITKV_PIN="7ef12017eb47e9b90deccf2eafd7026b6fc06b8b"

mkdir -p "$(dirname "$LOG")"
exec > >(tee "$LOG") 2>&1

cd /opt/r9700-vllm/src/vllm || { echo "FAILED: /opt/r9700-vllm/src/vllm does not exist."; exit 1; }

echo "============================================================"
echo "vLLM RDNA4/R9700 stack apply"
echo "Started: $(date -Is)"
echo "Repo:    $(pwd)"
echo "Base:    $BASE_COMMIT"
echo "Log:     $LOG"
echo "============================================================"

run() {
  echo
  echo "+ $*"
  "$@"
  local rc=$?
  if [ $rc -ne 0 ]; then echo "Command failed with exit code $rc: $*"; fi
  return $rc
}

clean_repo_to_base() {
  echo
  echo "============================================================"
  echo "Cleaning repo and resetting to pinned base"
  echo "============================================================"
  git cherry-pick --abort >/dev/null 2>&1 || true
  git merge --abort >/dev/null 2>&1 || true
  git rebase --abort >/dev/null 2>&1 || true
  run git reset --hard || return 1
  run git clean -fd || return 1
  run git fetch upstream main || return 1
  run git checkout -B r9700-c3284-base "$BASE_COMMIT" || return 1
  run git status --short || return 1
}

fetch_refs() {
  echo
  echo "============================================================"
  echo "Fetching pinned branch refs"
  echo "============================================================"
  run git fetch upstream main || return 1
  run git fetch ggz14 fix/gdn-kkt-rdna4-tp2:refs/remotes/ggz14/fix/gdn-kkt-rdna4-tp2 || return 1
  run git fetch ggz14 feat/aiter-unified-attn-gfx1201:refs/remotes/ggz14/feat/aiter-unified-attn-gfx1201 || return 1
  run git fetch ggz14 perf/rocm-fp8-kv-decode-dequant:refs/remotes/ggz14/perf/rocm-fp8-kv-decode-dequant || return 1
  run git fetch ar feat/fused-rope-fp8-kvcache:refs/remotes/ar/feat/fused-rope-fp8-kvcache || return 1
  run git fetch feiyehua feiyehua/rocm-gfx12xx-splitkv:refs/remotes/feiyehua/feiyehua/rocm-gfx12xx-splitkv || return 1
}

verify_pin() {
  local label="$1" ref="$2" expected="$3"
  echo
  echo "Verifying pin: $label"
  echo "Ref:      $ref"
  echo "Expected: $expected"
  local actual
  actual="$(git rev-parse "$ref" 2>/dev/null)" || { echo "FAILED: Could not resolve $ref"; return 1; }
  echo "Actual:   $actual"
  if [ "$actual" != "$expected" ]; then
    echo "FAILED: $label branch tip changed. This script will not apply a moving target."
    return 1
  fi
  echo "OK: $label is pinned correctly."
}

verify_all_pins() {
  echo
  echo "============================================================"
  echo "Verifying fork branch tips"
  echo "============================================================"
  verify_pin "GGZ14 gdn-kkt-rdna4-tp2" "$GGZ_GDN_REF" "$GGZ_GDN_PIN" || return 1
  verify_pin "GGZ14 aiter-unified-attn-gfx1201" "$GGZ_AITER_REF" "$GGZ_AITER_PIN" || return 1
  verify_pin "GGZ14 rocm-fp8-kv-decode-dequant" "$GGZ_FP8_REF" "$GGZ_FP8_PIN" || return 1
  verify_pin "A-R fused-rope-fp8-kvcache" "$AR_FUSED_REF" "$AR_FUSED_PIN" || return 1
  verify_pin "feiyehua rocm-gfx12xx-splitkv" "$FEI_SPLITKV_REF" "$FEI_SPLITKV_PIN" || return 1
}

resolve_known_fused_rope_conflict() {
  local commit="$1"
  if [ "$commit" != "$AR_FUSED_PIN" ]; then echo "No known auto-resolver for commit $commit"; return 1; fi
  if [ ! -f csrc/torch_bindings.cpp ]; then echo "Known conflict file missing: csrc/torch_bindings.cpp"; return 1; fi
  if ! grep -q "<<<<<<<" csrc/torch_bindings.cpp; then echo "No conflict markers found in csrc/torch_bindings.cpp"; return 1; fi
  if ! grep -q "get_cuda_view_from_cpu_tensor" csrc/torch_bindings.cpp; then echo "Conflict does not match known fused-RoPE stale registration block."; return 1; fi

  echo
  echo "Auto-resolving known fused-RoPE conflict in csrc/torch_bindings.cpp"
  python - <<'PY'
from pathlib import Path
p = Path('csrc/torch_bindings.cpp')
lines = p.read_text().splitlines(keepends=True)
out = []
i = 0
resolved = False
while i < len(lines):
    line = lines[i]
    if line.startswith('<<<<<<< HEAD'):
        block = []
        j = i
        while j < len(lines):
            block.append(lines[j])
            if lines[j].startswith('>>>>>>>'):
                break
            j += 1
        text = ''.join(block)
        if 'get_cuda_view_from_cpu_tensor' not in text:
            raise SystemExit('Found a conflict block, but it is not the known stale get_cuda_view conflict.')
        i = j + 1
        resolved = True
        continue
    out.append(line)
    i += 1
if not resolved:
    raise SystemExit('No known conflict block resolved.')
p.write_text(''.join(out))
print('Resolved csrc/torch_bindings.cpp.')
PY
  if grep -n "<<<<<<<\|=======\|>>>>>>>" csrc/torch_bindings.cpp; then echo "FAILED: conflict markers remain after auto-resolve."; return 1; fi
  run git diff --check || return 1
  run git add -A || return 1
  GIT_EDITOR=true run git cherry-pick --continue || return 1
  echo "OK: fused-RoPE conflict auto-resolved and cherry-pick continued."
}

apply_unique_no_merges_from_pin() {
  local label="$1" pin="$2"
  echo
  echo "============================================================"
  echo "Applying: $label"
  echo "Pinned commit: $pin"
  echo "============================================================"
  local base
  base="$(git merge-base upstream/main "$pin")" || { echo "FAILED: Could not compute merge-base for $label."; return 1; }
  echo "Merge-base: $base"
  echo
  echo "All unique commits, including merges:"
  git rev-list --reverse "${base}..${pin}" | tee "/opt/r9700-vllm/build-info/${label//[^a-zA-Z0-9_.-]/_}.all-commits.txt"
  echo
  echo "Non-merge commits to cherry-pick:"
  mapfile -t commits < <(git rev-list --reverse --no-merges "${base}..${pin}")
  printf '%s\n' "${commits[@]}" | tee "/opt/r9700-vllm/build-info/${label//[^a-zA-Z0-9_.-]/_}.picked-commits.txt"
  echo "Non-merge unique commits: ${#commits[@]}"
  if [ "${#commits[@]}" -eq 0 ]; then echo "No non-merge commits to apply; skipping."; return 0; fi
  local c
  for c in "${commits[@]}"; do
    echo
    echo "Cherry-picking $c"
    git cherry-pick -x "$c"
    local rc=$?
    if [ $rc -eq 0 ]; then echo "OK: cherry-picked $c"; continue; fi
    echo "Cherry-pick stopped on $c"
    git status --short
    if resolve_known_fused_rope_conflict "$c"; then echo "OK: recovered from known conflict for $c"; continue; fi
    echo "FAILED: unresolved cherry-pick failure for $label commit $c"
    git cherry-pick --abort >/dev/null 2>&1 || true
    return 1
  done
}

SUMMARY_PRIMARY="not-run"
SUMMARY_SECONDARY="not-run"
SUMMARY_SPLITKV="not-run"

clean_repo_to_base || { echo "FAILED: Could not reset repo to base."; exit 1; }
fetch_refs || { echo "FAILED: Could not fetch refs."; exit 1; }
verify_all_pins || { echo "FAILED: Pin verification failed."; exit 1; }

echo
echo "============================================================"
echo "Creating primary branch from base"
echo "============================================================"
run git checkout -B r9700-c3284-primary "$BASE_COMMIT"

if apply_unique_no_merges_from_pin "01-ggz14-gdn-kkt-rdna4-tp2" "$GGZ_GDN_PIN" \
  && apply_unique_no_merges_from_pin "02-ggz14-aiter-unified-attn-gfx1201" "$GGZ_AITER_PIN" \
  && apply_unique_no_merges_from_pin "03-ggz14-rocm-fp8-kv-decode-dequant" "$GGZ_FP8_PIN"; then
  run git tag -f r9700-primary-ok
  SUMMARY_PRIMARY="success"
else
  SUMMARY_PRIMARY="failed"
fi

if [ "$SUMMARY_PRIMARY" = "success" ]; then
  echo
  echo "============================================================"
  echo "Creating secondary branch from primary"
  echo "============================================================"
  run git checkout -B r9700-c3284-secondary r9700-c3284-primary
  if apply_unique_no_merges_from_pin "04-ar-fused-rope-fp8-kvcache" "$AR_FUSED_PIN"; then
    run git tag -f r9700-secondary-ok
    SUMMARY_SECONDARY="success"
  else
    SUMMARY_SECONDARY="failed"
  fi
else
  SUMMARY_SECONDARY="skipped-primary-failed"
fi

if [ "$SUMMARY_SECONDARY" = "success" ]; then
  echo
  echo "============================================================"
  echo "Creating splitKV experimental branch from secondary"
  echo "============================================================"
  run git checkout -B r9700-c3284-splitkv r9700-c3284-secondary
  if apply_unique_no_merges_from_pin "05-feiyehua-rocm-gfx12xx-splitkv" "$FEI_SPLITKV_PIN"; then
    run git tag -f r9700-splitkv-ok
    SUMMARY_SPLITKV="success"
  else
    SUMMARY_SPLITKV="failed"
  fi
else
  SUMMARY_SPLITKV="skipped-secondary-failed"
fi

echo
echo "============================================================"
echo "Final status"
echo "============================================================"
git status || true

echo
echo "Current branch:"
git branch --show-current || true

echo
echo "Recent log:"
git log --oneline --decorate --graph -40 || true

echo
echo "Final diffstat from base:"
git diff --stat "$BASE_COMMIT"..HEAD | tee /opt/r9700-vllm/build-info/vllm.final.diffstat.txt || true

echo
echo "============================================================"
echo "Summary"
echo "============================================================"
echo "Primary:   $SUMMARY_PRIMARY"
echo "Secondary: $SUMMARY_SECONDARY"
echo "SplitKV:   $SUMMARY_SPLITKV"
echo "Finished:  $(date -Is)"
echo "Log:       $LOG"

[ "$SUMMARY_PRIMARY" = "success" ] && [ "$SUMMARY_SECONDARY" = "success" ]
