#!/usr/bin/env bash
# Automation merge queue: `select` picks one eligible Copilot automation PR, `finish` merges or reports it.
# shellcheck disable=SC2016,SC2153 # jq variables are single-quoted; PR, BASE, BASE_SHA, HEAD_SHA, METHOD and VERDICT come from the workflow.
set -euo pipefail

: "${REPO:?}" "${GITHUB_OUTPUT:=/dev/null}" "${RUN_URL:=}"
DRY_RUN=${DRY_RUN:-true}
MARK='<!-- smartdnn-merge-queue-status -->'
SYNC_MARK='<!-- smartdnn-pr: sync -->'
BLOCKING_LABELS='hold do-not-merge tests-failing review-feedback needs-test needs-research needs-human'
MAX_CANDIDATES=${MAX_CANDIDATES:-25}
MAX_FAILURES=2
THREADS_QUERY='query($owner:String!,$name:String!,$n:Int!){repository(owner:$owner,name:$name){pullRequest(number:$n){reviewThreads(first:100){nodes{isResolved comments(first:1){nodes{author{__typename} body}}}}}}}'

out() { echo "$1=$2" >> "$GITHUB_OUTPUT"; }
default_branch() { gh api "repos/$REPO" --jq .default_branch; }
branch_sha() { gh api "repos/$REPO/git/ref/heads/$1" --jq .object.sha; }
closing_refs() {
  grep -oiE '(^|[^[:alnum:]_])(close[sd]?|fix(e[sd])?|resolve[sd]?):?[[:space:]]+#[0-9]+' <<< "$1" | grep -oE '[0-9]+$' | sort -un || true
}

# load_freeze: sets default, blockers (open merge-blocker issues) and allowed (blockers plus their sub-issues).
load_freeze() {
  local b
  default=$(default_branch)
  blockers=$(gh api "repos/$REPO/issues?state=open&labels=merge-blocker&per_page=100" --jq '.[] | select(.pull_request | not) | .number')
  allowed=$blockers
  for b in $blockers; do allowed+=$'\n'$(gh api --paginate "repos/$REPO/issues/$b/sub_issues?per_page=100" --jq '.[].number'); done
}
fixes_blocker() {
  local r
  for r in $(closing_refs "$1"); do grep -qxF "$r" <<< "$allowed" && return 0; done
  return 1
}

# Loads "<kind>\t<comment id>\t<base64 body>" rows for queue status and Branch Tester comments.
load_comments() {
  comments=$(gh api --paginate "repos/$REPO/issues/$1/comments?per_page=100" --jq '.[]
    | (if .user.login == "github-actions[bot]" and (.body | startswith("<!-- smartdnn-merge-queue-status -->")) then "status"
       elif .user.login == "Copilot" and (.body | startswith("<!-- smartdnn-branch-tester -->")) then "tester"
       else null end) as $k
    | select($k) | [$k, (.id | tostring), (.body | @base64)] | @tsv')
}
latest() { awk -F'\t' -v k="$1" -v f="$2" '$1 == k {v = $f} END {print v}' <<< "$comments"; }
failures() { latest status 3 | { base64 -d 2>/dev/null || true; } | grep -oE '<!-- failures:[0-9a-f ]* -->' | grep -oE '[0-9a-f]{40}' || true; }

# set_status PR TEXT [FAILED_SHA]: creates or edits the single status comment, keeping recorded failures.
set_status() {
  local pr=$1 text=$2 add_fail=${3:-} id old fails body
  load_comments "$pr"
  id=$(latest status 2); old=""
  [[ -n $id ]] && old=$(base64 -d <<< "$(latest status 3)")
  fails=$(failures | tr '\n' ' ')
  [[ -n $add_fail ]] && fails+="$add_fail "
  body="$MARK"$'\n'"$text"
  [[ -n ${fails// } ]] && body+=$'\n'"<!-- failures: ${fails% } -->"
  [[ $body == "$old" ]] && return 0
  echo "#$pr: $text"
  [[ $DRY_RUN == true ]] && return 0
  if [[ -n $id ]]; then
    gh api -X PATCH "repos/$REPO/issues/comments/$id" -f body="$body" > /dev/null
  else
    gh api "repos/$REPO/issues/$pr/comments" -f body="$body" > /dev/null
  fi
}

# evaluate PR: returns 0 when eligible (sets head, base, base_sha, method, mode); else sets reason ("" means skip quietly).
evaluate() {
  local n=$1 pr labels l tester reviews threads files body behind
  reason=""; mode=""
  pr=$(gh api "repos/$REPO/pulls/$n")
  head=$(jq -r .head.sha <<< "$pr"); base=$(jq -r .base.ref <<< "$pr"); body=$(jq -r '.body // ""' <<< "$pr")
  labels=" $(jq -r '[.labels[].name] | join(" ")' <<< "$pr") "
  [[ $(jq -r .state <<< "$pr") == open && $labels == *" automation-pr "* && $labels == *" tests-passing "* ]] || return 1
  for l in $BLOCKING_LABELS; do
    if [[ $labels == *" $l "* ]]; then reason="the $l label is set"; return 1; fi
  done
  if [[ $(jq -r .title <<< "$pr") == "[WIP]"* ]]; then reason="the title is marked [WIP]"; return 1; fi
  if [[ $base == "$default" && -n $blockers ]] && ! fixes_blocker "$body"; then
    reason="$default is frozen by merge-blocker issue(s) ${blockers//$'\n'/ } and this PR closes none of them or their sub-issues"; return 1
  fi
  case $(jq -r .mergeable <<< "$pr") in
    true) ;;
    false) reason="it has a merge conflict with $base"; return 1 ;;
    *) return 1 ;;
  esac
  load_comments "$n"
  if (( $(failures | grep -cxF "$head" || true) >= MAX_FAILURES )); then
    reason="the queue failed $MAX_FAILURES times at head $head, so it waits for a new commit"; return 1
  fi
  tester=$(latest tester 3)
  if [[ -z $tester || $(base64 -d <<< "$tester") != *"$head"* ]]; then reason="the latest Branch Tester result is not for head $head"; return 1; fi
  reviews=$(gh api --paginate "repos/$REPO/pulls/$n/reviews?per_page=100" --jq '.[] | select(.user.login == "copilot-swe-agent[bot]" or .user.login == "Copilot") | .id' | wc -l)
  if (( reviews == 0 )); then reason="it has no PR Reviewer review yet"; return 1; fi
  threads=$(gh api graphql -f owner="${REPO%/*}" -f name="${REPO#*/}" -F n="$n" -f query="$THREADS_QUERY" \
    --jq '[.data.repository.pullRequest.reviewThreads.nodes[] | select(.isResolved | not) | .comments.nodes[0] | select(.author.__typename == "User" or (.body | startswith("Blocking:")))] | length')
  if (( threads > 0 )); then reason="it has $threads unresolved human or Blocking review thread(s)"; return 1; fi
  files=$(gh api --paginate "repos/$REPO/pulls/$n/files?per_page=100" --jq '.[] | .filename, (.previous_filename // empty)')
  if grep -q '^\.github/' <<< "$files"; then reason="it changes files under .github/, which the queue never merges"; return 1; fi
  base_sha=$(branch_sha "$base")
  method=squash; [[ $body == *"$SYNC_MARK"* ]] && method=merge
  if [[ $base == "$default" ]]; then mode="test"; return 0; fi
  behind=$(gh api "repos/$REPO/compare/$base_sha...$head" --jq .behind_by)
  if (( behind > 0 )); then mode=update; reason="it is $behind commit(s) behind $base, so the queue updates the branch for the Branch Tester to retest"; return 0; fi
  mode=evidence
}

cmd_select() {
  local n rank b body cand
  load_freeze
  cand=$(mktemp)
  gh api --paginate "repos/$REPO/pulls?state=open&per_page=100" --jq '.[]
    | select(.user.login == "Copilot" and .user.type == "Bot" and .head.repo.full_name == .base.repo.full_name and (.head.ref | startswith("copilot/")))
    | ([.labels[].name]) as $l | select(($l | index("automation-pr")) and ($l | index("tests-passing")))
    | [(if ($l | index("P1")) then 1 elif ($l | index("P2")) then 2 elif ($l | index("P3")) then 3 else 4 end), .number, .base.ref, ((.body // "") | @base64)] | @tsv' \
    | while IFS=$'\t' read -r rank n b body; do
        # Merge-blocker fixes go first so a frozen default branch can recover.
        if [[ $b == "$default" && -n $blockers ]] && fixes_blocker "$(base64 -d <<< "$body")"; then rank=0; fi
        printf '%s\t%s\n' "$rank" "$n"
      done | sort -n -k1,1 -k2,2 | awk -v m="$MAX_CANDIDATES" 'NR <= m {print $2}' > "$cand"
  while read -r n <&3; do
    if ! evaluate "$n"; then
      [[ -n $reason ]] && set_status "$n" "Waiting: $reason."
      continue
    fi
    if [[ $mode == update ]]; then
      set_status "$n" "Waiting: $reason."
      [[ $DRY_RUN == true ]] || gh api -X PUT "repos/$REPO/pulls/$n/update-branch" -f expected_head_sha="$head" > /dev/null || echo "::warning::could not update #$n"
      continue
    fi
    echo "Selected #$n ($mode, $method) at $head onto $base $base_sha"
    out pr "$n"; out head "$head"; out base "$base"; out base_sha "$base_sha"; out method "$method"; out mode "$mode"
    [[ $mode == test ]] && set_status "$n" "Testing head $head merged onto $base at $base_sha in $RUN_URL"
    return 0
  done 3< "$cand"
  echo "No eligible pull request."
}

after_merge() {
  local r state
  if [[ $BASE == "$(default_branch)" ]]; then
    if [[ -z ${MERGE_TOKEN:-} ]]; then gh workflow run ci.yml -R "$REPO" --ref "$BASE" || echo "::warning::could not start ci.yml"; fi
    return 0
  fi
  for r in $(closing_refs "$(jq -r '.body // ""' <<< "$pr")"); do
    state=$(gh api "repos/$REPO/issues/$r" --jq 'if .pull_request then "pr" else .state end') || continue
    [[ $state == open ]] || continue
    gh api "repos/$REPO/issues/$r/comments" -f body="Closed by #$PR, merged into $BASE." > /dev/null || true
    gh api -X PATCH "repos/$REPO/issues/$r" -f state=closed -f state_reason=completed > /dev/null || echo "::warning::could not close #$r"
  done
}

merge_pr() {
  local result
  load_freeze
  if ! evaluate "$PR"; then
    set_status "$PR" "Did not merge: when rechecked after testing, ${reason:-GitHub had not computed mergeability yet}."; return 0
  fi
  if [[ $DRY_RUN == true ]]; then echo "Dry run: would merge #$PR ($METHOD) at $HEAD_SHA into $BASE"; return 0; fi
  if [[ $(jq -r .draft <<< "$pr") == true ]] && ! gh pr ready "$PR" -R "$REPO"; then
    set_status "$PR" "Did not merge: could not mark the pull request ready for review." "$HEAD_SHA"; return 0
  fi
  if ! result=$(GH_TOKEN=${MERGE_TOKEN:-$GH_TOKEN} gh api -X PUT "repos/$REPO/pulls/$PR/merge" -f merge_method="$METHOD" -f sha="$HEAD_SHA" 2>&1); then
    if [[ $result == *"Base branch was modified"* || $result == *"Head branch was modified"* ]]; then
      set_status "$PR" "Did not merge: a branch moved while merging, so it will be retested."
    else
      set_status "$PR" "Did not merge: GitHub rejected merging head $HEAD_SHA: ${result:0:300}" "$HEAD_SHA"
    fi
    return 0
  fi
  set_status "$PR" "Merged head $HEAD_SHA into $BASE by $METHOD in $RUN_URL"
  after_merge
}

fail_pr() {
  local log="" what=failed
  [[ $VERDICT == timeout ]] && what="timed out"
  [[ -n ${LOG_B64:-} ]] && log=$(base64 -d <<< "$LOG_B64" | sed "s/\`\`\`/'''/g")
  set_status "$PR" "Did not merge: the unit tests $what for head $HEAD_SHA merged onto $BASE at $BASE_SHA in $RUN_URL
<details><summary>Log tail</summary>

\`\`\`text
$log
\`\`\`

</details>" "$HEAD_SHA"
  if [[ $DRY_RUN == true ]]; then echo "Dry run: would relabel #$PR tests-failing"; return 0; fi
  gh api -X DELETE "repos/$REPO/issues/$PR/labels/tests-passing" > /dev/null || true
  gh api "repos/$REPO/issues/$PR/labels" -f "labels[]=tests-failing" > /dev/null
}

cmd_finish() {
  : "${PR:?}" "${HEAD_SHA:?}" "${BASE:?}" "${BASE_SHA:?}" "${METHOD:?}" "${VERDICT:?}"
  local labels l
  pr=$(gh api "repos/$REPO/pulls/$PR")
  if [[ $(jq -r .state <<< "$pr") != open ]]; then echo "#$PR is no longer open."; return 0; fi
  labels=" $(jq -r '[.labels[].name] | join(" ")' <<< "$pr") "
  if [[ $(jq -r .head.sha <<< "$pr") != "$HEAD_SHA" ]]; then set_status "$PR" "Did not merge: the head moved during testing, so it will be re-evaluated."; return 0; fi
  if [[ $(branch_sha "$BASE") != "$BASE_SHA" ]]; then set_status "$PR" "Did not merge: $BASE moved during testing, so it will be retested."; return 0; fi
  for l in $BLOCKING_LABELS; do
    if [[ $labels == *" $l "* ]]; then set_status "$PR" "Did not merge: the $l label was added during testing."; return 0; fi
  done
  if [[ $labels != *" tests-passing "* ]]; then set_status "$PR" "Did not merge: the tests-passing label was removed during testing."; return 0; fi
  case $VERDICT in
    pass) merge_pr ;;
    fail|timeout) fail_pr ;;
    stale) set_status "$PR" "Did not merge: head $HEAD_SHA was not found when testing, so it will be re-evaluated." ;;
    conflict) set_status "$PR" "Did not merge: head $HEAD_SHA does not merge cleanly into $BASE at $BASE_SHA." "$HEAD_SHA" ;;
    *) echo "::error::unknown verdict '$VERDICT'"; return 1 ;;
  esac
}

case ${1:-} in
  select) cmd_select ;;
  finish) cmd_finish ;;
  *) echo "usage: $0 select|finish" >&2; exit 2 ;;
esac
