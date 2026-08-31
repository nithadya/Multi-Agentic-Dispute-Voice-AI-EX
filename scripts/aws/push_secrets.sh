#!/usr/bin/env bash
# =============================================================================
# push_secrets.sh — sync .env → AWS SSM Parameter Store
#
# Reads .env at repo root, pushes each KEY=VALUE as an SSM SecureString
# under /disputeai/dev/<KEY>. The Copilot service manifests reference these
# parameters in their `secrets:` blocks, so the ECS task role decrypts +
# injects them as env vars at container start.
#
# Idempotent — each parameter is overwritten on every run, so this script
# is also how you rotate keys: edit .env, re-run, redeploy.
#
# Usage:
#     ./scripts/aws/push_secrets.sh           # uses /disputeai/dev/
#     ENV=prod ./scripts/aws/push_secrets.sh  # uses /disputeai/prod/
# =============================================================================
set -euo pipefail

# Disable path conversion in MSYS (Git Bash on Windows) to prevent
# SSM parameter paths (e.g. /disputeai/dev/...) from being converted to Windows paths.
export MSYS_NO_PATHCONV=1

ENV="${ENV:-dev}"
PREFIX="/copilot/disputeai/${ENV}/secrets"
PROFILE="${AWS_PROFILE:-dispute-ai}"
REGION="${AWS_REGION:-us-west-2}"

cd "$(dirname "$0")/../.."

if [ ! -f .env ]; then
  echo "ERROR: .env not found at $(pwd)/.env"
  exit 1
fi

COUNT=0
SKIPPED=0
while IFS='=' read -r raw_key raw_val; do
  # Skip blank lines and comments
  case "$raw_key" in
    ""|\#*) continue ;;
  esac
  key="$(echo "$raw_key" | xargs)"
  val="$raw_val"
  # Strip carriage returns (CRLF fix)
  val="${val%$'\r'}"
  val="${val%\"}"; val="${val#\"}"
  val="${val%\'}"; val="${val#\'}"
  if [ -z "$val" ]; then
    echo "  skip empty: $key"
    SKIPPED=$((SKIPPED + 1))
    continue
  fi
  param_name="${PREFIX}/${key}"
  aws ssm put-parameter \
    --profile "$PROFILE" --region "$REGION" \
    --name "$param_name" \
    --value "$val" \
    --type SecureString \
    --overwrite \
    --tags "Key=copilot-application,Value=disputeai" "Key=copilot-environment,Value=$ENV" \
    --no-cli-pager >/dev/null 2>&1 \
    && { echo "  ✓ $param_name"; COUNT=$((COUNT + 1)); } \
    || echo "  ✗ FAILED: $param_name"
done < .env

echo ""
echo "Done. Wrote $COUNT secrets to $PREFIX (skipped $SKIPPED empty)."
echo "Verify with:  aws ssm get-parameters-by-path --path $PREFIX --profile $PROFILE --region $REGION --query 'Parameters[].Name' --output table"
