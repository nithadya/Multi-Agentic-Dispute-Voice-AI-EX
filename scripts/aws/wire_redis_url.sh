#!/usr/bin/env bash
# =============================================================================
# wire_redis_url.sh — read the ElastiCache endpoint from CloudFormation
# and write redis://<endpoint>:6379/0 to SSM Parameter Store.
#
# Run this once after `copilot env deploy --name dev`. Service manifests
# already reference /disputeai/dev/REDIS_URL as a secret; this script makes
# that parameter point at the real ElastiCache endpoint instead of the
# placeholder hostname we used at scaffold time.
#
# Usage:
#     ./scripts/aws/wire_redis_url.sh
#     ENV=prod ./scripts/aws/wire_redis_url.sh
# =============================================================================
set -euo pipefail
export MSYS_NO_PATHCONV=1
APP="disputeai"
ENV="${ENV:-dev}"
PROFILE="${AWS_PROFILE:-dispute-ai}"
REGION="${AWS_REGION:-us-west-2}"

echo "==> Reading ElastiCache endpoint from CloudFormation exports…"
ENDPOINT=$(aws cloudformation list-exports \
  --profile "$PROFILE" --region "$REGION" \
  --query "Exports[?Name=='${APP}-${ENV}-RedisEndpoint'].Value" \
  --output text)

if [ -z "$ENDPOINT" ] || [ "$ENDPOINT" = "None" ]; then
  echo "ERROR: ${APP}-${ENV}-RedisEndpoint export not found. Has 'copilot env deploy' finished?" >&2
  exit 1
fi

PORT=$(aws cloudformation list-exports \
  --profile "$PROFILE" --region "$REGION" \
  --query "Exports[?Name=='${APP}-${ENV}-RedisPort'].Value" \
  --output text)

REDIS_URL="redis://${ENDPOINT}:${PORT:-6379}/0"
echo "==> Endpoint: $REDIS_URL"

echo "==> Updating /disputeai/${ENV}/REDIS_URL in SSM…"
aws ssm put-parameter \
  --profile "$PROFILE" --region "$REGION" \
  --name "/disputeai/${ENV}/REDIS_URL" \
  --value "$REDIS_URL" \
  --type SecureString \
  --overwrite \
  --no-cli-pager >/dev/null

echo "==> Verifying…"
aws ssm get-parameter \
  --profile "$PROFILE" --region "$REGION" \
  --name "/disputeai/${ENV}/REDIS_URL" \
  --with-decryption \
  --query 'Parameter.Value' \
  --output text

echo ""
echo "Done. Services will pick up the new REDIS_URL on next deploy."
