#!/usr/bin/env bash
# =============================================================================
# deploy_redis.sh — provision ElastiCache after `copilot env deploy`.
#
# Reads VPC ID + public subnet IDs from the Copilot env stack's CloudFormation
# exports, deploys scripts/aws/cfn/redis-cluster.yml, waits for the cluster
# to be available, and writes the endpoint into SSM Parameter Store at
# /disputeai/dev/REDIS_URL.
#
# Idempotent — re-running just updates the existing stack (no-op if nothing
# changed).
# =============================================================================
set -euo pipefail

# Disable path conversion in MSYS (Git Bash on Windows) to prevent
# "/disputeai/dev/REDIS_URL" from being converted to a Windows path.
export MSYS_NO_PATHCONV=1

APP="disputeai"
ENV="${ENV:-dev}"
PROFILE="${AWS_PROFILE:-dispute-ai}"
REGION="${AWS_REGION:-us-west-2}"
STACK_NAME="disputeai-${ENV}-redis"
TEMPLATE="$(dirname "$0")/cfn/redis-cluster.yml"

cd "$(dirname "$0")/../.."

echo "==> Reading VPC + subnet IDs from Copilot env stack exports…"
VPC_ID=$(aws cloudformation list-exports \
  --profile "$PROFILE" --region "$REGION" \
  --query "Exports[?Name=='${APP}-${ENV}-VpcId'].Value" \
  --output text)
SUBNETS=$(aws cloudformation list-exports \
  --profile "$PROFILE" --region "$REGION" \
  --query "Exports[?Name=='${APP}-${ENV}-PublicSubnets'].Value" \
  --output text)

if [ -z "$VPC_ID" ] || [ "$VPC_ID" = "None" ]; then
  echo "ERROR: ${APP}-${ENV}-VpcId not found. Did 'copilot env deploy --name $ENV' finish?" >&2
  exit 1
fi
if [ -z "$SUBNETS" ] || [ "$SUBNETS" = "None" ]; then
  echo "ERROR: ${APP}-${ENV}-PublicSubnets not found." >&2
  exit 1
fi

echo "    VPC:     $VPC_ID"
echo "    Subnets: $SUBNETS"

echo ""
echo "==> Deploying $STACK_NAME (cache.t4g.micro, ~\$13/mo)…"
aws cloudformation deploy \
  --profile "$PROFILE" --region "$REGION" \
  --stack-name "$STACK_NAME" \
  --template-file "$TEMPLATE" \
  --parameter-overrides \
      VpcId="$VPC_ID" \
      SubnetIds="$SUBNETS" \
      EnvName="$ENV" \
  --capabilities CAPABILITY_IAM \
  --no-fail-on-empty-changeset

echo ""
echo "==> Reading cluster endpoint…"
ENDPOINT=$(aws cloudformation describe-stacks \
  --profile "$PROFILE" --region "$REGION" \
  --stack-name "$STACK_NAME" \
  --query "Stacks[0].Outputs[?OutputKey=='RedisEndpoint'].OutputValue" \
  --output text)
PORT=$(aws cloudformation describe-stacks \
  --profile "$PROFILE" --region "$REGION" \
  --stack-name "$STACK_NAME" \
  --query "Stacks[0].Outputs[?OutputKey=='RedisPort'].OutputValue" \
  --output text)

REDIS_URL="redis://${ENDPOINT}:${PORT}/0"
echo "    Endpoint: $REDIS_URL"

echo ""
echo "==> Updating /disputeai/${ENV}/REDIS_URL in SSM…"
aws ssm put-parameter \
  --profile "$PROFILE" --region "$REGION" \
  --name "/disputeai/${ENV}/REDIS_URL" \
  --value "$REDIS_URL" \
  --type SecureString \
  --overwrite \
  --no-cli-pager >/dev/null

echo ""
echo "✅ Redis ready. REDIS_URL stored in SSM."
echo "   Next: copilot svc deploy --name api/worker/voice"
