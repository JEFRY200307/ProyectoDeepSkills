#!/bin/bash
# EC2 "user data": extracts the tsfresh features of the 134 TSC datasets, uploads them
# to S3 and powers the machine off (with shutdown behavior = Terminate, it is deleted).
# Guide: docs/aws_tsfresh.md. Only BUCKET needs editing.
#
# Instance: c7a.8xlarge (or c6i.8xlarge), Amazon Linux 2023, 40 GB disk,
# IAM role with access to BUCKET, shutdown behavior = Terminate.

BUCKET="CAMBIAR-por-el-nombre-de-tu-bucket"
BRANCH="phase-1"
MAX_HOURS=30   # hard cap: the machine stops after this even if not finished

export HOME=/root
WORK=/opt/ts2tabpfn
LOG=/var/log/tsfresh.log
mkdir -p "$WORK" && cd "$WORK" || exit 1
exec > >(tee -a "$LOG") 2>&1

upload() {
  aws s3 sync "$WORK/features" "s3://$BUCKET/features" --only-show-errors
  aws s3 cp "$LOG" "s3://$BUCKET/logs/tsfresh.log" --only-show-errors
}

finish() {
  status=$?
  echo "[$(date -u +%FT%TZ)] finishing, exit status $status"
  upload
  if [ -d "$WORK/features/tsfresh" ]; then
    (cd "$WORK/features" && zip -qr "$WORK/tsfresh_features.zip" tsfresh)
    aws s3 cp "$WORK/tsfresh_features.zip" "s3://$BUCKET/tsfresh_features.zip" --only-show-errors
  fi
  aws s3 cp "$LOG" "s3://$BUCKET/logs/tsfresh.log" --only-show-errors
  shutdown -h now
}
trap finish EXIT

echo "[$(date -u +%FT%TZ)] start on $(nproc) vCPU"
dnf install -y -q git unzip zip python3-pip
pip3 install -q uv==0.9.5
git clone -q -b "$BRANCH" https://github.com/JEFRY200307/ProyectoDeepSkills.git repo
git -C repo log --oneline -1
uv venv -q --python 3.10 venv
uv pip install -q --python venv/bin/python -r repo/requirements.txt
echo "[$(date -u +%FT%TZ)] environment ready"

# Data: from the bucket if you uploaded it there, otherwise from the authors' Google Drive.
if aws s3 ls "s3://$BUCKET/tsc_data.zip" > /dev/null 2>&1; then
  aws s3 cp "s3://$BUCKET/tsc_data.zip" tsc_data.zip --only-show-errors
else
  venv/bin/python -m gdown 1p8D-haDDwPKloKMBQCQHgENYHVsjixR4 -O tsc_data.zip
fi
unzip -q tsc_data.zip -d data
echo "[$(date -u +%FT%TZ)] data ready: $(ls data | wc -l) folders"

# Resume: features already in the bucket are not extracted again.
aws s3 sync "s3://$BUCKET/features" features --only-show-errors

# Upload progress and the log every 10 minutes.
( while true; do sleep 600; upload; done ) &

timeout "${MAX_HOURS}h" venv/bin/python repo/scripts/extract_tsfresh.py \
  --data-dir data --features-dir features --n-jobs "$(nproc)"
