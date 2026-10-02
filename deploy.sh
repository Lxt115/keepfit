#!/bin/sh
# 自动部署脚本：拉取最新代码并重建容器
set -e

cd "$(dirname "$0")"

echo "[deploy] $(date '+%Y-%m-%d %H:%M:%S') 开始部署"

# 拉取最新代码（强制与远程一致，避免本地改动冲突）
git fetch origin main
git reset --hard origin/main

# 重建并重启容器
docker compose up -d --build

# 清理悬空镜像，释放磁盘
docker image prune -f

echo "[deploy] $(date '+%Y-%m-%d %H:%M:%S') 部署完成"
