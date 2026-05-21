# GitHub Actions 云端构建 - 先决条件与操作指南

## 你需要提供的先决条件

### 1. GitHub 账号
- 免费账号即可，注册地址: https://github.com/signup
- 如果你已有账号，跳过

### 2. 创建一个空仓库
- 访问: https://github.com/new
- 仓库名: 例如 `ascend-910b-env`
- 设为 **Private** (如果不想公开) 或 **Public**
- **重要**: 不要勾选 "Add README"、".gitignore"、"Choose a license"
- 点击 "Create repository"

### 3. 生成 Personal Access Token (PAT)
用于后续从 GHCR 拉取镜像，步骤:

1. 访问: https://github.com/settings/tokens
2. 点击 **"Generate new token (classic)"**
3. 名称: `ascend-image-pull`
4. 过期时间: 建议 90 天
5. 勾选以下权限:
   - `read:packages` (必须 - 拉取镜像)
   - `write:packages` (必须 - 推送/覆盖镜像)
6. 点击 "Generate token"
7. **立即复制生成的 Token**（只显示一次）

### 4. git 命令行工具
- 如果你电脑上已安装 git，跳过
- 未安装的话: `winget install Git.Git`（Windows 管理员终端运行）

## 构建原理

```
你的电脑 (Windows)          GitHub Actions (云端)           910B 服务器
     │                           │                           │
     │ 1. git push               │                           │
     ├────────────────────────>│                           │
     │                           │ 2. 拉取 ascendai/pytorch  │
     │                           │    基础镜像 (Docker Hub)   │
     │                           │ 3. QEMU 模拟 ARM64 构建    │
     │                           │ 4. 推送结果到 GHCR        │
     │                           ├────────────────────────>│
     │                           │                           │ 5. docker login GHCR
     │<──────────────────────────┼───────────────────────────┤
     │  在 Actions 页面查看状态   │                           │ 6. docker pull 镜像
     │                           │                           │ 7. docker run 容器
```

## 操作步骤

### Step 1: 推送代码到 GitHub

```bash
cd D:/AI_Code/hw

git init
git add .
git commit -m "init: ascend 910b remote sensing container"

# 替换为你的实际仓库地址
git remote add origin https://github.com/<你的用户名>/<仓库名>.git
git push -u origin main
```

推送后 GitHub Actions **自动触发**构建。

### Step 2: 等待构建完成

- 访问: `https://github.com/<你的用户名>/<仓库名>/actions`
- 预计耗时: **30-60 分钟** (QEMU ARM64 模拟较慢)
- 如果首次失败，通常是因为磁盘空间不足，可重试一次

### Step 3: 在 910B 服务器上拉取镜像

```bash
# 登录 GHCR (替换为你的 PAT)
docker login ghcr.io -u <你的GitHub用户名> -p <你的PAT>

# 拉取镜像
docker pull ghcr.io/<你的用户名>/<仓库名>:latest

# 或拉取 ARM64 专用标签
docker pull ghcr.io/<你的用户名>/<仓库名>:arm64-latest
```

### Step 4: 启动容器

```bash
docker run -it --rm --name ascend-rs \
  --device=/dev/davinci0 \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/:ro \
  -v $(pwd)/workspace:/workspace \
  --network=host --ipc=host \
  ghcr.io/<你的用户名>/<仓库名>:latest \
  bash
```

## 构建产物

| 产物 | 用途 | 标签 |
|------|------|------|
| ARM64 生产镜像 | 910B 服务器实际使用 | `:latest` / `:arm64-latest` |
| x86 CPU 测试镜像 | 本地验证 Python 包 | `:cpu-test-latest` |

## 注意事项

1. **QEMU 构建速度**: 首次构建较慢，但 GitHub Actions 有 6 小时超时，通常够用
2. **缓存优化**: 第二次构建会快很多（Docker 层缓存）
3. **镜像大小**: 最终镜像约 10-15 GB，确保 910B 服务器有足够磁盘空间
4. **基础镜像来源**: `ascendai/pytorch` 来自 Docker Hub，公开可用，无需认证

## 费用

- GitHub Actions 免费额度: 每月 2000 分钟
- 本次构建预计消耗: 30-60 分钟
- GitHub Packages 存储: 免费 500MB（如果镜像超了这个大小，可能需要清理旧的镜像版本）

如果镜像超过 500MB 限制，可以在 Settings > Actions > General > Workflow permissions 中开启自动清理，或者每次构建后删除旧版本。
