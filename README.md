# Ascend 910B 遥感大模型容器环境

本项目为华为 Ascend 910B 服务器构建完整的容器化 AI 训练环境，用于将 NVIDIA A100 上训练的遥感图像大模型迁移到昇腾平台。

## 构建方式

### 方式一：GitHub Actions 云端构建（推荐）

无需本地 Docker 和 910B 硬件，利用 GitHub 免费 CI 算力构建：

```bash
# 1. 在 GitHub 上创建仓库，推送本项目
git init
git add .
git commit -m "init: ascend 910b remote sensing container"
git remote add origin https://github.com/YOUR-USER/YOUR-REPO.git
git push -u origin main

# 2. 触发构建
# 推送后 GitHub Actions 自动触发，或手动触发：
# 访问: https://github.com/YOUR-USER/YOUR-REPO/actions

# 3. 构建完成后，从 GHCR 拉取镜像
bash scripts/pull_image.sh

# 或将镜像拷贝到 910B 服务器后加载
# docker load -i ascend-rs-image.tar.gz
```

**构建产物**:
- `linux/arm64` 生产镜像 → `ghcr.io/<your-user>/<your-repo>:latest`
- `linux/amd64` CPU 测试镜像 → `ghcr.io/<your-user>/<your-repo>:cpu-test-latest`
- 如未推送 registry，产物可通过 Actions 页面下载 `.tar.gz`

**GitHub PAT 权限**: 需要 `read:packages` 和 `write:packages` scope，在 https://github.com/settings/tokens 创建。

### 方式二：在 910B 服务器上本地构建

```bash
# 1. 将本项目拷贝到 910B 服务器
scp -r . user@910b-server:/workspace/ascend-rs/
cd /workspace/ascend-rs/

# 2. 服务器环境准备（首次）
bash scripts/setup_910b_server.sh

# 3. 构建容器镜像
bash build.sh

# 4. 启动容器
bash run.sh

# 5. 验证环境（容器内）
python scripts/verify_env.py
python scripts/quick_test.py
```

### 方式三：本地 Windows Docker Desktop（仅 CPU 测试）

```powershell
# 需要 Docker Desktop 已安装
docker build -f Dockerfile.local-test -t ascend-rs-test:local .
docker run --rm -it ascend-rs-test:local
```

## 项目结构

```
├── Dockerfile                    # 生产容器构建文件 (多阶段)
├── Dockerfile.production         # 生产容器 (单阶段，推荐)
├── Dockerfile.local-test         # 本地测试容器 (x86 CPU only)
├── docker-compose.yml            # 容器编排配置
├── build.sh                      # 构建脚本
├── run.sh                        # 运行脚本
├── .dockerignore                 # Docker 构建排除规则
├── requirements/                 # Python 依赖列表
│   ├── base.txt                  # 基础依赖 (numpy, pyyaml, etc.)
│   ├── image-processing.txt      # 图像处理 (opencv, gdal, etc.)
│   ├── remote-sensing.txt        # 遥感相关 (shapely, pyproj, etc.)
│   ├── scientific.txt            # 科学计算 (scipy, networkx)
│   ├── distributed.txt           # 分布式训练 (horovod, onnx)
│   └── ai-framework.txt          # AI 框架 (timm, thop)
├── scripts/
│   ├── verify_env.py             # 环境验证脚本
│   ├── quick_test.py             # NPU 快速测试
│   └── setup_910b_server.sh      # 910B 服务器初始化脚本
├── migration/                    # 模型迁移工具
│   ├── code_adapter.py           # 代码自动适配 (CUDA -> NPU)
│   ├── model_converter.py        # 模型权重转换
│   └── distributed_adapter.py    # 分布式训练配置生成
├── config/
│   └── sample_train_config.py    # OpenMMLab 训练配置模板
└── docs/
    └── MIGRATION_GUIDE.md        # 完整迁移指南
```

## 迁移流程

1. **构建镜像** → GitHub Actions 云端构建 或 910B 本地构建
2. **拉取镜像** → `bash scripts/pull_image.sh` 或 `docker load -i *.tar.gz`
3. **代码适配** → `python migration/code_adapter.py --patch /your/project`
4. **权重转换** → `python migration/model_converter.py --input model.pth --output model_npu.pth`
5. **训练验证** → 单卡测试 → 多卡测试 → 正式训练

详细步骤请参考 [docs/MIGRATION_GUIDE.md](docs/MIGRATION_GUIDE.md)

## 环境要求

- **硬件**: 华为 Ascend 910B 服务器 (至少 1 卡)
- **系统**: 银河麒麟 OS / Ubuntu 20.04+ / CentOS 7+
- **软件**: CANN 8.0.RC2+, Docker 20.10+
- **网络**: 用于拉取基础镜像和安装 Python 包

## 依赖包清单

详见 [Req.docx](Req.docx) 和本项目 `requirements/` 目录。

已覆盖:
- 基础编译: cmake, ninja, GCC+, setuptools
- 基础库: numpy, pyyaml, pillow, tqdm, loguru 等 20+ 包
- 图像处理: gdal, opencv, matplotlib, scikit-image, albumentations 等
- AI 框架: torch_npu, mmcv, mmdetection, mmsegmentation, ultralytics 等
- 遥感专用: open-CD, mmrotate, shapely, pyproj, pyshp 等
- 分布式: HCCL, horovod, onnx
- 编程环境: python 3.9+, conda, perl, java, vim, ssh

## 故障排查

详见 [docs/MIGRATION_GUIDE.md#8-常见问题排查](docs/MIGRATION_GUIDE.md)

## 参考链接

- [昇腾社区](https://gitee.com/ascend)
- [CANN 文档](https://www.hiascend.com/document)
- [torch_npu](https://gitee.com/ascend/pytorch)
- [OpenMMLab](https://github.com/open-mmlab)
