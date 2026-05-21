# A100 -> Ascend 910B 遥感大模型迁移完整指南

## 目录

1. [架构差异概览](#1-架构差异概览)
2. [容器环境部署](#2-容器环境部署)
3. [代码迁移适配](#3-代码迁移适配)
4. [模型权重迁移](#4-模型权重迁移)
5. [分布式训练迁移](#5-分布式训练迁移)
6. [OpenMMLab生态适配](#6-openmmlab生态适配)
7. [性能调优](#7-性能调优)
8. [常见问题排查](#8-常见问题排查)

---

## 1. 架构差异概览

| 维度 | NVIDIA A100 | 华为 Ascend 910B |
|------|-------------|-------------------|
| 架构 | Ampere (GPU) | Da Vinci (NPU) |
| 内存 | 40/80GB HBM2e | 64GB HBM2e |
| 算力 | 312 TFLOPS (FP16) | 376 TFLOPS (FP16) |
| 通信 | NCCL | HCCL |
| 框架 | CUDA + cuDNN | CANN + AscendCL |
| PyTorch | `torch.cuda` | `torch.npu` (torch_npu) |
| 混合精度 | `torch.cuda.amp` | `torch.npu.amp` |
| 容器基础 | `nvcr.io/nvidia/pytorch` | `ascendai/pytorch` |

### 关键映射关系

```
NVIDIA A100              →   华为 Ascend 910B
─────────────────────────────────────────────
torch.cuda               →   torch.npu
torch.cuda.is_available()→   torch.npu.is_available()
device='cuda'            →   device='npu'
.cuda()                  →   .npu()
CUDA_VISIBLE_DEVICES     →   ASCEND_RT_VISIBLE_DEVICES
NCCL backend             →   HCCL backend
torch.cuda.amp           →   torch.npu.amp
nvcr.io/.../pytorch      →   ascendai/pytorch
cuDNN                    →   Ascend CANN
```

---

## 2. 容器环境部署

### 前置条件

- 910B 服务器已安装 CANN 工具包 (推荐 8.0.RC2+)
- Docker 已安装并配置 NPU 设备透传
- 网络连接正常 (用于拉取基础镜像和安装依赖)

### 2.1 基础镜像选择

推荐使用华为官方提供的 PyTorch 基础镜像:

```bash
# 查看可用的基础镜像
docker pull ascendai/pytorch:2.1.0-cann8.0.RC2-py3.9

# 或查看华为昇腾社区的最新镜像
# 参考: https://ascendhub.huawei.com/
```

### 2.2 构建容器

在 910B 服务器上执行:

```bash
# 1. 将本项目所有文件拷贝到服务器
scp -r ./project/ user@910b-server:/workspace/ascend-rs/
cd /workspace/ascend-rs/

# 2. 执行构建
bash build.sh

# 3. 启动容器
bash run.sh
```

### 2.3 验证环境

进入容器后运行:

```bash
# 验证所有依赖
python /workspace/scripts/verify_env.py

# 快速 NPU 测试
python /workspace/scripts/quick_test.py

# 手动检查 NPU
npu-smi info
python -c "import torch; import torch_npu; print(torch.npu.device_count())"
```

---

## 3. 代码迁移适配

### 3.1 自动适配工具

本项目提供了自动代码适配工具:

```bash
# 扫描需要修改的代码
python migration/code_adapter.py --scan /path/to/your/project

# 自动修补 (生成 .bak 备份)
python migration/code_adapter.py --patch /path/to/your/project

# 预览修改 (不实际修改文件)
python migration/code_adapter.py --dry-run /path/to/your/project
```

### 3.2 手动修改清单

#### 设备相关 (必须修改)

```python
# Before (A100)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# After (910B)
device = torch.device('npu' if hasattr(torch, 'npu') and torch.npu.is_available() else 'cpu')
model = model.to(device)
```

#### 导入相关 (必须添加)

```python
# 在每个训练脚本开头添加
import torch
import torch_npu  # 必须在 distributed init 之前 import
```

#### 混合精度 (必须修改)

```python
# Before (A100)
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

# After (910B)
from torch.npu.amp import autocast, GradScaler
scaler = GradScaler()
```

#### 分布式训练 (必须修改)

```python
# Before (A100)
dist.init_process_group(backend='nccl', init_method='env://')

# After (910B)
dist.init_process_group(backend='hccl', init_method='env://')
```

### 3.3 需要特别注意的算子

| 算子/功能 | A100 状态 | 910B 状态 | 处理方式 |
|-----------|-----------|-----------|----------|
| Conv2d/Conv3d | 完全支持 | 完全支持 | 无需修改 |
| BatchNorm | 完全支持 | 完全支持 | 无需修改 |
| LayerNorm | 完全支持 | 完全支持 | 无需修改 |
| 可变形卷积 (DCN) | 完全支持 | 支持 (mmcv) | 使用 mmcv 版本 |
| RoI Align | 完全支持 | 支持 (mmcv) | 使用 mmcv 版本 |
| Flash Attention | 完全支持 | 有限支持 | 需要 CANN 8.0+ |
| 自定义 CUDA 算子 | 完全支持 | 需要改写 | 使用 CANN TBE 改写 |
| torch.compile | 支持 | 不支持 | 暂时移除 |

---

## 4. 模型权重迁移

### 4.1 直接加载 (推荐)

PyTorch state dict 格式完全兼容，通常可以直接加载:

```python
import torch
import torch_npu

# 加载 A100 训练的权重
checkpoint = torch.load('model_a100.pth', map_location='cpu')
model.load_state_dict(checkpoint['state_dict'])

# 转移到 NPU
model = model.to('npu')
```

### 4.2 使用模型转换工具

```bash
# 转换权重文件格式
python migration/model_converter.py \
    --input model_a100.pth \
    --output model_npu.pth \
    --dtype float16

# 验证转换后的权重
python migration/model_converter.py \
    --input model_npu.pth \
    --verify
```

### 4.3 ONNX 导出 (用于推理部署)

```bash
python migration/model_converter.py \
    --input model_a100.pth \
    --onnx model_npu.onnx \
    --input-shape 1,3,512,512
```

---

## 5. 分布式训练迁移

### 5.1 生成启动脚本

```bash
# 单机 8 卡训练
python migration/distributed_adapter.py \
    --cards 8 \
    --training-script train.py \
    --training-args "--config config.py --epochs 100" \
    --generate-launch-script train_8npu.sh

chmod +x train_8npu.sh
bash train_8npu.sh
```

### 5.2 环境变量对照

```bash
# A100 环境
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO

# 910B 环境 (替换为)
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HCCL_CONNECT_TIMEOUT=120
export HCCL_EXEC_TIMEOUT=0
```

### 5.3 多机训练

```bash
# Node 0
python migration/distributed_adapter.py \
    --cards 8 --nnodes 2 --node-rank 0 \
    --master-addr 192.168.1.100 \
    --generate-launch-script train_node0.sh

# Node 1
python migration/distributed_adapter.py \
    --cards 8 --nnodes 2 --node-rank 1 \
    --master-addr 192.168.1.100 \
    --generate-launch-script train_node1.sh
```

---

## 6. OpenMMLab 生态适配

### 6.1 支持状态

| 组件 | NPU 支持 | 说明 |
|------|----------|------|
| mmengine | 原生支持 | 无需修改 |
| mmcv | 原生支持 | 需安装 NPU 版本 |
| mmdetection | 原生支持 | 配置 device='npu' |
| mmsegmentation | 原生支持 | 配置 device='npu' |
| mmrotate | 支持 | 遥感旋转检测 |
| mmpretrain | 支持 | 预训练模型 |
| mmrazor | 支持 | 模型压缩 |
| mmdeploy | 需验证 | ONNX 导出需自定义后端 |

### 6.2 OpenMMLab 配置文件修改

```python
# config.py 中需要添加/修改的内容

# 设备设置
device = 'npu'  # 原 'cuda'

# 分布式训练 backend
dist_params = dict(backend='hccl')  # 原 'nccl'

# 混合精度
optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',  # 或具体数值
)

# DataLoader workers 调整 (NPU 环境建议)
train_dataloader = dict(
    batch_size=4,
    num_workers=4,  # 根据 NPU 节点的 CPU 核心数调整
    persistent_workers=True,
)
```

### 6.3 open-CD 变化检测适配

open-CD 是基于 MMDetection 的遥感变化检测工具箱，迁移步骤:

```bash
# 1. 确认 open-CD 已正确安装
pip list | grep open-cd

# 2. 修改配置文件中的 device
# configs/your_config.py:
device = 'npu'

# 3. 训练命令
bash tools/dist_train.sh configs/your_config.py 8 --launcher pytorch
```

---

## 7. 性能调优

### 7.1 内存优化

```python
# 启用 NPU 内存碎片整理
import os
os.environ['PYTORCH_NPU_ALLOC_CONF'] = 'expandable_segments:True'

# 或使用配置文件设置
```

### 7.2 通信优化

```bash
# HCCL 通信参数调优
export HCCL_BUFFSIZE=8       # 通信缓冲区大小 (默认2, 增大可能提升多卡性能)
export HCCL_CONNECT_TIMEOUT=120  # 连接超时 (秒)
export HCCL_EXEC_TIMEOUT=0   # 执行超时 (0=无限制)
```

### 7.3 数据加载优化

```python
# DataLoader 配置建议
train_dataloader = dict(
    batch_size=8,           # NPU 通常可使用较大 batch
    num_workers=8,          # 根据 CPU 核心数设置
    pin_memory=True,        # 启用锁页内存
    persistent_workers=True,
    prefetch_factor=2,
)
```

### 7.4 混合精度训练

```python
# 推荐配置
from torch.npu.amp import autocast, GradScaler

scaler = GradScaler()

for data in dataloader:
    with autocast():
        output = model(data)
        loss = criterion(output, target)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

---

## 8. 常见问题排查

### Q1: `torch.npu.is_available()` 返回 False

```bash
# 检查 1: CANN 是否正确安装
ls /usr/local/Ascend/driver

# 检查 2: NPU 设备是否可见
npu-smi info

# 检查 3: /dev/davinci* 设备权限
ls -la /dev/davinci*

# 检查 4: torch_npu 版本是否与 CANN 匹配
pip show torch-npu
cat /usr/local/Ascend/ascend-toolkit/latest/VERSION

# 检查 5: 环境变量
echo $LD_LIBRARY_PATH | tr ':' '\n' | grep Ascend
```

### Q2: 分布式训练 HCCL 初始化失败

```bash
# 设置调试模式
export HCCL_DEBUG=1
export ASCEND_SLOG_PRINT_TO_STDOUT=1

# 检查网卡
ip addr

# 确认节点间网络连通
ping <other_node_ip>
```

### Q3: 内存溢出 (OOM)

```python
# 解决方案 1: 减小 batch size
# 解决方案 2: 启用梯度累积
accumulative_steps = 4  # 等效 batch_size * accumulative_steps
# 解决方案 3: 使用梯度检查点
model.gradient_checkpointing_enable()
```

### Q4: 自定义 CUDA 算子不兼容

需要将其改写为 CANN TBE (Tensor Boost Engine) 算子:
1. 参考 CANN 文档: https://www.hiascend.com/document
2. 使用 TBE DSL 或 AI CPU 算子开发
3. 部分常见算子已在 mmcv 的 NPU 版本中提供

### Q5: 推理性能低于预期

```bash
# 使用 MindIE 进行推理优化
# 1. 导出 ONNX 模型
# 2. 使用 ATC 工具转换为 OM 格式
atc --model=model.onnx \
    --framework=5 \
    --output=model_npu \
    --soc_version=Ascend910B
```

---

## 附录 A: 完整迁移检查清单

- [ ] 910B 服务器 CANN 已安装并验证
- [ ] Docker 容器构建完成
- [ ] 容器内 `verify_env.py` 全部 PASS
- [ ] 容器内 `quick_test.py` 通过
- [ ] 代码中所有 `.cuda()` / `device='cuda'` 已替换为 NPU 版本
- [ ] `torch_npu` 已在每个训练脚本中 import
- [ ] 分布式 backend 已从 `nccl` 改为 `hccl`
- [ ] 混合精度已从 `torch.cuda.amp` 改为 `torch.npu.amp`
- [ ] A100 模型权重已加载验证
- [ ] 单卡训练测试通过
- [ ] 多卡训练测试通过
- [ ] 训练精度/损失与 A100 一致
- [ ] 推理性能满足要求

## 附录 B: 参考链接

- 昇腾社区: https://gitee.com/ascend
- 昇腾文档: https://www.hiascend.com/document
- CANN 文档: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition
- torch_npu: https://gitee.com/ascend/pytorch
- OpenMMLab 昇腾适配: https://gitee.com/ascend/modelzoo
- MindIE 推理: https://www.hiascend.com/en/software/mindie
