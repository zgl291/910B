# Example training configuration for Ascend 910B
# Use this as a template for your OpenMMLab-based remote sensing models

# ---------------------------------------------------------------------------
# Device and distributed training
# ---------------------------------------------------------------------------
device = 'npu'  # Changed from 'cuda' to 'npu'
dist_params = dict(backend='hccl')  # Changed from 'nccl' to 'hccl'

# ---------------------------------------------------------------------------
# Model (example: ResNet-50 backbone for remote sensing classification)
# ---------------------------------------------------------------------------
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3,),
        style='pytorch',
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=10,  # Adjust for your dataset
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ),
)

# ---------------------------------------------------------------------------
# Data (example: remote sensing image dataset)
# ---------------------------------------------------------------------------
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(512, 512)),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackClsInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(512, 512)),
    dict(type='PackClsInputs'),
]

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CustomDataset',
        data_root='data/remote_sensing/',
        ann_file='train.txt',
        data_prefix='',
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CustomDataset',
        data_root='data/remote_sensing/',
        ann_file='val.txt',
        data_prefix='',
        pipeline=test_pipeline,
    ),
)

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
optim_wrapper = dict(
    type='AmpOptimWrapper',  # Mixed precision training
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001),
)

param_scheduler = dict(
    type='CosineAnnealingLR',
    T_max=100,
    eta_min=0.0001,
)

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=100, val_interval=5)
val_cfg = dict()
test_cfg = dict()

default_scope = 'mmpretrain'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=10, save_best='auto'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='VisualizationHook', enable=False),
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='hccl'),
)

visualizer = dict(type='UniversalVisualizer')

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
log_level = 'INFO'
load_from = None
resume = False
