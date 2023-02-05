_base_ = [
    '../_base_/models/mobilenet_v2_1x.py',
    '../_base_/schedules/imagenet_bs256_epochstep.py',
    '../_base_/default_runtime.py'
]

# ---- 模型配置 ----
# 这里使用 init_cfg 来加载预训练模型，通过这种方式，只有主干网络的权重会被加载。
# 另外还修改了分类头部的 num_classes 来匹配我们的数据集。

model = dict(
    backbone=dict(
        init_cfg = dict(
            type='Pretrained', 
            checkpoint='https://download.openmmlab.com/mmclassification/v0/mobilenet_v2/mobilenet_v2_batch256_imagenet_20200708-3b2dc3af.pth', 
            prefix='backbone')
    ),
    head=dict(
        num_classes=2,
        topk = (1, )
    ))

# ---- 数据集配置 ----
# 我们已经将数据集重新组织为 ImageNet 格式
dataset_type = 'ImageNet'
img_norm_cfg = dict(
     mean=[124.508, 116.050, 106.438],
     std=[58.577, 57.310, 57.437],
     to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=224, backend='pillow'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(256, -1), backend='pillow'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    # 设置每个 GPU 上的 batch size 和 workers 数, 根据你的硬件来修改这些选项。
    samples_per_gpu=32,
    workers_per_gpu=2,
    # 指定训练集类型和路径
    train=dict(
        type=dataset_type,
        data_prefix='data/garbage_classification/train',
        classes='data/garbage_classification/classes.txt',
        pipeline=train_pipeline),
    # 指定验证集类型和路径
    val=dict(
        type=dataset_type,
        data_prefix='data/garbage_classification/val',
        ann_file='data/garbage_classification/val.txt',
        classes='data/garbage_classification/classes.txt',
        pipeline=test_pipeline),
    # 指定测试集类型和路径
    test=dict(
        type=dataset_type,
        data_prefix='data/garbage_classification/test',
        ann_file='data/garbage_classification/test.txt',
        classes='data/garbage_classification/classes.txt',
        pipeline=test_pipeline))

# 设置验证指标
evaluation = dict(metric='accuracy', metric_options={'topk': (1, )})

# ---- 优化器设置 ----
# 通常在微调任务中，我们需要一个较小的学习率，训练轮次可以较短。
# 设置学习率
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# 设置学习率调度器
lr_config = dict(policy='step', step=1, gamma=0.1)
runner = dict(type='EpochBasedRunner', max_epochs=2)

# ---- 运行设置 ----
# 每 10 个训练批次输出一次日志
log_config = dict(interval=10)
