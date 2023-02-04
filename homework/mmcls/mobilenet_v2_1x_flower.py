## 继承原有的 config 配置文件
_base_ = [
    '../_base_/models/mobilenet_v2_1x.py',               # 模型结构
    '../_base_/datasets/imagenet_bs32_pil_resize.py',    # 数据预处理、数据扩增
    '../_base_/schedules/imagenet_bs256_epochstep.py',   # 学习率、训练策略
    '../_base_/default_runtime.py'                       # 日志输出
]

## 以下仅写需要修改的部分即可

## 模型结构
model = dict(
    head=dict(
        num_classes=5,     # 分类个数
        topk=(1, 3),        # top-k
    ))

## 数据集
data = dict(
    samples_per_gpu = 32, # 单卡 batchsize
    workers_per_gpu=2,
    # 训练集
    train = dict(
        data_prefix = 'data/flower/train', # 训练集图像路径
        classes = 'data/flower/classes.txt'                    # 类别txt文件
    ),
    # 验证集
    val = dict(
        data_prefix = 'data/flower/val',           # 验证集图像路径
        ann_file = 'data/flower/val.txt',                      # 验证集标签txt文件
        classes = 'data/flower/classes.txt'                    # 类别txt文件
    ),
    # 测试集
    test = dict(
        data_prefix = 'data/flower/test',         # 测试集图像路径
        ann_file = 'data/flower/test.txt',                     # 测试集标签txt文件
        classes = 'data/flower/classes.txt'                    # 类别txt文件
    )
)
## 评估指标
evaluation = dict(interval=1, metric=['accuracy','precision','f1_score'], metric_options=dict(topk=(1,)))

## 学习率与优化器
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

# 学习率策略
lr_config = dict(
    policy='step',
    step=[1])
runner = dict(type='EpochBasedRunner', max_epochs=2)

## 预训练模型
# load_from = None # 随机初始化
load_from = 'https://download.openmmlab.com/mmclassification/v0/mobilenet_v2/mobilenet_v2_batch256_imagenet_20200708-3b2dc3af.pth'



