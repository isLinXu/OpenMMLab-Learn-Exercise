
import os
filePrefix='Test'
fileSuffix='.py'
#with open(filename,'w') as f:
fo = open("mobilenet.py", "a")
def config_path():
    # 载入已经存在的配置文件
    from mmcv import Config
    from mmcls.utils import auto_select_device
    checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/mobilenet_v2/mobilenet_v2_batch256_imagenet_20200708-3b2dc3af.pth'
    cfg = Config.fromfile('configs/mobilenet_v2/mobilenet-v2_8xb32_in1k.py')
    cfg.device = auto_select_device()

    # 修改模型分类头中的类别数目
    cfg.model.head.num_classes = 2
    cfg.model.head.topk = (1, )

    # 加载预训练权重
    cfg.model.backbone.init_cfg = dict(type='Pretrained', checkpoint=checkpoint_file, prefix='backbone')

    # 根据你的电脑情况设置 sample size 和 workers 
    cfg.data.samples_per_gpu = 32
    cfg.data.workers_per_gpu = 2

    # 指定训练集路径
    cfg.data.train.data_prefix = 'data/cats_dogs_dataset/training_set/training_set'
    cfg.data.train.classes = 'data/cats_dogs_dataset/classes.txt'

    # 指定验证集路径
    cfg.data.val.data_prefix = 'data/cats_dogs_dataset/val_set/val_set'
    cfg.data.val.ann_file = 'data/cats_dogs_dataset/val.txt'
    cfg.data.val.classes = 'data/cats_dogs_dataset/classes.txt'

    # 指定测试集路径
    cfg.data.test.data_prefix = 'data/cats_dogs_dataset/test_set/test_set'
    cfg.data.test.ann_file = 'data/cats_dogs_dataset/test.txt'
    cfg.data.test.classes = 'data/cats_dogs_dataset/classes.txt'

    # 设定数据集归一化参数
    normalize_cfg = dict(type='Normalize', mean=[124.508, 116.050, 106.438], std=[58.577, 57.310, 57.437], to_rgb=True)
    cfg.data.train.pipeline[3] = normalize_cfg
    cfg.data.val.pipeline[3] = normalize_cfg
    cfg.data.test.pipeline[3] = normalize_cfg

    # 修改评价指标选项
    cfg.evaluation['metric_options']={'topk': (1, )}

    # 设置优化器
    cfg.optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
    cfg.optimizer_config = dict(grad_clip=None)

    # 设置学习率策略
    cfg.lr_config = dict(policy='step', step=1, gamma=0.1)
    cfg.runner = dict(type='EpochBasedRunner', max_epochs=2)

    # 设置工作目录以保存模型和日志
    cfg.work_dir = './work_dirs/cats_dogs_dataset'

    # 设置每 10 个训练批次输出一次日志
    cfg.log_config.interval = 10

    # 设置随机种子，并启用 cudnn 确定性选项以保证结果的可重复性
    from mmcls.apis import set_random_seed
    cfg.seed = 0
    set_random_seed(0, deterministic=True)

    cfg.gpu_ids = range(1)
    fo.write(str(cfg.dump()))
config_path()

# f.write('')

