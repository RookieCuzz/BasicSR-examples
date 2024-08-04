import cv2
import os
import torch
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.degradations import add_jpg_compression
from basicsr.data.transforms import augment, mod_crop, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor, scandir
from basicsr.utils.registry import DATASET_REGISTRY

@DATASET_REGISTRY.register()  # 使用装饰器将数据集类注册到全局注册表中
class ExampleDataset(data.Dataset):
    """Example dataset.

    1. 读取GT（高质量）图像
    2. 使用cv2双三次下采样和JPEG压缩生成LQ（低质量）图像

    Args:
        opt (dict): 训练数据集的配置。包含以下键：
            dataroot_gt (str): GT数据根路径。
            io_backend (dict): IO后端类型及其他参数。
            gt_size (int): GT图像裁剪的尺寸。
            use_flip (bool): 是否使用水平翻转。
            use_rot (bool): 是否使用旋转（垂直翻转和转置h和w的实现）。
            scale (bool): 缩放因子，将自动添加。
            phase (str): 'train' 或 'val'。
    """

    def __init__(self, opt):
        super(ExampleDataset, self).__init__()
        self.opt = opt
        # 文件客户端（IO后端）
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder = opt['dataroot_gt']
        # 目前仅支持文件夹模式，其他模式请参见：
        # https://github.com/xinntao/BasicSR/blob/master/basicsr/data/
        self.paths = [os.path.join(self.gt_folder, v) for v in scandir(self.gt_folder)]

    def __getitem__(self, index):
        # 延迟初始化文件客户端
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # 加载GT图像。维度顺序：HWC；通道顺序：BGR；图像范围：[0, 1]，float32。
        gt_path = self.paths[index]
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)
        img_gt = mod_crop(img_gt, scale)

        # 生成LQ图像
        # 下采样
        h, w = img_gt.shape[0:2]
        img_lq = cv2.resize(img_gt, (w // scale, h // scale), interpolation=cv2.INTER_CUBIC)
        # 添加JPEG压缩
        img_lq = add_jpg_compression(img_lq, quality=70)

        # 训练阶段的数据增强
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # 随机裁剪
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            # 翻转和旋转
            img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_flip'], self.opt['use_rot'])

        # BGR转换为RGB，HWC转换为CHW，numpy转换为tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        img_lq = torch.clamp((img_lq * 255.0).round(), 0, 255) / 255.

        # 归一化
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {'lq': img_lq, 'gt': img_gt, 'lq_path': gt_path, 'gt_path': gt_path}

    def __len__(self):
        # 返回数据集的大小
        return len(self.paths)
