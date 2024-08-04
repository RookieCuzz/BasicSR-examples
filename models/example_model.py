from collections import OrderedDict

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.models.sr_model import SRModel
from basicsr.utils import get_root_logger
from basicsr.utils.registry import MODEL_REGISTRY

@MODEL_REGISTRY.register()
class ExampleModel(SRModel):
    """Example model based on the SRModel class.

    In this example model, we want to implement a new model that trains with both L1 and L2 loss.

    New defined functions:
        init_training_settings(self)
        feed_data(self, data)
        optimize_parameters(self, current_iter)

    Inherited functions:
        __init__(self, opt)
        setup_optimizers(self)
        test(self)
        dist_validation(self, dataloader, current_iter, tb_logger, save_img)
        nondist_validation(self, dataloader, current_iter, tb_logger, save_img)
        _log_validation_metric_values(self, current_iter, dataset_name, tb_logger)
        get_current_visuals(self)
        save(self, epoch, current_iter)
    """

    def init_training_settings(self):
        """初始化训练设置，包括网络、损失函数、优化器和调度器。"""
        self.net_g.train()
        train_opt = self.opt['train']

        # 设置EMA（Exponential Moving Average）
        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # 定义带有EMA的网络net_g
            # net_g_ema只用于单GPU测试和保存，不需要用DistributedDataParallel包装
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # 加载预训练模型
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # 复制net_g权重
            self.net_g_ema.eval()

        # 定义损失函数
        self.l1_pix = build_loss(train_opt['l1_opt']).to(self.device)
        self.l2_pix = build_loss(train_opt['l2_opt']).to(self.device)

        # 设置优化器和调度器
        self.setup_optimizers()
        self.setup_schedulers()

    def feed_data(self, data):
        """将输入数据传递到模型中。

        Args:
            data (dict): 输入数据，包含低质量（LQ）和高质量（GT）图像。
        """
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        """优化参数，计算损失并更新模型权重。

        Args:
            current_iter (int): 当前迭代次数。
        """
        self.optimizer_g.zero_grad()  # 清除梯度
        self.output = self.net_g(self.lq)  # 前向传播

        l_total = 0
        loss_dict = OrderedDict()

        # 计算L1损失
        l_l1 = self.l1_pix(self.output, self.gt)
        l_total += l_l1
        loss_dict['l_l1'] = l_l1

        # 计算L2损失
        l_l2 = self.l2_pix(self.output, self.gt)
        l_total += l_l2
        loss_dict['l_l2'] = l_l2
        l_total.backward()  # 反向传播
        self.optimizer_g.step()  # 更新权重

        self.log_dict = self.reduce_loss_dict(loss_dict)  # 记录损失

        # 更新EMA模型
        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
