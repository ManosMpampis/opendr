from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.util.yacs import CfgNode

cfg = CfgNode(new_allowed=True)
cfg.save_dir = "./"

# params for Quantization Aware Training
cfg.qat = CfgNode(new_allowed=True)
cfg.qat.enable = False
cfg.qat.qconfig = "qnnpack"
cfg.qat.freeze_bn = -1
cfg.qat.freeze_quantizer_parameters = -1

# common params for NETWORK
cfg.model = CfgNode(new_allowed=True)
cfg.model.arch = CfgNode(new_allowed=True)
cfg.model.arch.fuse = False
cfg.model.arch.ch_l = False
cfg.model.arch.backbone = CfgNode(new_allowed=True)
cfg.model.arch.fpn = CfgNode(new_allowed=True)
cfg.model.arch.head = CfgNode(new_allowed=True)

# DATASET related params
cfg.data = CfgNode(new_allowed=True)
cfg.data.train = CfgNode(new_allowed=True)
cfg.data.train.cache_images = ""
cfg.data.val = CfgNode(new_allowed=True)
cfg.data.val.cache_images = ""
cfg.data.bench_test = CfgNode(new_allowed=True)
cfg.device = CfgNode(new_allowed=True)
cfg.device.precision = 32
# train
cfg.schedule = CfgNode(new_allowed=True)
cfg.schedule.effective_batchsize = 1

# logger
cfg.log = CfgNode()
cfg.log.interval = 50

# testing
cfg.test = CfgNode()
# size of images for each device


def load_config(cfg, args_cfg):
    cfg.defrost()
    cfg.merge_from_file(args_cfg)
    cfg.freeze()


if __name__ == "__main__":
    import sys

    with open(sys.argv[1], "w") as f:
        print(cfg, file=f)
