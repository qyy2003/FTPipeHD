from utils.yacs import CfgNode

cfg = CfgNode(new_allowed=True)
cfg.save_dir = './save'

# DATASET related params
cfg.data = CfgNode(new_allowed=True)
cfg.data.train = CfgNode(new_allowed=True)
cfg.data.val = CfgNode(new_allowed=True)
# train
cfg.schedule = CfgNode(new_allowed=True)

# logger
cfg.log = CfgNode()
cfg.log.interval = 50

def load_config(cfg, args_cfg):
    cfg.defrost()
    cfg.merge_from_file(args_cfg)
    cfg.freeze()


if __name__ == '__main__':
    import sys

    with open(sys.argv[1], 'w') as f:
        print(cfg, file=f)
