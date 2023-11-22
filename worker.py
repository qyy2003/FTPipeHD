import torch
import numpy as np
from global_variables.common import set_program_args,set_device

from utils.init import arg_parse
from flask_api.worker.index import app
from global_variables.config import cfg, load_config
from utils.init import init_logger


if __name__ == "__main__":
    args = arg_parse()

    # Pre-load the model
    # torch.autograd.set_detect_anomaly(True)

    # Seed Initialization
    init_seed = 1
    torch.manual_seed(init_seed)
    np.random.seed(init_seed) # 用于numpy的随机数

    # cuda related
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    device="cpu"
    set_device(device)
    load_config(cfg, args.config)
    init_logger(args.local_rank, cfg.save_dir + "_worker_" + str(args.port))
    set_program_args(args)
    
    app.run(host='0.0.0.0', port=args.port, threaded=True)
