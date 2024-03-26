import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import yaml
import logging
import argparse

from lib.helpers.dataloader_helper import build_dataloader
from lib.helpers.model_helper import build_model
from lib.helpers.optimizer_helper import build_optimizer
from lib.helpers.scheduler_helper import build_lr_scheduler
from lib.helpers.trainer_helper import Trainer
from lib.helpers.tester_helper import Tester
from datetime import datetime
import numpy as np
from torch.utils.data import DataLoader
from lib.datasets.rope3d import Rope3D
import torch
import torch.multiprocessing as mp
import random
import torch.backends.cudnn as cudnn
import torch.distributed as dist

def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def create_logger(log_file):
    log_format = '%(asctime)s  %(levelname)5s  %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format, filename=log_file)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(log_format))
    logging.getLogger(__name__).addHandler(console)
    return logging.getLogger(__name__)


def main_worker(local_rank, nprocs, args):  
    # load cfg
    args.local_rank = local_rank
    assert (os.path.exists(args.config))
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    
    import shutil
    if not args.evaluate and local_rank==0:
        if os.path.exists(os.path.join(cfg['trainer']['log_dir'], 'lib/')):
            shutil.rmtree(os.path.join(cfg['trainer']['log_dir'], 'lib/'))
        
        shutil.copytree('./lib', os.path.join(cfg['trainer']['log_dir'], 'lib/'))

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    # ip = random.randint(1000,10000)
    dist.init_process_group(backend='nccl',
                        init_method='tcp://127.0.0.1:'+str(args.ip),
                        world_size=args.nprocs,
                        rank=local_rank)

    os.makedirs(cfg['trainer']['log_dir'],exist_ok=True)
    logger = create_logger(os.path.join(cfg['trainer']['log_dir'],'train.log'))    
    

    train_set = Rope3D(root_dir=cfg['dataset']['root_dir'], split='train', cfg=cfg['dataset'])
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    train_loader = DataLoader(dataset=train_set,
                                batch_size= int(cfg['dataset']['batch_size'] * 4 / args.nprocs),
                                num_workers=2,
                                shuffle=False,
                                pin_memory=True,
                                drop_last=False,
                                sampler=train_sampler)

    val_set = Rope3D(root_dir=cfg['dataset']['root_dir'], split='val', cfg=cfg['dataset'])
    val_loader = DataLoader(dataset=val_set,
                                batch_size=cfg['dataset']['batch_size']*4,
                                num_workers=2,
                                shuffle=False,
                                pin_memory=True,
                                drop_last=False)
    # build model
    model = build_model(cfg['model'],train_loader.dataset.cls_mean_size)
    if args.evaluate:
        tester = Tester(cfg, model, val_loader, logger)
        tester.test()
        return                                                                   


    # print(local_rank)
    torch.cuda.set_device(local_rank)
    model.cuda(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[local_rank],find_unused_parameters=True)

    # print(f"model: {next(model.parameters()).device}")

    #  build optimizer
    optimizer = build_optimizer(cfg['optimizer'], model)

    # build lr & bnm scheduler
    lr_scheduler, warmup_lr_scheduler = build_lr_scheduler(cfg['lr_scheduler'], optimizer, last_epoch=-1)

    trainer = Trainer(cfg=cfg,
                      model=model,
                      optimizer=optimizer,
                      train_loader=train_loader,
                      test_loader=val_loader,
                      lr_scheduler=lr_scheduler,
                      warmup_lr_scheduler=warmup_lr_scheduler,
                      logger=logger,
                      train_sampler=train_sampler,
                      local_rank=local_rank,
                      args=args)
    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='implementation of GUPNet')
    parser.add_argument('-e', '--evaluate', dest='evaluate',action='store_true',help='evaluate model on validation set')
    parser.add_argument('--config', type=str, default = 'lib/config.yaml')
    parser.add_argument('--seed',default=None,type=int, help='seed for initializing training. ')
    parser.add_argument('--local_rank',default=0,type=int,help='node rank for distributed training')
    parser.add_argument('--ip',default=1222,type=int,help='node rank for distributed training')
    

    args = parser.parse_args()
    args.nprocs = torch.cuda.device_count()
    main_worker(args.local_rank,args.nprocs, args)
    # mp.spawn(main_worker, nprocs=args.nprocs, args=(args.nprocs, args))
