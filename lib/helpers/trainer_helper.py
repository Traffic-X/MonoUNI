import os
import tqdm

import torch
import torch.nn as nn
import numpy as np
import pdb
from lib.helpers.save_helper import get_checkpoint_state
from lib.helpers.save_helper import save_checkpoint
from lib.helpers.save_helper import load_checkpoint
from lib.losses.loss_function import GupnetLoss,Hierarchical_Task_Learning
from lib.helpers.decode_helper import extract_dets_from_outputs
from lib.helpers.decode_helper import decode_detections
from lib.eval_tools import eval

class Trainer(object):
    def __init__(self,
                 cfg,
                 model,
                 optimizer,
                 train_loader,
                 test_loader,
                 lr_scheduler,
                 warmup_lr_scheduler,
                 logger,
                 train_sampler,
                 local_rank,
                 args):
        self.cfg_train = cfg['trainer']
        self.cfg_test = cfg['tester']
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.lr_scheduler = lr_scheduler
        self.warmup_lr_scheduler = warmup_lr_scheduler
        self.logger = logger
        self.epoch = 0
        
        # for eval
        self.eval_cls = cfg['dataset']['eval_cls']
        self.root_dir = cfg['dataset']['root_dir']
        self.label_dir = os.path.join(self.root_dir, 'label_2_4cls_filter_with_roi_for_eval')
        self.calib_dir = os.path.join(self.root_dir, 'calib')
        self.de_norm_dir = os.path.join(self.root_dir, 'denorm')
        
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.class_name = test_loader.dataset.class_name
        self.train_sampler = train_sampler
        # self.val_sampler = val_sampler
        self.local_rank = local_rank
        self.args = args
        # print(local_rank)
        
        
        if self.cfg_train.get('resume_model', None):
            assert os.path.exists(self.cfg_train['resume_model'])
            self.epoch = load_checkpoint(self.model, self.optimizer, self.cfg_train['resume_model'], self.logger, map_location="cpu")
            self.lr_scheduler.last_epoch = self.epoch - 1

        # self.model = torch.nn.DataParallel(model).to(self.device)


    def train(self):
        start_epoch = self.epoch
        # self.train_sampler.set_epoch(0)
        # self.val_sampler.set_epoch(0)
        ei_loss = self.compute_e0_loss()
        loss_weightor = Hierarchical_Task_Learning(ei_loss)
        for epoch in range(start_epoch, self.cfg_train['max_epoch']):
            # train one epoch

            self.train_sampler.set_epoch(epoch)
            # self.val_sampler.set_epoch(epoch)
            self.logger.info('------ TRAIN EPOCH %03d ------' %(epoch + 1))
            if self.warmup_lr_scheduler is not None and epoch < 5:
                self.logger.info('Learning Rate: %f' % self.warmup_lr_scheduler.get_lr()[0])
            else:
                self.logger.info('Learning Rate: %f' % self.lr_scheduler.get_lr()[0])

            # reset numpy seed.
            # ref: https://github.com/pytorch/pytorch/issues/5059
            np.random.seed(np.random.get_state()[1][0] + epoch)
            loss_weights = loss_weightor.compute_weight(ei_loss,self.epoch)
            log_str = 'Weights: '
            for key in sorted(loss_weights.keys()):
                log_str += ' %s:%.4f,' %(key[:-4], loss_weights[key])   
            self.logger.info(log_str)                     
            ei_loss = self.train_one_epoch(loss_weights)
            self.epoch += 1
            
            # update learning rate
            if self.warmup_lr_scheduler is not None and epoch < 5:
                self.warmup_lr_scheduler.step()
            else:
                self.lr_scheduler.step()
                
            # save trained model
            if (self.epoch % self.cfg_train['save_frequency']) == 0 and self.local_rank==0:
                os.makedirs(self.cfg_train['log_dir']+'/checkpoints', exist_ok=True)
                ckpt_name = os.path.join(self.cfg_train['log_dir']+'/checkpoints', 'checkpoint_epoch_%d' % self.epoch)
                save_checkpoint(get_checkpoint_state(self.model.module, self.optimizer, self.epoch), ckpt_name, self.logger)
            
            if (self.epoch % self.cfg_train['eval_frequency']) == 0 and self.local_rank==0 :
                self.logger.info('------ EVAL EPOCH %03d ------' % (self.epoch))
                self.eval_one_epoch()

            


        return None
    
    def compute_e0_loss(self):
        self.model.train()
        disp_dict = {}
        progress_bar = tqdm.tqdm(total=len(self.train_loader), leave=True, desc='pre-training loss stat')
        with torch.no_grad():        
            for batch_idx, (inputs,calibs,coord_ranges, targets, info, calib_pitch_cos, calib_pitch_sin) in enumerate(self.train_loader):
                inputs = inputs.cuda(self.local_rank, non_blocking=True)
                calibs = calibs.cuda(self.local_rank, non_blocking=True)
                for key in targets:
                    targets[key] = targets[key].cuda(self.local_rank, non_blocking=True)
                # targets = targets.cuda(self.local_rank, non_blocking=True)
                calib_pitch_cos = calib_pitch_cos.cuda(self.local_rank, non_blocking=True)
                calib_pitch_sin = calib_pitch_sin.cuda(self.local_rank, non_blocking=True)
                coord_ranges = coord_ranges.cuda(self.local_rank, non_blocking=True)
                # print(f"data: {images.device}, model: {next(self.model.parameters()).device}")
         
                # train one batch
                criterion = GupnetLoss(self.epoch)
                # print(f"inputs: {inputs.device}, model: {next(self.model.parameters()).device}")
                outputs = self.model(inputs,coord_ranges,calibs,targets, calib_pitch_sin=calib_pitch_sin, calib_pitch_cos=calib_pitch_cos)
                _, loss_terms = criterion(outputs, targets)
                
                trained_batch = batch_idx + 1
                # accumulate statistics
                for key in loss_terms.keys():
                    if key not in disp_dict.keys():
                        disp_dict[key] = 0
                    disp_dict[key] += loss_terms[key]      
                progress_bar.update()
            progress_bar.close()
            for key in disp_dict.keys():
                disp_dict[key] /= trained_batch             
        return disp_dict
         
    def train_one_epoch(self,loss_weights=None):
        self.model.train()
        disp_dict = {}
        stat_dict = {}
        for batch_idx, (inputs,calibs,coord_ranges, targets, info, calib_pitch_cos, calib_pitch_sin) in enumerate(self.train_loader):

            inputs = inputs.cuda(self.local_rank, non_blocking=True)
            calibs = calibs.cuda(self.local_rank, non_blocking=True)
            for key in targets:
                targets[key] = targets[key].cuda(self.local_rank, non_blocking=True)
            coord_ranges = coord_ranges.cuda(self.local_rank, non_blocking=True)
            calib_pitch_cos = calib_pitch_cos.cuda(self.local_rank, non_blocking=True)
            calib_pitch_sin = calib_pitch_sin.cuda(self.local_rank, non_blocking=True)
            # for key in targets.keys(): targets[key] = targets[key].to(self.device)
            # train one batch
            self.optimizer.zero_grad()
            criterion = GupnetLoss(self.epoch)
            
            outputs = self.model(inputs,coord_ranges,calibs,targets, calib_pitch_sin=calib_pitch_sin, calib_pitch_cos=calib_pitch_cos)
            total_loss, loss_terms = criterion(outputs, targets)
            
            if loss_weights is not None:
                total_loss = torch.zeros(1).cuda()
                for key in loss_weights.keys():
                    total_loss += loss_weights[key].detach()*loss_terms[key]
            total_loss.backward()
            self.optimizer.step()
            
            trained_batch = batch_idx + 1

            # accumulate statistics
            for key in loss_terms.keys():
                if key not in stat_dict.keys():
                    stat_dict[key] = 0
                stat_dict[key] += loss_terms[key] 
            for key in loss_terms.keys():
                if key not in disp_dict.keys():
                    disp_dict[key] = 0
                disp_dict[key] += loss_terms[key]   
            # display statistics in terminal
            if trained_batch % self.cfg_train['disp_frequency'] == 0:
                log_str = 'BATCH[%04d/%04d]' % (trained_batch, len(self.train_loader))
                for key in sorted(disp_dict.keys()):
                    disp_dict[key] = disp_dict[key] / self.cfg_train['disp_frequency']
                    log_str += ' %s:%.4f,' %(key, disp_dict[key])
                    disp_dict[key] = 0  # reset statistics
                self.logger.info(log_str)
                
        for key in stat_dict.keys():
            stat_dict[key] /= trained_batch
                            
        return stat_dict
    
    def eval_one_epoch(self):
        self.model.eval()
        results = {}
        disp_dict = {}
        progress_bar = tqdm.tqdm(total=len(self.test_loader), leave=True, desc='Evaluation Progress')
        with torch.no_grad():
            for batch_idx, (inputs, calibs, coord_ranges, _, info, calib_pitch_cos, calib_pitch_sin) in enumerate(self.test_loader):
                # load evaluation data and move data to current device.
                inputs = inputs.cuda(self.local_rank, non_blocking=True)
                calibs = calibs.cuda(self.local_rank, non_blocking=True)
                coord_ranges = coord_ranges.cuda(self.local_rank, non_blocking=True)
                calib_pitch_cos = calib_pitch_cos.cuda(self.local_rank, non_blocking=True)
                calib_pitch_sin = calib_pitch_sin.cuda(self.local_rank, non_blocking=True)
    
                # the outputs of centernet
                outputs = self.model(inputs,coord_ranges,calibs,K=50,mode='val', calib_pitch_sin=calib_pitch_sin, calib_pitch_cos=calib_pitch_cos)

                dets = extract_dets_from_outputs(outputs,calibs, K=50)
                dets = dets.detach().cpu().numpy()
                
                # get corresponding calibs & transform tensor to numpy
                calibs = [self.test_loader.dataset.get_calib(index)  for index in info['img_id']]
                denorms = [self.test_loader.dataset.get_denorm(index)  for index in info['img_id']]
                # _aaa = info.items()
                # info = {}
                info['img_id'] = info['img_id']
                info['img_size'] = info['img_size'].detach().cpu().numpy()
                info['bbox_downsample_ratio'] = info['bbox_downsample_ratio'].detach().cpu().numpy()
                
                # info = {key: val.detach().cpu().numpy() for key, val in info.items()}
                cls_mean_size = self.test_loader.dataset.cls_mean_size
                dets = decode_detections(dets = dets,
                                        info = info,
                                        calibs = calibs,
                                        denorms = denorms,
                                        cls_mean_size=cls_mean_size,
                                        threshold = self.cfg_test['threshold'])                 
                results.update(dets)
                progress_bar.update()
            progress_bar.close()
        out_dir = os.path.join(self.cfg_train['out_dir'], 'EPOCH_' + str(self.epoch))
        self.save_results(results,out_dir)
        # return 0
        Car_res = eval.do_repo3d_eval(
            self.logger,
            self.label_dir,
            os.path.join(out_dir, 'data'),
            self.calib_dir,
            self.de_norm_dir,
            self.eval_cls,
            ap_mode=40)
        return Car_res
           
                
    def save_results(self, results, output_dir='./outputs'):
        output_dir = os.path.join(output_dir, 'data')
        os.makedirs(output_dir, exist_ok=True)

        for img_id in results.keys():
            out_path = os.path.join(output_dir, img_id+'.txt')
            f = open(out_path, 'w')
            for i in range(len(results[img_id])):
                class_name = self.class_name[int(results[img_id][i][0])]
                f.write('{} 0.0 0'.format(class_name))
                for j in range(1, len(results[img_id][i])):
                    f.write(' {:.2f}'.format(results[img_id][i][j]))
                f.write('\n')
            f.close()        
        
      
