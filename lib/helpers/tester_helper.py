import os
import tqdm

import torch
import numpy as np

from lib.helpers.save_helper import load_checkpoint
from lib.helpers.decode_helper import extract_dets_from_outputs
from lib.helpers.decode_helper import decode_detections
import lib.eval_tools.eval as eval

class Tester(object):
    def __init__(self, cfg, model, data_loader, logger):
        self.cfg = cfg['tester']
        # for eval
        self.eval_cls = cfg['dataset']['eval_cls']
        self.root_dir = cfg['dataset']['root_dir']
        self.label_dir = os.path.join(self.root_dir, 'label_2_4cls_filter_with_roi_for_eval')
        self.calib_dir = os.path.join(self.root_dir, 'calib')
        self.de_norm_dir = os.path.join(self.root_dir, 'denorm')
        self.model = model
        self.data_loader = data_loader
        self.logger = logger
        self.class_name = data_loader.dataset.class_name
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if self.cfg.get('resume_model', None):
            load_checkpoint(model = self.model,
                        optimizer = None,
                        filename = self.cfg['resume_model'],
                        logger = self.logger,
                        map_location=self.device)

        self.model.to(self.device)


    def test(self):
        torch.set_grad_enabled(False)
        self.model.eval()

        results = {}
        progress_bar = tqdm.tqdm(total=len(self.data_loader), leave=True, desc='Evaluation Progress')
        for batch_idx, (inputs, calibs, coord_ranges, _, info, calib_pitch_cos, calib_pitch_sin) in enumerate(self.data_loader):
            # load evaluation data and move data to current device.
            inputs = inputs.to(self.device)
            calibs = calibs.to(self.device)
            coord_ranges = coord_ranges.to(self.device)
            calib_pitch_cos = calib_pitch_cos.to(self.device)
            calib_pitch_sin = calib_pitch_sin.to(self.device)

            # the outputs of centernet
            outputs = self.model(inputs,coord_ranges,calibs,K=50,mode='val', calib_pitch_sin=calib_pitch_sin, calib_pitch_cos=calib_pitch_cos)

            dets = extract_dets_from_outputs(outputs,calibs, K=50)
            dets = dets.detach().cpu().numpy()
            
            # get corresponding calibs & transform tensor to numpy
            calibs = [self.data_loader.dataset.get_calib(index)  for index in info['img_id']]
            denorms = [self.data_loader.dataset.get_denorm(index)  for index in info['img_id']]
            info['img_id'] = info['img_id']
            info['img_size'] = info['img_size'].detach().cpu().numpy()
            info['bbox_downsample_ratio'] = info['bbox_downsample_ratio'].detach().cpu().numpy()
            cls_mean_size = self.data_loader.dataset.cls_mean_size
            dets = decode_detections(dets = dets,
                                    info = info,
                                    calibs = calibs,
                                    denorms = denorms,
                                    cls_mean_size=cls_mean_size,
                                    threshold = self.cfg['threshold'])                 
            results.update(dets)
            progress_bar.update()

        out_dir = os.path.join(self.cfg['out_dir'])
        self.save_results(results,out_dir)
        progress_bar.close()
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
        
      







