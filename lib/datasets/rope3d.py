import os
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import random
from tqdm import tqdm
import copy

from lib.datasets.utils import angle2class
from lib.datasets.utils import gaussian_radius
from lib.datasets.utils import draw_umich_gaussian
from lib.datasets.utils import get_angle_from_box3d,check_range
from lib.datasets.rope3d_utils import get_objects_from_label
from lib.datasets.rope3d_utils import Calibration,Denorm
from lib.datasets.rope3d_utils import get_affine_transform
from lib.datasets.rope3d_utils import affine_transform
from lib.datasets.rope3d_utils import compute_box_3d
import pdb

class Rope3D(data.Dataset):
    def __init__(self, root_dir, split, cfg):
        # basic configuration
        self.num_classes = 4
        self.max_objs = 50
        self.class_name =  ['car','big_vehicle','pedestrian','cyclist']
        self.cls2id = {'car': 0,'big_vehicle': 1,'pedestrian': 2,'cyclist': 3}
        self.resolution = np.array([960, 512])  # W * H
        self.use_3d_center = cfg['use_3d_center']
        self.writelist = cfg['writelist']
        if cfg['class_merging']:
            self.writelist.extend(['Van', 'Truck'])
        if cfg['use_dontcare']:
            self.writelist.extend(['DontCare'])
        if cfg['load_data_once']:
            self.load_data_once = True
        else:
            self.load_data_once = False
        '''    
        ['Car': np.array([3.88311640418,1.62856739989,1.52563191462]),
         'Pedestrian': np.array([0.84422524,0.66068622,1.76255119]),
         'Cyclist': np.array([1.76282397,0.59706367,1.73698127])] 
        ''' 
        ##l,w,h
        self.cls_mean_size = np.array([[1.288762253204939, 1.6939648801353426, 4.25589251897889],
                                       [1.7199308570318539, 1.7356837654961508, 4.641152817981265],
                                       [2.682263889273618, 2.3482764551684268, 6.940250839428722],
                                       [2.9588510594399073, 2.5199248789610693, 10.542197736838778]])
        # data split loading
        # print(split)
        assert split in ['train', 'val']
        self.split = split
        split_dir = os.path.join(root_dir, 'ImageSets', split + '.txt')
        self.idx_list = [x.strip() for x in open(split_dir).readlines()]

        # path configuration
        self.data_dir = root_dir
        self.image_dir = os.path.join(self.data_dir, 'image_2')
        self.label_dir = os.path.join(self.data_dir, 'label_2_4cls_for_train')
        self.calib_dir = os.path.join(self.data_dir, 'calib')
        self.denorm_dir = os.path.join(self.data_dir, 'denorm')
        self.box3d_dense_depth_dir = os.path.join(self.data_dir, 'box3d_depth_dense')

        self.interval_max = cfg['interval_max']
        self.interval_min = cfg['interval_min']

        # data augmentation configuration
        self.data_augmentation = True if split == 'train' else False
        self.random_flip = cfg['random_flip']
        self.random_crop = cfg['random_crop']
        self.scale = cfg['scale']
        self.shift = cfg['shift']
        self.crop_with_optical_center = cfg['crop_with_optical_center']
        self.crop_with_optical_center_with_fx_limit = True
        self.scale_expand = cfg['scale_expand']
        

        self.labels = []
        # print('load_all_labels')
        for i in tqdm(range(len(self.idx_list)),desc='load_all_labels Progress'):
            idx = self.idx_list[i]
            label_file = os.path.join(self.label_dir,  idx+'.txt')
            assert os.path.exists(label_file)
            self.labels.append(get_objects_from_label(label_file))

        # print('load_all_calib')
        self.calib = []
        for i in tqdm(range(len(self.idx_list)),desc='load_all_calib Progress'):
            idx = self.idx_list[i]
            calib_file = os.path.join(self.calib_dir,  idx+'.txt')
            if os.path.exists(calib_file) != True:
                print("calib_file : ", calib_file)
            assert os.path.exists(calib_file)
            self.calib.append(Calibration(calib_file))

        if self.load_data_once:
            self.img_list = []
            for i in tqdm(range(len(self.idx_list)),desc='load_img Progress'):
                idx = self.idx_list[i]
                img_file = os.path.join(self.image_dir, idx+'.jpg')
                assert os.path.exists(img_file)
                self.img_list.append(cv2.resize(cv2.imread(img_file),(960,540)))
        self.denorms = []
        for i in tqdm(range(len(self.idx_list)),desc='load_all_denorms Progress'):
            idx = self.idx_list[i]
            denorm_file = os.path.join(self.denorm_dir,  idx+'.txt')
            assert os.path.exists(denorm_file)
            self.denorms.append(Denorm(denorm_file))



        # statistics
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # others
        self.downsample = 4
        
    def get_image(self, idx):
        if self.load_data_once:
            return copy.deepcopy(self.img_list[self.idx_list.index(idx)])

        img_file = os.path.join(self.image_dir, idx+'.jpg')
        # print(img_file)
        assert os.path.exists(img_file)
        return cv2.imread(img_file)  # (H, W, 3) RGB mode


    def get_label(self, idx):
        return self.labels[self.idx_list.index(idx)]
        # label_file = os.path.join(self.label_dir,  idx+'.txt')
        # assert os.path.exists(label_file)
        # return get_objects_from_label(label_file)

    def get_calib(self, idx):
        return self.calib[self.idx_list.index(idx)]
        # calib_file = os.path.join(self.calib_dir,  idx+'.txt')
        # assert os.path.exists(calib_file)
        # return Calibration(calib_file)

    def get_denorm(self,idx):
        # denorm_file = os.path.join(self.denorm_dir, '%s.txt' % idx)
        # assert os.path.exists(denorm_file)
        # return Denorm(denorm_file)
        return copy.deepcopy(self.denorms[self.idx_list.index(idx)])


    def __len__(self):
        return self.idx_list.__len__()

    def Flip_with_optical_center(self, img, calib,mean):
        cx = calib.P2[0, 2]
        cy = calib.P2[1, 2]
        h,w,_ = img.shape
        if cx < w/2:
            x_min = 0
            x_max = int(cx*2)+1
        else:
            x_max = int(w)
            x_min = int(x_max- (x_max - cx-1)*2)-1
        crop_box = [x_min, 0, x_max, int(h)]
        crop_img = img[0:int(h),x_min:x_max, :]
        flip_img = crop_img[:, ::-1, :]
        res_img = np.ones((h, w, 3), dtype=img.dtype)
        res_img *= np.array(mean, dtype=img.dtype)
        res_img[0:int(h),x_min:x_max,:] = flip_img
        return res_img,crop_box

        
    def __getitem__(self, item):
        
        inputs, calib_, coord_range, targets, info, pitch_cos, pitch_sin = self.get_data(item)
        if self.split=='val':
            return inputs, calib_, coord_range, targets, info, pitch_cos, pitch_sin 
        while np.sum(targets['mask_2d'])==0:
            if random.uniform(0,1) < 0.3:
                return  inputs, calib_, coord_range, targets, info, pitch_cos, pitch_sin
            id = np.random.randint(0,len(self.idx_list))
            inputs, calib_, coord_range, targets, info, pitch_cos, pitch_sin = self.get_data(id)
        return inputs, calib_, coord_range, targets, info, pitch_cos, pitch_sin





    def get_data(self, item):
        #  ============================   get inputs   ===========================
        index = self.idx_list[item]  # index mapping, get real data id
        # image loading
        # print(index)
        img = self.get_image(index)
        calib = copy.deepcopy(self.get_calib(index))
        img_size = np.array([img.shape[1],img.shape[0]])
        random_crop_flag, random_flip_flag, random_expand_flag = False, False, False
        expand_scale = 1
        crop_scale = 1
        center = np.array(img_size) /2.
        Denorm_ = self.get_denorm(index)
        if self.split == 'train':
            vis_depth_generate_new = cv2.imread('{}/{}.png'.format(self.box3d_dense_depth_dir, index), -1) / 256.
            vis_depth_generate_new =vis_depth_generate_new[:,:,np.newaxis]
            vis_depth_generate_new = np.concatenate((vis_depth_generate_new, vis_depth_generate_new,vis_depth_generate_new),-1)
        if self.data_augmentation:

        # if False:
            if np.random.random() < 0.5:
                random_flip_flag = True
                img,flip_box = self.Flip_with_optical_center(img, calib,mean=self.mean)
                vis_depth_generate_new,_ = self.Flip_with_optical_center(vis_depth_generate_new, calib,mean=[0,0,0])
            if np.random.random() < 0.5:
                random_crop_flag = True
                crop_scale = np.random.uniform(self.scale,1) 
                # if self.crop_with_optical_center_with_fx_limit:
                #     cx = calib.P2[0, 2]
                #     cy = calib.P2[1, 2]
                #     fx = calib.P2[0, 0]
                #     fy = calib.P2[1, 1]
                #     new_cx = cx * crop_scale
                #     new_cy = cy * crop_scale
                #     crop_h = int(img.shape[0] * crop_scale)
                #     crop_w = int(img.shape[1] * crop_scale)
                #     crop_y = int(cy-new_cy)
                #     crop_x = int(cx-new_cx)
                #     crop_box = [crop_x, crop_y, crop_x + crop_w, crop_y + crop_h]
                #     img = img[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w, :]
                # print(crop_scale)
                if self.crop_with_optical_center:
                    cx = calib.P2[0, 2]
                    cy = calib.P2[1, 2]
                    new_cx = cx * crop_scale
                    new_cy = cy * crop_scale
                    crop_h = int(img.shape[0] * crop_scale)
                    crop_w = int(img.shape[1] * crop_scale)
                    crop_y = int(cy-new_cy)
                    crop_x = int(cx-new_cx)
                    crop_box = [crop_x, crop_y, crop_x + crop_w, crop_y + crop_h]
                    img = img[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w, :]
                    vis_depth_generate_new = vis_depth_generate_new[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w, :]
                    
                else:
                    crop_h = int(img.shape[0] * crop_scale)
                    crop_w = int(img.shape[1] * crop_scale)
                    crop_y = random.randint(0,img_size[0]-crop_h)
                    crop_x = random.randint(0,img_size[1]-crop_w)
                    crop_box = [crop_x, crop_y, crop_x + crop_w, crop_y + crop_h]
                    img = img[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w, :]
            if np.random.random() < 0.5:
                random_expand_flag = True
                expand_scale = np.random.uniform(1,self.scale_expand) 
                cx = calib.P2[0, 2] * crop_scale
                cy = calib.P2[1, 2] * crop_scale
                new_cx = cx * expand_scale
                new_cy = cy * expand_scale
                new_h = int(img.shape[0] * expand_scale)
                new_w = int(img.shape[1] * expand_scale)
                expand_y = int(new_cy-cy)
                expand_x = int(new_cx-cx)
                canvas = np.ones((new_h, new_w, 3), dtype=img.dtype)
                canvas *= np.array(self.mean, dtype=img.dtype)
                canvas[expand_y:expand_y + img.shape[0], expand_x:expand_x + img.shape[1], :] = img
                img = canvas
                
                canvas_vis_depth_generate_new = np.zeros((new_h, new_w, 3), dtype=vis_depth_generate_new.dtype)
                canvas_vis_depth_generate_new[expand_y:expand_y + vis_depth_generate_new.shape[0], expand_x:expand_x + vis_depth_generate_new.shape[1], :] = vis_depth_generate_new
                vis_depth_generate_new = canvas_vis_depth_generate_new


        # vis_depth_generate_new_ = cv2.resize(vis_depth_generate_new,(self.resolution[0],self.resolution[1]))
        if self.split == 'train':
            vis_depth_generate_new = vis_depth_generate_new[:,:,0]

        

        new_img_size = [img.shape[1],img.shape[0]]

        img = cv2.resize(img,(self.resolution[0],self.resolution[1]))
        # img_cp = copy.deepcopy(img)
        img = img.astype(np.float32)  / 255.0
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)  
                
        features_size = self.resolution // self.downsample# W * H

        down_w = new_img_size[0] / self.resolution[0]
        down_h = new_img_size[1] / self.resolution[1]

        if random_crop_flag:
            center = [crop_x+int(crop_w/2),crop_y+int(crop_h/2)]
        if random_expand_flag:
            if not random_crop_flag:
                center = [int(new_w/2)-expand_x,int(new_h/2)-expand_y]
            else:
                center = [int(new_w/2)-expand_x+crop_x,int(new_h/2)-expand_y+crop_y]
        # center = np.array(new_img_size) /2.
        crop_size =  np.array(new_img_size)
        coord_range = np.array([center-crop_size/2,center+crop_size/2]).astype(np.float32) 


        #  ============================   get labels   ==============================
        if self.split == 'train':
            objects = copy.deepcopy(self.get_label(index))

            # labels encoding
            heatmap = np.zeros((self.num_classes, features_size[1], features_size[0]), dtype=np.float32) # C * H * W
            size_2d = np.zeros((self.max_objs, 2), dtype=np.float32)
            offset_2d = np.zeros((self.max_objs, 2), dtype=np.float32)
            depth_offset = np.zeros((self.max_objs, 5), dtype=np.float32)
            depth_mask = np.zeros((self.max_objs, 5), dtype=np.int64)
            depth_bin = np.zeros((self.max_objs, 1), dtype=np.int64)
            heading_bin = np.zeros((self.max_objs, 1), dtype=np.int64)
            heading_res = np.zeros((self.max_objs, 1), dtype=np.float32)
            depth = np.zeros((self.max_objs, 1), dtype=np.float32)
            src_size_3d = np.zeros((self.max_objs, 3), dtype=np.float32)
            size_3d = np.zeros((self.max_objs, 3), dtype=np.float32)
            offset_3d = np.zeros((self.max_objs, 2), dtype=np.float32)
            cls_ids = np.zeros((self.max_objs), dtype=np.int64)
            indices = np.zeros((self.max_objs), dtype=np.int64)
            mask_2d = np.zeros((self.max_objs), dtype=np.bool)
            depth_bin_ind = np.zeros((self.max_objs, 5), dtype=np.int64)


            vis_depth = np.zeros((self.max_objs, 5, 7, 7), dtype=np.float32)
            att_depth = np.zeros((self.max_objs, 5, 7, 7), dtype=np.float32)
            depth_mask = np.zeros((self.max_objs,5, 7, 7), dtype=np.bool)


            object_num = len(objects) if len(objects) < self.max_objs else self.max_objs
            for i in range(object_num):
                objects____ = objects[i]
                if objects[i].cls_type not in self.writelist:
                    continue

                # print(objects[i].pos)
                objects[i].pos = Denorm_.ground_center2object_center(objects[i].pos.reshape(3,1),objects[i].h).reshape(-1)
                depth_ = objects[i].pos[-1]
                # print(objects[i].pos)
                if random_crop_flag or random_expand_flag:
                    
                    objects[i].pos[-1] = objects[i].pos[-1] * crop_scale * expand_scale

                    
                # process 2d bbox & get 2d center
                bbox_2d = objects[i].box2d.copy()
                if objects[i].level_str == 'UnKnown' or objects[i].pos[-1] < 2 or objects[i].pos[-1] > 180:
                    continue

                if random_flip_flag:
                    if self._ioa_matrix(bbox_2d,flip_box) < 0.3:
                        continue
                    [x1, _, x2, _] = bbox_2d
                    if flip_box[0] == 0:
                        bbox_2d[0] = 2 * calib.P2[0, 2] - x2
                        bbox_2d[2] = 2 * calib.P2[0, 2] - x1
                    else:
                        bbox_2d[0] = flip_box[0] + img_size[0] - x2
                        bbox_2d[2] = flip_box[0] + img_size[0] - x1
                    objects[i].ry = np.pi - objects[i].ry
                    objects[i].pos[0] *= -1
                    if objects[i].ry > np.pi:  objects[i].ry -= 2 * np.pi
                    if objects[i].ry < -np.pi: objects[i].ry += 2 * np.pi
                if random_crop_flag:
                    if self._ioa_matrix(bbox_2d,crop_box) < 0.3:
                        # a = 1
                        continue

                    bbox_2d[0] = bbox_2d[0] - crop_box[0]
                    bbox_2d[1] = bbox_2d[1] - crop_box[1]
                    bbox_2d[2] = bbox_2d[2] - crop_box[0]
                    bbox_2d[3] = bbox_2d[3] - crop_box[1]
                if random_expand_flag:
                    bbox_2d[0] = bbox_2d[0] + expand_x
                    bbox_2d[1] = bbox_2d[1] + expand_y
                    bbox_2d[2] = bbox_2d[2] + expand_x
                    bbox_2d[3] = bbox_2d[3] + expand_y

                [x1_ori, y1_ori, x2_ori, y2_ori] = bbox_2d
                x1_ori = (int)(x1_ori)
                x2_ori = (int)(x2_ori)
                y1_ori = (int)(y1_ori)
                y2_ori = (int)(y2_ori)
                width_ori = x2_ori - x1_ori
                height_ori = y2_ori - y1_ori
            
                vis_depth_roi_new = np.zeros((7, 7), dtype=np.float32)
                depth_mask_roi_new = np.ones((7, 7), dtype=np.bool)
                tan_roi = np.zeros((7, 7), dtype=np.float32)
                w_stride = width_ori / 7
                h_stride = height_ori / 7
                for w_r in range(7):
                    for h_r in range(7):
                        h_cur = (int)(np.round((0.5 + h_r) * h_stride))
                        w_cur = (int)(np.round((0.5 + w_r) * w_stride))
                        if h_cur >= height_ori:
                            h_cur = height_ori - 1
                        if w_cur >= width_ori:
                            w_cur = width_ori - 1
                        y1_cur = y1_ori + h_cur
                        x1_cur = x1_ori + w_cur
                        if y1_cur >= new_img_size[1]:
                            y1_cur = new_img_size[1] - 1
                        if x1_cur >= new_img_size[0]:
                            x1_cur = new_img_size[0] - 1
                        vis_depth_roi_new[h_r, w_r] = vis_depth_generate_new[y1_cur, x1_cur]
                        tan_roi[h_r, w_r] = (y1_cur - calib.P2[1, 2]) * 1.0 / calib.P2[1,1]
                        if vis_depth_roi_new[h_r, w_r] == 0.0:
                            depth_mask_roi_new[h_r, w_r] = False

                bbox_2d[0] = bbox_2d[0] / down_w 
                bbox_2d[1] = bbox_2d[1] / down_h
                bbox_2d[2] = bbox_2d[2] / down_w
                bbox_2d[3] = bbox_2d[3] / down_h


                # cv2.rectangle(img_cp,(int(bbox_2d[0]),int(bbox_2d[1])),(int(bbox_2d[2]),int(bbox_2d[3])),(0,255,255))
                # vis_depth_generate_new_
                # cv2.rectangle(vis_depth_generate_new_,(int(bbox_2d[0]),int(bbox_2d[1])),(int(bbox_2d[2]),int(bbox_2d[3])),(0,255,255))

                # modify the 2d bbox according to pre-compute downsample ratio
                bbox_2d[:] /= self.downsample
                center_2d = np.array([(bbox_2d[0] + bbox_2d[2]) / 2, (bbox_2d[1] + bbox_2d[3]) / 2], dtype=np.float32)

                # process 3d bbox & get 3d center
                center_3d = objects[i].pos 
                center_3d = center_3d.reshape(-1, 3)  # shape adjustment (N, 3)

                center_3d, _ = calib.rect_to_img(center_3d)  # project 3D center to image plane
                center_3d = center_3d[0]  # shape adjustment
                u_3d = center_3d[0]
                v_3d = center_3d[1]
                # print(center_3d)

                if random_crop_flag or random_expand_flag:
                    center_3d[0] = center_3d[0] * crop_scale * expand_scale
                    center_3d[1] = center_3d[1] * crop_scale * expand_scale

                center_3d[0] = center_3d[0] / down_w 
                center_3d[1] = center_3d[1] / down_h

                # cv2.circle(img_cp,(int(center_3d[0]),int(center_3d[1])),2,(0,0,255))
                # cv2.imwrite('show_train_nips/'+index+'.jpg',img_cp)
                # cv2.imwrite('show_train_nips/'+index+'.png',vis_depth_generate_new_)

                center_3d /= self.downsample      
            
                # generate the center of gaussian heatmap [optional: 3d center or 2d center]
                center_heatmap = center_3d.astype(np.int32) if self.use_3d_center else center_2d.astype(np.int32)
                if center_heatmap[0] < 0 or center_heatmap[0] >= features_size[0]: continue
                if center_heatmap[1] < 0 or center_heatmap[1] >= features_size[1]: continue
    
                # generate the radius of gaussian heatmap
                w, h = bbox_2d[2] - bbox_2d[0], bbox_2d[3] - bbox_2d[1]
                radius = gaussian_radius((w, h))
                radius = max(0, int(radius))
    

                
                if objects[i].cls_type in ['Van', 'Truck', 'DontCare']:
                    draw_umich_gaussian(heatmap[1], center_heatmap, radius)
                    continue
    
                cls_id = self.cls2id[objects[i].cls_type]
                cls_ids[i] = cls_id
                draw_umich_gaussian(heatmap[cls_id], center_heatmap, radius)
    
                # encoding 2d/3d offset & 2d size
                indices[i] = center_heatmap[1] * features_size[0] + center_heatmap[0]
                offset_2d[i] = center_2d - center_heatmap
                size_2d[i] = 1. * w, 1. * h
    
                # encoding depth
                # depth = objects[i].pos[-1]
                # depth[i] = objects[i].pos[-1]

                
                fx = calib.P2[0,0]
                fy = calib.P2[1,1]

                fp = np.sqrt(1.0/(fx * fx) + 1.0/(fy * fy)) / 1.41421356

                min_f = 2250
                max_f = 2800

                depth[i] = depth_
                vis_depth_roi = vis_depth_roi_new
                depth_mask_roi = depth_mask_roi_new
                vis_depth_roi_ind = (vis_depth_roi > depth[i] - 3) & \
                                    (vis_depth_roi < depth[i] + 3) & \
                                    (vis_depth_roi > 0)
                
                vis_depth_roi_ind = vis_depth_roi_ind * depth_mask_roi
                vis_depth_roi[~vis_depth_roi_ind] = 0
                
                norm_theta = (Denorm_.pitch_cos - Denorm_.pitch_sin * tan_roi)
                tan_center3d = (v_3d - calib.P2[1, 2]) * 1.0 / calib.P2[1,1]
                norm_theat_center3d = (Denorm_.pitch_cos - Denorm_.pitch_sin * tan_center3d)

                _crop_ratio = crop_scale * expand_scale
                


                if depth_ * _crop_ratio * fp / norm_theat_center3d < self.interval_max[0] / min_f:
                    vis_depth[i,0] = (vis_depth_roi* _crop_ratio * fp / norm_theta - (self.interval_min[0]-4.5) / max_f) / ((self.interval_max[0]+4.5) / min_f - (self.interval_min[0]-4.5) / max_f)
                    att_depth[i,0] = depth[i] -  vis_depth_roi
                    depth_bin_ind[i,0] = 1
                    depth_mask[i,0] = vis_depth_roi_ind

                if depth_ * _crop_ratio * fp / norm_theat_center3d >  self.interval_min[1] / max_f and depth_ * _crop_ratio * fp / norm_theat_center3d <  self.interval_max[1]/min_f:
                    vis_depth[i,1] = (vis_depth_roi * _crop_ratio * fp / norm_theta - ((self.interval_min[1]-4.5) / max_f)) / ((self.interval_max[1]+4.5)/ min_f - (self.interval_min[1]-4.5) / max_f)
                    att_depth[i,1] = depth[i] -  vis_depth_roi
                    depth_bin_ind[i,1] = 1
                    depth_mask[i,1] = vis_depth_roi_ind


                if depth_ * _crop_ratio * fp / norm_theat_center3d >  self.interval_min[2] / max_f and depth_ * _crop_ratio * fp / norm_theat_center3d <  self.interval_max[2]/min_f:
                    vis_depth[i,2] = (vis_depth_roi * _crop_ratio * fp / norm_theta - ((self.interval_min[2]-4.5) / max_f)) / ((self.interval_max[2]+4.5) / min_f - (self.interval_min[2]-4.5) / max_f)
                    att_depth[i,2] = depth[i] -  vis_depth_roi
                    depth_bin_ind[i,2] = 1
                    depth_mask[i,2] = vis_depth_roi_ind


                if depth_ * _crop_ratio * fp / norm_theat_center3d >  self.interval_min[3] / max_f and depth_ * _crop_ratio * fp / norm_theat_center3d <  self.interval_max[3]/min_f:
                    vis_depth[i,3] = (vis_depth_roi * _crop_ratio * fp / norm_theta - ((self.interval_min[3]-4.5) / max_f)) / ((self.interval_max[3]+4.5) / min_f - (self.interval_min[3]-4.5) / max_f)
                    att_depth[i,3] = depth[i] -  vis_depth_roi
                    depth_bin_ind[i,3] = 1
                    depth_mask[i,3] = vis_depth_roi_ind

                if depth_ * _crop_ratio * fp / norm_theat_center3d >  self.interval_min[4] / max_f and depth_  * _crop_ratio * fp / norm_theat_center3d <  self.interval_max[4]/min_f:
                    vis_depth[i,4] = (vis_depth_roi * _crop_ratio * fp / norm_theta - ((self.interval_min[4]-4.5) / max_f)) / ((self.interval_max[4]+4.5) / min_f - (self.interval_min[4]-4.5) / max_f)
                    att_depth[i,4] = depth[i] -  vis_depth_roi
                    depth_bin_ind[i,4] = 1
                    depth_mask[i,4] = vis_depth_roi_ind




                # print(depth)
                # if depth * fp < 40 / min_f:
                #     depth_mask[i,0] = 1
                #     depth_bin[i] = 0
                #     depth_offset[i,0] = (depth * fp - 0)/(40 / min_f)
                # if depth * fp > 30 / max_f and depth * fp < 70/min_f:
                #     depth_mask[i,1] = 1
                #     depth_offset[i,1] = (depth* fp - (30 / max_f)) / (70/min_f-30/max_f)
                #     depth_bin[i] = 1
                # if  depth * fp > 60 / max_f and depth * fp < 100/min_f:
                #     depth_mask[i,2] = 1
                #     depth_offset[i,2] = (depth* fp - (60 / max_f)) / (100/min_f-60/max_f)
                #     depth_bin[i] = 2
                # if depth * fp > 90 / max_f and depth * fp < 130/min_f:
                #     depth_mask[i,3] = 1
                #     depth_offset[i,3] = (depth* fp - (90 / max_f)) / (130/min_f-90/max_f)
                #     depth_bin[i] = 3
                # if depth* fp>120/max_f:
                #     depth_mask[i,4] = 1
                #     depth_offset[i,4] = (depth* fp - 120/max_f)/(180/min_f-120/max_f)
                #     depth_bin[i] = 4


    
                # encoding heading angle
                #heading_angle = objects[i].alpha
                heading_angle = calib.ry2alpha(objects[i].ry, (objects[i].box2d[0]+objects[i].box2d[2])/2)
                if heading_angle > np.pi:  heading_angle -= 2 * np.pi  # check range
                if heading_angle < -np.pi: heading_angle += 2 * np.pi
                heading_bin[i], heading_res[i] = angle2class(heading_angle)
    
                # encoding 3d offset & size_3d
                offset_3d[i] = center_3d - center_heatmap
                src_size_3d[i] = np.array([objects[i].h, objects[i].w, objects[i].l], dtype=np.float32)
                # mean_size = self.cls_mean_size[self.cls2id[objects[i].cls_type]]
                size_3d[i] = src_size_3d[i]

                #objects[i].trucation <=0.5 and objects[i].occlusion<=2 and (objects[i].box2d[3]-objects[i].box2d[1])>=25:
                if objects[i].trucation <=1 and objects[i].occlusion<=2:    
                    mask_2d[i] = 1    
            targets = {
                   'depth':depth,
                   'size_2d': size_2d,
                   'heatmap': heatmap,
                   'offset_2d': offset_2d,
                   'indices': indices,
                   'size_3d': size_3d,
                   'offset_3d': offset_3d,
                   'heading_bin': heading_bin,
                   'heading_res': heading_res,
                   'cls_ids': cls_ids,
                   'mask_2d':mask_2d,

                   'vis_depth': vis_depth,
                   'att_depth': att_depth,
                   'depth_mask': depth_mask,
                   'depth_bin_ind':depth_bin_ind}
        else:
            targets = {}
        # collect return data
        inputs = img
        info = {'img_id': index,
                'img_size': img_size,
                'bbox_downsample_ratio': img_size/features_size,}   
        return inputs, calib.P2, coord_range, targets, info, Denorm_.pitch_cos, Denorm_.pitch_sin   #calib.P2
    
    def _ioa_matrix(self, a, b):
        max_i = np.maximum(a[:2],b[:2])
        min_i = np.minimum(a[2:],b[2:])

        area_i = np.prod(min_i - max_i) * (max_i < min_i).all()
        area_a = np.prod(a[2:] - a[:2])
        # area_b = np.prod(b[2:] - b[:2], axis=1)
        # area_o = (area_a+ area_b - area_i)
        return area_i / (area_a + 1e-10)





if __name__ == '__main__':
    from torch.utils.data import DataLoader
    cfg = {'random_flip':0.0, 'random_crop':1.0, 'scale':0.4, 'shift':0.1, 'use_dontcare': False,
           'class_merging': False, 'writelist':['Pedestrian', 'Car', 'Cyclist'], 'use_3d_center':False}
    dataset = Rope3D('../../data', 'train', cfg)
    dataloader = DataLoader(dataset=dataset, batch_size=1)
    print(dataset.writelist)

    for batch_idx, (inputs, targets, info) in enumerate(dataloader):
        # test image
        img = inputs[0].numpy().transpose(1, 2, 0)
        img = (img * dataset.std + dataset.mean) * 255
        img = Image.fromarray(img.astype(np.bool))
        img.show()
        # print(targets['size_3d'][0][0])

        # test heatmap
        heatmap = targets['heatmap'][0]  # image id
        heatmap = Image.fromarray(heatmap[0].numpy() * 255)  # cats id
        heatmap.show()

        break


    # print ground truth fisrt
    objects = dataset.get_label(0)
    for object in objects:
        print(object.to_kitti_format())
