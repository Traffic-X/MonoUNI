import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.helpers.decode_helper import _transpose_and_gather_feat
from lib.losses.focal_loss import focal_loss_cornernet as focal_loss
from lib.losses.uncertainty_loss import laplacian_aleatoric_uncertainty_loss
import operator

class Hierarchical_Task_Learning:
    def __init__(self,epoch0_loss,stat_epoch_nums=5):
        self.index2term = [*epoch0_loss.keys()]
        self.term2index = {term:self.index2term.index(term) for term in self.index2term}  #term2index
        self.stat_epoch_nums = stat_epoch_nums
        self.past_losses=[]
        self.loss_graph = {'seg_loss':[],
                           'size2d_loss':[], 
                           'offset2d_loss':[],
                           'offset3d_loss':['size2d_loss','offset2d_loss'], 
                           'size3d_loss':['size2d_loss','offset2d_loss'], 
                           'heading_loss':['size2d_loss','offset2d_loss'], 
                           'depth_loss':['size2d_loss','size3d_loss','offset2d_loss']}                                 
    def compute_weight(self,current_loss,epoch):
        T=150
        #compute initial weights
        loss_weights = {}
        eval_loss_input = torch.cat([_.unsqueeze(0) for _ in current_loss.values()]).unsqueeze(0)
        for term in self.loss_graph:
            if len(self.loss_graph[term])==0:
                loss_weights[term] = torch.tensor(1.0).to(current_loss[term].device)
            else:
                loss_weights[term] = torch.tensor(0.0).to(current_loss[term].device) 
        #update losses list
        if len(self.past_losses)==self.stat_epoch_nums:
            past_loss = torch.cat(self.past_losses)
            mean_diff = (past_loss[:-2]-past_loss[2:]).mean(0)
            if not hasattr(self, 'init_diff'):
                self.init_diff = mean_diff
            c_weights = 1-(mean_diff/self.init_diff).relu().unsqueeze(0)
            
            time_value = min(((epoch-5)/(T-5)),1.0)
            for current_topic in self.loss_graph:
                if len(self.loss_graph[current_topic])!=0:
                    control_weight = 1.0
                    for pre_topic in self.loss_graph[current_topic]:
                        control_weight *= c_weights[0][self.term2index[pre_topic]]      
                    loss_weights[current_topic] = time_value**(1-control_weight)
                    if loss_weights[current_topic] !=loss_weights[current_topic]:
                        print('Nan')
                        loss_weights[current_topic] = torch.ones(1)
            #pop first list
            self.past_losses.pop(0)
        self.past_losses.append(eval_loss_input)   
        return loss_weights
    def update_e0(self,eval_loss):
        self.epoch0_loss = torch.cat([_.unsqueeze(0) for _ in eval_loss.values()]).unsqueeze(0)


class GupnetLoss(nn.Module):
    def __init__(self,epoch):
        super().__init__()
        self.stat = {}
        self.epoch = epoch


    def forward(self, preds, targets, task_uncertainties=None):

        seg_loss = self.compute_segmentation_loss(preds, targets)
        bbox2d_loss = self.compute_bbox2d_loss(preds, targets)
        bbox3d_loss = self.compute_bbox3d_loss(preds, targets)
        
        loss = seg_loss + bbox2d_loss + bbox3d_loss
        
        return loss, self.stat


    def compute_segmentation_loss(self, input, target):
        input['heatmap'] = torch.clamp(input['heatmap'].sigmoid_(), min=1e-4, max=1 - 1e-4)
        loss = focal_loss(input['heatmap'], target['heatmap'])
        self.stat['seg_loss'] = loss
        return loss


    def compute_bbox2d_loss(self, input, target):
        # compute size2d loss
        
        size2d_input = extract_input_from_tensor(input['size_2d'], target['indices'], target['mask_2d'])
        size2d_target = extract_target_from_tensor(target['size_2d'], target['mask_2d'])
        size2d_loss = F.l1_loss(size2d_input, size2d_target, reduction='mean')
        # compute offset2d loss
        offset2d_input = extract_input_from_tensor(input['offset_2d'], target['indices'], target['mask_2d'])
        offset2d_target = extract_target_from_tensor(target['offset_2d'], target['mask_2d'])
        offset2d_loss = F.l1_loss(offset2d_input, offset2d_target, reduction='mean')


        loss = offset2d_loss + size2d_loss   
        self.stat['offset2d_loss'] = offset2d_loss
        self.stat['size2d_loss'] = size2d_loss
        return loss


    def compute_bbox3d_loss(self, input, target, mask_type = 'mask_2d'):
        vis_depth = input['vis_depth'][input['train_tag']]
        att_depth = input['att_depth'][input['train_tag']]
        vis_depth_target = extract_target_from_tensor(target['vis_depth'], target[mask_type])
        att_offset_target = extract_target_from_tensor(target['att_depth'], target[mask_type])
        depth_mask_target = extract_target_from_tensor(target['depth_mask'], target[mask_type])
        ins_depth_target = extract_target_from_tensor(target['depth'], target[mask_type])
        
        vis_depth_uncer = input['vis_depth_uncer'][input['train_tag']]
        att_depth_uncer = input['att_depth_uncer'][input['train_tag']]

        # show_vis = vis_depth[depth_mask_target]
        # show_vis_target = vis_depth_target[depth_mask_target]
        # show_att = att_depth[depth_mask_target]
        # show_att_target = att_offset_target[depth_mask_target]




        vis_depth_loss = laplacian_aleatoric_uncertainty_loss(vis_depth[depth_mask_target],
                                                              vis_depth_target[depth_mask_target],
                                                              vis_depth_uncer[depth_mask_target],bili = 40)

        att_depth_loss = laplacian_aleatoric_uncertainty_loss(att_depth[depth_mask_target],
                                                              att_offset_target[depth_mask_target],
                                                              att_depth_uncer[depth_mask_target])

        ins_depth = input['ins_depth'][input['train_tag']]
        ins_depth_uncer = input['ins_depth_uncer'][input['train_tag']]
        # show_ins = ins_depth_target
        # show_ins_target = ins_depth[depth_mask_target]
        ins_depth = ins_depth.contiguous()
        ins_depth_uncer = ins_depth_uncer.contiguous()
        ins_depth = ins_depth.view(ins_depth.shape[0],-1)
        ins_depth_loss = laplacian_aleatoric_uncertainty_loss(ins_depth.view(ins_depth.shape[0],-1),
                                                                ins_depth_target.repeat(1,ins_depth.shape[-1]),
                                                                ins_depth_uncer.view(ins_depth.shape[0],-1),reduction='none')

        
        if True:
            depth_bin =input['depth_bin'][input['train_tag']]
            depth_bin_target = extract_target_from_tensor(target['depth_bin_ind'], target[mask_type])

            ins_depth_loss = ins_depth_loss.view(ins_depth.shape[0],5,-1)
            ins_depth_loss = torch.mean(ins_depth_loss,-1)
            ins_depth_loss = torch.mean(depth_bin_target * ins_depth_loss) 
            depth_bin =input['depth_bin'][input['train_tag']]

            depth_bin_target = extract_target_from_tensor(target['depth_bin_ind'], target[mask_type])
            depth_bin_loss_0 = F.cross_entropy(depth_bin[:,:2],depth_bin_target[:,0]) 
            depth_bin_loss_1 = F.cross_entropy(depth_bin[:,2:4],depth_bin_target[:,1]) 
            depth_bin_loss_2 = F.cross_entropy(depth_bin[:,4:6],depth_bin_target[:,2]) 
            depth_bin_loss_3 = F.cross_entropy(depth_bin[:,6:8],depth_bin_target[:,3]) 
            depth_bin_loss_4 = F.cross_entropy(depth_bin[:,8:10],depth_bin_target[:,4]) 
            depth_bin_loss = depth_bin_loss_0 + depth_bin_loss_1 + depth_bin_loss_2 +depth_bin_loss_3 + depth_bin_loss_4

        depth_loss = vis_depth_loss + att_depth_loss + ins_depth_loss + depth_bin_loss
        depth_loss = depth_loss * 10

        #print(depth_offset[:5])
        #print(depth_offset_target[:5])
        # compute offset3d loss
        offset3d_input = input['offset_3d'][input['train_tag']]  
        offset3d_target = extract_target_from_tensor(target['offset_3d'], target[mask_type])
        
        offset3d_loss = F.l1_loss(offset3d_input, offset3d_target, reduction='mean') * 10
        #print(offset3d_input[0:5],'\n',offset3d_target[0:5])
        # compute size3d loss
        size3d_input = input['size_3d'][input['train_tag']] 
        size3d_target = extract_target_from_tensor(target['size_3d'], target[mask_type])
        # print
        # print(size3d_input[:5],'\n',size3d_target[:5])
        size3d_loss = F.l1_loss(size3d_input, size3d_target, reduction='mean') * 10 
        # size3d_loss = F.l1_loss(size3d_input[:,1:], size3d_target[:,1:], reduction='mean')*2/3+\
        #        laplacian_aleatoric_uncertainty_loss(size3d_input[:,0:1], size3d_target[:,0:1], input['h3d_log_variance'][input['train_tag']])/3
        
        
        
        # compute heading loss
        heading_loss = compute_heading_loss(input['heading'][input['train_tag']] ,
                                            target[mask_type],  ## NOTE
                                            target['heading_bin'],
                                            target['heading_res'])
        loss = depth_loss + offset3d_loss + size3d_loss + heading_loss
        
        self.stat['depth_loss'] = depth_loss
        self.stat['offset3d_loss'] = offset3d_loss
        self.stat['size3d_loss'] = size3d_loss
        self.stat['heading_loss'] = heading_loss 
        return loss




### ======================  auxiliary functions  =======================

def extract_input_from_tensor(input, ind, mask):
    input = _transpose_and_gather_feat(input, ind)  # B*C*H*W --> B*K*C
    return input[mask]  # B*K*C --> M * C


def extract_target_from_tensor(target, mask):
    return target[mask]

#compute heading loss two stage style  

def compute_heading_loss(input, mask, target_cls, target_reg):
    mask = mask.view(-1)   # B * K  ---> (B*K)
    target_cls = target_cls.view(-1)  # B * K * 1  ---> (B*K)
    target_reg = target_reg.view(-1)  # B * K * 1  ---> (B*K)

    # classification loss
    input_cls = input[:, 0:12]
    target_cls = target_cls[mask]
    # print(input_cls.shape)
    # print('target_cls',target_cls.shape)
    
    
    # regression loss
    input_reg = input[:, 12:24]
    target_reg = target_reg[mask]
    cls_onehot = torch.zeros(target_cls.shape[0], 12).cuda().scatter_(dim=1, index=target_cls.view(-1, 1), value=1)
    input_reg = torch.sum(input_reg * cls_onehot, 1)
    reg_loss = F.l1_loss(input_reg, target_reg, reduction='mean')
    
    cls_loss = reg_loss
    if target_cls.shape[0]!=0:
        cls_loss = F.cross_entropy(input_cls, target_cls, reduction='mean')
    else:
        print('error')
        # print(target_cls.shape)
        # return reg_loss
    return cls_loss + reg_loss    
'''    

def compute_heading_loss(input, ind, mask, target_cls, target_reg):
    """
    Args:
        input: features, shaped in B * C * H * W
        ind: positions in feature maps, shaped in B * 50
        mask: tags for valid samples, shaped in B * 50
        target_cls: cls anns, shaped in B * 50 * 1
        target_reg: reg anns, shaped in B * 50 * 1
    Returns:
    """
    input = _transpose_and_gather_feat(input, ind)   # B * C * H * W ---> B * K * C
    input = input.view(-1, 24)  # B * K * C  ---> (B*K) * C
    mask = mask.view(-1)   # B * K  ---> (B*K)
    target_cls = target_cls.view(-1)  # B * K * 1  ---> (B*K)
    target_reg = target_reg.view(-1)  # B * K * 1  ---> (B*K)

    # classification loss
    input_cls = input[:, 0:12]
    input_cls, target_cls = input_cls[mask], target_cls[mask]
    cls_loss = F.cross_entropy(input_cls, target_cls, reduction='mean')

    # regression loss
    input_reg = input[:, 12:24]
    input_reg, target_reg = input_reg[mask], target_reg[mask]
    cls_onehot = torch.zeros(target_cls.shape[0], 12).cuda().scatter_(dim=1, index=target_cls.view(-1, 1), value=1)
    input_reg = torch.sum(input_reg * cls_onehot, 1)
    reg_loss = F.l1_loss(input_reg, target_reg, reduction='mean')
    
    return cls_loss + reg_loss
'''


if __name__ == '__main__':
    input_cls  = torch.zeros(2, 50, 12)  # B * 50 * 24
    input_reg  = torch.zeros(2, 50, 12)  # B * 50 * 24
    target_cls = torch.zeros(2, 50, 1, dtype=torch.int64)
    target_reg = torch.zeros(2, 50, 1)

    input_cls, target_cls = input_cls.view(-1, 12), target_cls.view(-1)
    cls_loss = F.cross_entropy(input_cls, target_cls, reduction='mean')

    a = torch.zeros(2, 24, 10, 10)
    b = torch.zeros(2, 10).long()
    c = torch.ones(2, 10).long()
    d = torch.zeros(2, 10, 1).long()
    e = torch.zeros(2, 10, 1)
    print(compute_heading_loss(a, b, c, d, e))

