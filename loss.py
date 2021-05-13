import torch 
import torch.nn as nn 
from utils import intersection_over_union 

class YoloLoss(nn.Module):
    def __init__(self, S=7,B=2,C=20):
        super(YoloLoss,self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.S = S
        self.B=B
        self.C=C
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        predictions=predictions.reshape(-1, self.S, self.S, self.B*5+self.C)
        # Note: here we only have one targer. We compare both predicted box with the same x,y,w,h from the last five channels in targets 
        iou_b1 = intersection_over_union(predictions[..., 21:25], target[...,21:25])
        iou_b2 = intersection_over_union(predictions[..., 26:30], target[...,21:25])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
        iou_maxes, bestbox_idx = torch.max(ious, dim=0) # bestbox_idx is the index to indicate which box is response to predict 
        exists_obj = target[...,20].unsqueeze(3) #to keep the third dimension. Is there an object is cell i? n*7*7*1 
        
        #coord loss 
        box_predictions = exists_obj * (
            (       #get the best box
                bestbox_idx * predictions[..., 26:30] + (1-bestbox_idx) * predictions[...,21:25] #bestbox_idx suppose to be 0 or 1, index to indicate whcih box to use 
            )
        )
        box_targets = exists_obj * target[...,21:25]
        box_predictions[...,2:4] = torch.sign(box_predictions[...,2:4]) * torch.sqrt(torch.abs(box_predictions[...,2:4] + 1e-6)) # handle possible negative values 

        box_targets[...,2:4] = torch.sqrt(box_targets[..., 2:4])
        # (N,S,S,4) -> (N*S*S, 4)
        box_loss = self.mse(torch.flatten(box_predictions, end_dim=-2), 
                            torch.flatten(box_targets, end_dim=-2), )

        #obj loss
        pred_box = (bestbox_idx * predictions[..., 25:26] + (1-bestbox_idx) * predictions[...,20:21])
        obj_loss = self.mse(
            torch.flatten(exists_obj*pred_box),
            torch.flatten(exists_obj*target[..., 20:21])
        )

        no_obj_loss = self.mse(
                torch.flatten(
                    (
                        1- exists_obj) * predictions[...,25:26], start_dim=1
                        ), 
                        torch.flatten((1-exists_obj)*target[...,20:21], start_dim=1
                        )
        ) 
        
        no_obj_loss += self.mse(
             torch.flatten(
                 (
                     1- exists_obj) * predictions[...,20:21], start_dim=1
                     ), 
                     torch.flatten((1-exists_obj)*target[...,20:21], start_dim=1
                     )
        )        
        #class loss 
        class_loss = self.mse(
            torch.flatten(exists_obj * predictions[...,:20], end_dim=-2), 
            torch.flatten(exists_obj*target[...,:20], end_dim=-2)
        )


        loss = (
            self.lambda_coord * box_loss 
            + obj_loss
            + self.lambda_noobj * no_obj_loss
            + class_loss
        )
        return loss 
    

