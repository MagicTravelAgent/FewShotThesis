import torch

class Evaluator:
    r""" Computes intersection and union between prediction and ground-truth """
    @classmethod
    def initialize(cls):
        cls.ignore_index = 255

    @classmethod
    def iou_precent_y(cls, union, intersection, pred_mask, y_mask):
        if union == 0:
            iou = 0
        else:
            iou = intersection / union

        if (pred_mask < 1):
            percent_y = 0
        else:
            percent_y = intersection / (y_mask + 1)
        return iou, percent_y

    @classmethod
    def classify_prediction(cls, pred_mask, batch):
        gt_mask = batch.get('query_mask')
        query_name = batch.get("query_name")
        output_dict = {}
        output_dict["neg_inst"] = batch.get("neg_inst").tolist()[0]
        # here collect information about the query mask: size, globules (and their size), 
        output_dict["query_name"] = query_name[0]

        # Apply ignore_index in PASCAL-5i masks (following evaluation scheme in PFE-Net (TPAMI 2020))
        if not output_dict["neg_inst"]:
            query_ignore_idx = batch.get('query_ignore_idx')
            if query_ignore_idx is not None:
                assert torch.logical_and(query_ignore_idx, gt_mask).sum() == 0
                query_ignore_idx *= cls.ignore_index
                gt_mask = gt_mask + query_ignore_idx
                pred_mask[gt_mask == cls.ignore_index] = cls.ignore_index
                

        # compute intersection and union of each episode in a batch
        area_inter, area_pred, area_gt = [],  [], []
        for _pred_mask, _gt_mask in zip(pred_mask, gt_mask):

            _inter = _pred_mask[_pred_mask == _gt_mask]

            if _inter.size(0) == 0:  # as torch.histc returns error if it gets empty tensor (pytorch 1.5.1)
                _area_inter = torch.tensor([0, 0], device=_pred_mask.device)
            else:
                _area_inter = torch.histc(_inter, bins=2, min=0, max=1)
            area_inter.append(_area_inter)
            area_pred.append(torch.histc(_pred_mask, bins=2, min=0, max=1))
            area_gt.append(torch.histc(_gt_mask, bins=2, min=0, max=1))
        area_inter = torch.stack(area_inter).t()
        area_pred = torch.stack(area_pred).t()
        area_gt = torch.stack(area_gt).t()
        area_union = area_pred + area_gt - area_inter
        
        output_dict["pred_mask_size_foreground"] = area_pred.tolist()[1][0]
        output_dict["pred_mask_size_background"] = area_pred.tolist()[0][0]
        
        output_dict["y_mask_size_foreground"] = area_gt.tolist()[1][0]
        output_dict["y_mask_size_background"] = area_pred.tolist()[0][0]

        output_dict["intersection_size_foreground"] = area_inter.tolist()[1][0]
        output_dict["intersection_size_background"] = area_inter.tolist()[0][0]

        output_dict["wrong_pred_size_foreground"] =  output_dict["pred_mask_size_foreground"] - output_dict["intersection_size_foreground"]
        output_dict["wrong_pred_size_background"] =  output_dict["pred_mask_size_background"] - output_dict["intersection_size_background"]

        output_dict["union_foreground"] = output_dict["y_mask_size_foreground"] + output_dict["pred_mask_size_foreground"] - output_dict["intersection_size_foreground"]
        output_dict["union_background"] = output_dict["y_mask_size_background"] + output_dict["pred_mask_size_background"] - output_dict["intersection_size_background"]
        
        output_dict["IOU_foreground"], output_dict["percent_y_foreground"] = cls.iou_precent_y(output_dict["union_foreground"], output_dict["intersection_size_foreground"], output_dict["pred_mask_size_foreground"], output_dict["y_mask_size_foreground"])
        output_dict["IOU_background"], output_dict["percent_y_background"] = cls.iou_precent_y(output_dict["union_background"], output_dict["intersection_size_background"], output_dict["pred_mask_size_background"], output_dict["y_mask_size_background"])

        # return a dictionary containing the whole evaluation
        return area_inter, area_union, output_dict