import os

from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import cv2

from docs.HSNet.Common import Utils

class Visualizer:

    @classmethod
    def initialize(cls, visualize, path):
        cls.visualize = visualize
        if not visualize:
            return

        cls.colors = {'red': (255, 50, 50), 'blue': (102, 140, 255)}
        for key, value in cls.colors.items():
            cls.colors[key] = tuple([c / 255 for c in cls.colors[key]])

        cls.mean_img = [0.485, 0.456, 0.406]
        cls.std_img = [0.229, 0.224, 0.225]
        cls.to_pil = transforms.ToPILImage()
        cls.vis_path = "logs/"+path + '.log/vis/'
        if not os.path.exists(cls.vis_path): os.makedirs(cls.vis_path)

    @classmethod
    def visualize_prediction_batch(cls, spt_img_b, spt_mask_b, qry_img_b, qry_mask_b, pred_mask_b, batch_idx, iou_b=None):
        spt_img_b = Utils.to_cpu(spt_img_b)
        spt_mask_b = Utils.to_cpu(spt_mask_b)
        qry_img_b = Utils.to_cpu(qry_img_b)
        qry_mask_b = Utils.to_cpu(qry_mask_b)
        pred_mask_b = Utils.to_cpu(pred_mask_b)
        iou_b = Utils.to_cpu(iou_b[0])
        # cls_id_b = to_cpu(cls_id_b)

        for sample_idx, (spt_img, spt_mask, qry_img, qry_mask, pred_mask) in \
                enumerate(zip(spt_img_b, spt_mask_b, qry_img_b, qry_mask_b, pred_mask_b)):
            iou = iou_b[sample_idx] if iou_b is not None else None
            cls.visualize_prediction(spt_img, spt_mask, qry_img, qry_mask, pred_mask, batch_idx, sample_idx, True, iou)

    @classmethod
    def to_numpy(cls, tensor, type):
        if type == 'img':
            return np.array(cls.to_pil(cls.unnormalize(tensor))).astype(np.uint8)
        elif type == 'mask':
            return np.array(tensor).astype(np.uint8)
        else:
            raise Exception('Undefined tensor type: %s' % type)

    @classmethod
    def visualize_prediction(cls, spt_imgs, spt_masks, qry_img, qry_mask, pred_mask, batch_idx, sample_idx, label, iou=None):

        spt_color = cls.colors['blue']
        qry_color = cls.colors['red']
        pred_color = cls.colors['red']

        spt_imgs = [cls.to_numpy(spt_img, 'img') for spt_img in spt_imgs]
        spt_pils = [cls.to_pil(spt_img) for spt_img in spt_imgs]
        spt_masks = [cls.to_numpy(spt_mask, 'mask') for spt_mask in spt_masks]
        spt_masked_pils = [Image.fromarray(cls.apply_mask(spt_img, spt_mask, spt_color)) for spt_img, spt_mask in zip(spt_imgs, spt_masks)]

        qry_img = cls.to_numpy(qry_img, 'img')
        qry_pil = cls.to_pil(qry_img)
        qry_mask = cls.to_numpy(qry_mask, 'mask')
        pred_mask = cls.to_numpy(pred_mask, 'mask')
        pred_masked_pil = Image.fromarray(cls.apply_mask(qry_img.astype(np.uint8), pred_mask.astype(np.uint8), pred_color))
        qry_masked_pil = Image.fromarray(cls.apply_mask(qry_img.astype(np.uint8), qry_mask.astype(np.uint8), qry_color))

        merged_pil = cls.merge_image_pair(spt_masked_pils + [pred_masked_pil, qry_masked_pil])

        iou = iou.item() if iou else 0.0
        merged_pil.save(cls.vis_path + '%d_%d_iou-%.2f' % (batch_idx, sample_idx, iou) + '.jpg')

    @classmethod
    def visualize_prediction_CNN(cls, idx, qry_img, qry_mask, pred_mask, iou=None):

        qry_color = cls.colors['red']
        pred_color = cls.colors['red']
        
        qry_img = cls.to_numpy(qry_img, 'img')
        qry_mask = cls.to_numpy(qry_mask, 'mask')

        # pred_mask.show()
        pred_mask = np.array(pred_mask)/255

        qry_masked_pil = Image.fromarray(cls.apply_mask(qry_img.astype(np.uint8), qry_mask.astype(np.uint8), qry_color))
        pred_masked_pil = Image.fromarray(cls.apply_mask(qry_img.astype(np.uint8), pred_mask.astype(np.uint8), pred_color))
        
        merged_pil = cls.merge_image_pair([pred_masked_pil, qry_masked_pil])

        iou = iou.item() if iou else 0.0
        merged_pil.save(cls.vis_path + '%d_CNN_iou-%.2f' % (idx, iou) + '.jpg')

    @classmethod
    def merge_image_pair(cls, pil_imgs):
        r""" Horizontally aligns a pair of pytorch tensor images (3, H, W) and returns PIL object """

        canvas_width = sum([pil.size[0] for pil in pil_imgs])
        canvas_height = max([pil.size[1] for pil in pil_imgs])
        canvas = Image.new('RGB', (canvas_width, canvas_height))

        xpos = 0
        for pil in pil_imgs:
            canvas.paste(pil, (xpos, 0))
            xpos += pil.size[0]

        return canvas

    @classmethod
    def apply_mask(cls, image, mask, color, alpha=0.5):
        r""" Apply mask to the given image. """

        # resizing to the correct size ((which they didnt test?))
        mask = Image.fromarray(mask)
        mask = mask.resize((400, 400), Image.NEAREST)
        mask = np.array(mask)

        for c in range(3):
            image[:, :, c] = np.where(mask == 1,
                                      image[:, :, c] *
                                      (1 - alpha) + alpha * color[c] * 255,
                                      image[:, :, c])
        return image

    @classmethod
    def unnormalize(cls, img):
        img = img.clone()
        for im_channel, mean, std in zip(img, cls.mean_img, cls.std_img):
            im_channel.mul_(std).add_(mean)
        return img