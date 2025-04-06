import torch


def box_cxcywh_to_xyxy(x):
    """
    Convert bounding boxes from [cx, cy, w, h] to [x1, y1, x2, y2] format.
    
    Args:
        x: Tensor of shape (..., 4) in [cx, cy, w, h] format
        
    Returns:
        Tensor of shape (..., 4) in [x1, y1, x2, y2] format
    """
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    """
    Convert bounding boxes from [x1, y1, x2, y2] to [cx, cy, w, h] format.
    
    Args:
        x: Tensor of shape (..., 4) in [x1, y1, x2, y2] format
        
    Returns:
        Tensor of shape (..., 4) in [cx, cy, w, h] format
    """
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def box_iou(boxes1, boxes2):
    """
    Compute intersection over union (IoU) between pairs of boxes.
    
    Args:
        boxes1: Tensor of shape (N, 4) in [x1, y1, x2, y2] format
        boxes2: Tensor of shape (M, 4) in [x1, y1, x2, y2] format
        
    Returns:
        iou: Tensor of shape (N, M) with IoU values for each pair
        union: Tensor of shape (N, M) with union areas
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    
    # Calculate intersection coordinates
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]
    
    # Calculate intersection areas
    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    intersection = wh[:, :, 0] * wh[:, :, 1]  # [N, M]
    
    # Calculate union areas
    union = area1[:, None] + area2 - intersection
    
    # Calculate IoU
    iou = intersection / union
    
    return iou, union


def box_area(boxes):
    """
    Compute area of bounding boxes.
    
    Args:
        boxes: Tensor of shape (..., 4) in [x1, y1, x2, y2] format
        
    Returns:
        Tensor of shape (...) with areas
    """
    return (boxes[..., 2] - boxes[..., 0]) * (boxes[..., 3] - boxes[..., 1])


def clip_boxes(boxes, image_size):
    """
    Clip boxes to image boundaries.
    
    Args:
        boxes: Tensor of shape (N, 4) in [x1, y1, x2, y2] format
        image_size: Tuple (height, width)
        
    Returns:
        Clipped boxes
    """
    h, w = image_size
    boxes = boxes.clone()
    boxes[..., 0].clamp_(min=0, max=w)
    boxes[..., 1].clamp_(min=0, max=h)
    boxes[..., 2].clamp_(min=0, max=w)
    boxes[..., 3].clamp_(min=0, max=h)
    return boxes 