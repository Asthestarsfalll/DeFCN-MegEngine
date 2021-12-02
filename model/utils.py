from collections import namedtuple

import megengine.distributed as dist
import megengine.functional as F
from megengine import Tensor


def get_padded_tensor(
    array: Tensor, multiple_number: int = 32, pad_value: float = 0
) -> Tensor:
    """ pad the nd-array to multiple stride of th e

    Args:
        array (Tensor):
            the tensor with the shape of [batch, channel, height, width]
        multiple_number (int):
            make the height and width can be divided by multiple_number
        pad_value (int): the value to be padded

    Returns:
        padded_array (Tensor)
    """
    batch, chl, t_height, t_width = array.shape
    padded_height = (
        (t_height + multiple_number - 1) // multiple_number * multiple_number
    )
    padded_width = (t_width + multiple_number -
                    1) // multiple_number * multiple_number

    padded_array = F.full(
        (batch, chl, padded_height, padded_width), pad_value, dtype=array.dtype
    )

    ndim = array.ndim
    if ndim == 4:
        padded_array[:, :, :t_height, :t_width] = array
    elif ndim == 3:
        padded_array[:, :t_height, :t_width] = array
    else:
        raise Exception("Not supported tensor dim: %d" % ndim)
    return padded_array


class ShapeSpec(namedtuple("_ShapeSpec", ["channels", "height", "width", "stride"])):
    """
    A simple structure that contains basic shape specification about a tensor.
    Useful for getting the modules output channels when building the graph.
    """

    def __new__(cls, channels=None, height=None, width=None, stride=None):
        return super().__new__(cls, channels, height, width, stride)


def cat(tensor, axis=0):
    if len(tensor) == 1:
        return tensor[0]
    return F.concat(tensor, axis)


def permute_to_N_HWA_K(tensor, K):
    """
    Transpose/reshape a tensor from (N, (A x K), H, W) to (N, (HxWxA), K)
    """
    assert len(tensor.shape) == 4, tensor.shape
    N, _, H, W = tensor.shape
    tensor = tensor.transpose(0, 2, 3, 1).reshape(N, -1, K)  # Size=(N, HWA, K)
    return tensor


def all_reduce_mean(array: Tensor) -> Tensor:
    if dist.get_world_size() > 1:
        array = dist.functional.all_reduce_sum(array) / dist.get_world_size()
    return array


def get_clipped_boxes(boxes, hw):
    """ Clip the boxes into the image region."""
    # x1 >=0
    box_x1 = F.clip(boxes[:, 0::4], lower=0, upper=hw[1])
    # y1 >=0
    box_y1 = F.clip(boxes[:, 1::4], lower=0, upper=hw[0])
    # x2 < im_info[1]
    box_x2 = F.clip(boxes[:, 2::4], lower=0, upper=hw[1])
    # y2 < im_info[0]
    box_y2 = F.clip(boxes[:, 3::4], lower=0, upper=hw[0])

    clip_box = F.concat([box_x1, box_y1, box_x2, box_y2], axis=1)

    return clip_box
