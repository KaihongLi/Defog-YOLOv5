"""
    Created on Mon Mar 06 2023
    The DIP(differentiable image process module) consists of six different filters with adjustable hyperparameters,
    include Defog, White Balance(WB), Gamma, Contrast, Tone and Sharpen. These are implemented by PyTorch.
    Original from : IA-YOLO(https://github.com/wenyyu/Image-Adaptive-YOLO)
"""
import cv2
import torch
import math
import numpy as np
import torch.nn.functional as F


def DarkChannel(im):
    b, g, r = torch.split(im, 1, dim=0)
    dc = torch.min(torch.min(r, g), b)
    return dc


def AtmLight(im, dark):
    [h, w] = im.shape[1:]
    imsz = h * w
    numpx = int(max(math.floor(imsz / 1000), 1))
    # if numpx == 1:
    #     numpx = int(math.floor(imsz / 2))
    # else:
    #     numpx = 1000
    darkvec = dark.reshape(1, imsz)
    imvec = im.reshape(3, imsz)

    indices = darkvec.argsort(1)
    indices = indices[0][(imsz - numpx):imsz]
    atmsum = torch.zeros([3]).to(im.device)
    for ind in range(0, numpx):
        atmsum = atmsum + imvec[:, indices[ind]]

    A = atmsum / numpx
    return A.unsqueeze(axis=-1)


def DarkIcA(im, A):
    im3 = torch.empty(im.shape, dtype=im.dtype).to(im.device)
    for ind in range(0, 3):
        im3[ind, :, :] = im[ind, :, :] / A[0, ind]
    return DarkChannel(im3)


# # im (8, 3, 640, 640)
# # b, g, r (8, 1, 640, 640)
# def DarkChannel(im):
#     b, g, r = torch.split(im, 1, dim=1)
#     dc = torch.min(torch.min(r, g), b)
#     return dc
#
# # im (8, 3, 640, 640)
# # dark (8, 1, 640, 640)
# # atmsum (8, 3)
# def AtmLight(im, dark, atmsum):
#     [h, w] = im.shape[2:]
#     imsz = h * w
#     numpx = int(max(math.floor(imsz / 1000), 1))
#     if numpx == 1:
#         numpx = int(math.floor(imsz / 2))
#     else:
#         numpx = int(math.floor(imsz / 256))
#     darkvec = dark.reshape(dark.shape[0], 1, imsz)
#     imvec = im.reshape(im.shape[0], 3, imsz)
#
#     indices = darkvec.argsort(2)
#     indices = indices[:, 0, (imsz - numpx):imsz]
#     for x in range(0, im.shape[0]):
#         for ind in range(0, numpx):
#             atmsum[x] = atmsum[x] + imvec[x, :, indices[x, ind]]
#
#     A = atmsum / numpx
#     return A
#
#
# def DarkIcA(im, A, im3):
#     for x in range(0, im.shape[0]):
#         for ind in range(0, 3):
#             im3[x, ind, :, :] = im[x, ind, :, :] / A[x, ind]
#     return DarkChannel(im3)


def rgb2lum(image):
    image = 0.27 * image[:, :, :, 0] + 0.67 * image[:, :, :,
                                              1] + 0.06 * image[:, :, :, 2]
    return image[:, :, :, None]


def tanh01(x):
    return torch.tanh(x) * 0.5 + 0.5


def tanh_range(l, r, initial=None):
    def get_activation(left, right, initial):

        def activation(x):
            if initial is not None:
                bias = math.atanh(2 * (initial - left) / (right - left) - 1)
            else:
                bias = 0
            return tanh01(x + bias) * (right - left) + left

        return activation

    return get_activation(l, r, initial)


def lerp(a, b, l):
    return (1 - l) * a + l * b


class Filter:

    def __init__(self, cfg):
        self.cfg = cfg
        # self.height, self.width, self.channels = list(map(int, net.get_shape()[1:]))

        # Specified in child classes
        self.num_filter_parameters = None
        self.short_name = None
        self.filter_parameters = None

    def get_short_name(self):
        assert self.short_name
        return self.short_name

    def get_num_filter_parameters(self):
        assert self.num_filter_parameters
        return self.num_filter_parameters

    def get_begin_filter_parameter(self):
        return self.begin_filter_parameter

    def extract_parameters(self, features):
        # output_dim = self.get_num_filter_parameters(
        # ) + self.get_num_mask_parameters()
        # features = ly.fully_connected(
        #     features,
        #     self.cfg.fc1_size,
        #     scope='fc1',
        #     activation_fn=lrelu,
        #     weights_initializer=tf.contrib.layers.xavier_initializer())
        # features = ly.fully_connected(
        #     features,
        #     output_dim,
        #     scope='fc2',
        #     activation_fn=None,
        #     weights_initializer=tf.contrib.layers.xavier_initializer())
        indexs = range(self.get_begin_filter_parameter(), self.get_begin_filter_parameter() + self.get_num_filter_parameters())
        return features[:, indexs], features[:, indexs]

    # Should be implemented in child classes
    def filter_param_regressor(self, features):
        assert False

    # Process the whole image, without masking
    # Should be implemented in child classes
    def process(self, img, param, defog, IcA):
        assert False

    def debug_info_batched(self):
        return False

    def no_high_res(self):
        return False

    # Apply the whole filter with masking
    def apply(self,
              img,
              img_features=None,
              defog_A=None,
              IcA=None,
              specified_parameter=None,
              high_res=None):
        assert (img_features is None) ^ (specified_parameter is None)
        if img_features is not None:
            filter_features, mask_parameters = self.extract_parameters(img_features)
            filter_parameters = self.filter_param_regressor(filter_features)
        else:
            assert not self.use_masking()
            filter_parameters = specified_parameter
            mask_parameters = torch.zeros([1, self.get_num_mask_parameters()], dtype=torch.float32)
        if high_res is not None:
            # working on high res...
            pass
        debug_info = {}
        # We only debug the first image of this batch
        if self.debug_info_batched():
            debug_info['filter_parameters'] = filter_parameters
        else:
            debug_info['filter_parameters'] = filter_parameters[0]
        # self.mask_parameters = mask_parameters
        # self.mask = self.get_mask(img, mask_parameters)
        # debug_info['mask'] = self.mask[0]
        # low_res_output = lerp(img, self.process(img, filter_parameters), self.mask)
        low_res_output = self.process(img, filter_parameters, defog_A, IcA)

        if high_res is not None:
            if self.no_high_res():
                high_res_output = high_res
            else:
                self.high_res_mask = self.get_mask(high_res, mask_parameters)
                # high_res_output = lerp(high_res,
                #                        self.process(high_res, filter_parameters, defog, IcA),
                #                        self.high_res_mask)
        else:
            high_res_output = None
        # return low_res_output, high_res_output, debug_info
        return low_res_output, filter_parameters

    def use_masking(self):
        return self.cfg['masking']

    def get_num_mask_parameters(self):
        return 6

    # Input: no need for tanh or sigmoid
    # Closer to 1 values are applied by filter more strongly
    def get_mask(self, img, mask_parameters):
        if not self.use_masking():
            print('* Masking Disabled')
            return torch.ones([1, 1, 1, 1], dtype=torch.float32)
        else:
            print('* Masking Enabled')
        # Six parameters for one filter
        filter_input_range = 5
        assert mask_parameters.shape[1] == self.get_num_mask_parameters()
        mask_parameters = tanh_range(
            l=-filter_input_range, r=filter_input_range,
            initial=0)(mask_parameters)
        size = list(map(int, img.shape[1:3]))
        grid = np.zeros(shape=[1] + size + [2], dtype=np.float32)

        shorter_edge = min(size[0], size[1])
        for i in range(size[0]):
            for j in range(size[1]):
                grid[0, i, j,
                     0] = (i + (shorter_edge - size[0]) / 2.0) / shorter_edge - 0.5
                grid[0, i, j,
                     1] = (j + (shorter_edge - size[1]) / 2.0) / shorter_edge - 0.5
        grid = torch.Tensor(grid)
        # Ax + By + C * L + D
        inp = grid[:, :, :, 0, None] * mask_parameters[:, None, None, 0, None] + \
              grid[:, :, :, 1, None] * mask_parameters[:, None, None, 1, None] + \
              mask_parameters[:, None, None, 2, None] * (rgb2lum(img) - 0.5) + \
              mask_parameters[:, None, None, 3, None] * 2
        # Sharpness and inversion
        inp *= self.cfg['maximum_sharpness'] * mask_parameters[:, None, None, 4,
                                               None] / filter_input_range
        mask = torch.nn.functional.sigmoid(inp)
        # Strength
        mask = mask * (
                mask_parameters[:, None, None, 5, None] / filter_input_range * 0.5 +
                0.5) * (1 - self.cfg['minimum_strength']) + self.cfg['minimum_strength']
        print('mask', mask.shape)
        return mask

    # def visualize_filter(self, debug_info, canvas):
    #   # Visualize only the filter information
    #   assert False

    def visualize_mask(self, debug_info, res):
        return cv2.resize(
            debug_info['mask'] * np.ones((1, 1, 3), dtype=np.float32),
            dsize=res,
            interpolation=cv2.cv2.INTER_NEAREST)

    def draw_high_res_text(self, text, canvas):
        cv2.putText(
            canvas,
            text, (30, 128),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (0, 0, 0),
            thickness=5)
        return canvas


# Defog Filter
class DefogFilter(Filter):  # Defog_param is in [Defog_range]

    def __init__(self, cfg):
        Filter.__init__(self, cfg)
        self.short_name = 'DF'
        self.begin_filter_parameter = cfg['defog_begin_param']
        self.num_filter_parameters = 1

    def filter_param_regressor(self, features):
        return tanh_range(*self.cfg['defog_range'])(features)

    def process(self, img, param, defog_A, IcA):
        # print('      defog_A:', img.shape)
        # print('      defog_A:', IcA.shape)
        # print('      defog_A:', defog_A.shape)

        tx = 1 - param[:, :, None, None] * IcA
        # tx = 1 - 0.5*IcA

        # tx = torch.as_tensor(tx)
        tx_1 = tx.repeat(1, 3, 1, 1)
        return (img - defog_A[:, :, None, None]) / torch.maximum(tx_1, torch.tensor([0.01]).to(tx_1.device)) + defog_A[
                                                                                                               :, :,
                                                                                                               None,
                                                                                                               None]


# WB Filter
class ImprovedWhiteBalanceFilter(Filter):

   def __init__(self, cfg):
        Filter.__init__(self, cfg)
        self.short_name = 'W'
        self.channels = 3
        self.begin_filter_parameter = cfg['wb_begin_param']
        self.num_filter_parameters = self.channels

   def filter_param_regressor(self, features):
        log_wb_range = 0.5
        mask = torch.tensor(((0, 1, 1))).reshape(1, 3).to(features)
        # mask = np.array(((1, 0, 1)), dtype=np.float32).reshape(1, 3)
        # print(mask.shape)
        assert mask.shape == (1, 3)
        features = features * mask
        color_scaling = torch.exp(tanh_range(-log_wb_range, log_wb_range)(features))
        # There will be no division by zero here unless the WB range lower bound is 0
        # normalize by luminance
        color_scaling = color_scaling * (1.0 / (
            1e-5 + 0.27 * color_scaling[:, 0] + 0.67 * color_scaling[:, 1] +
            0.06 * color_scaling[:, 2])[:, None])
        return color_scaling

   def process(self, img, param, defog, IcA):
        return img * param[:, :, None, None]
        # return img


# Gamma Filter
class GammaFilter(Filter):  #gamma_param is in [-gamma_range, gamma_range]

  def __init__(self, cfg):
    Filter.__init__(self, cfg)
    self.short_name = 'G'
    self.begin_filter_parameter = cfg['gamma_begin_param']
    self.num_filter_parameters = 1

  def filter_param_regressor(self, features):
    log_gamma_range = np.log(self.cfg['gamma_range'])
    return torch.exp(tanh_range(-log_gamma_range, log_gamma_range)(features))

  def process(self, img, param, defog_A, IcA):
    param_1 = param.repeat(1, 3)
    return torch.pow(torch.maximum(img, torch.tensor([0.0001]).to(img)), param_1[:, :, None, None])


# Contrast Filter
class ContrastFilter(Filter):

    def __init__(self, cfg):
        Filter.__init__(self, cfg)
        self.short_name = 'Ct'
        self.begin_filter_parameter = cfg['contrast_begin_param']

        self.num_filter_parameters = 1

    def filter_param_regressor(self, features):
        # return tf.sigmoid(features)
        return torch.tanh(features)

    def process(self, img, param, defog, IcA):
        luminance = torch.minimum(torch.maximum(rgb2lum(img), torch.tensor([0.0]).to(img)), torch.tensor([1.0]).to(img))
        contrast_lum = -torch.cos(math.pi * luminance) * 0.5 + 0.5
        contrast_image = img / (luminance + 1e-6) * contrast_lum
        return lerp(img, contrast_image, param[:, :, None, None])
        # return lerp(img, contrast_image, 0.5)

    # def visualize_filter(self, debug_info, canvas):
    #   exposure = debug_info['filter_parameters'][0]
    #   cv2.rectangle(canvas, (8, 40), (56, 52), (1, 1, 1), cv2.FILLED)
    #   cv2.putText(canvas, 'Ct %+.2f' % exposure, (8, 48),
    #               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0))


# Tone Filter
class ToneFilter(Filter):

  def __init__(self, cfg):
    Filter.__init__(self, cfg)
    self.curve_steps = cfg['curve_steps']
    self.short_name = 'T'
    self.begin_filter_parameter = cfg['tone_begin_param']

    self.num_filter_parameters = cfg['curve_steps']

  def filter_param_regressor(self, features):
    tone_curve = torch.reshape(
        features, (-1, 1, self.cfg['curve_steps']))[:, None, None, :]
    tone_curve = tanh_range(*self.cfg['tone_curve_range'])(tone_curve)
    return tone_curve

  def process(self, img, param, defog, IcA):
    # img = tf.minimum(img, 1.0)
    # param = tf.constant([[0.52, 0.53, 0.55, 1.9, 1.8, 1.7, 0.7, 0.6], [0.52, 0.53, 0.55, 1.9, 1.8, 1.7, 0.7, 0.6],
    #                       [0.52, 0.53, 0.55, 1.9, 1.8, 1.7, 0.7, 0.6], [0.52, 0.53, 0.55, 1.9, 1.8, 1.7, 0.7, 0.6],
    #                       [0.52, 0.53, 0.55, 1.9, 1.8, 1.7, 0.7, 0.6], [0.52, 0.53, 0.55, 1.9, 1.8, 1.7, 0.7, 0.6]])
    # param = tf.constant([[0.52, 0.53, 0.55, 1.9, 1.8, 1.7, 0.7, 0.6]])
    # param = tf.reshape(
    #     param, shape=(-1, 1, self.cfg.curve_steps))[:, None, None, :]

    tone_curve = param
    tone_curve_sum = torch.sum(tone_curve, dim=4) + 1e-30
    total_image = img * 0
    for i in range(self.cfg['curve_steps']):
      total_image += torch.clip(img - 1.0 * i / self.cfg['curve_steps'], 0, 1.0 / self.cfg['curve_steps']) \
                     * param[:, :, :, :, i]
    # p_cons = [0.52, 0.53, 0.55, 1.9, 1.8, 1.7, 0.7, 0.6]
    # for i in range(self.cfg.curve_steps):
    #   total_image += tf.clip_by_value(img - 1.0 * i / self.cfg.curve_steps, 0, 1.0 / self.cfg.curve_steps) \
    #                  * p_cons[i]
    total_image *= self.cfg['curve_steps'] / tone_curve_sum
    img = total_image
    return img


# Sharpen Filter
class UsmFilter(Filter):  # Usm_param is in [Defog_range]

  def __init__(self, cfg):
    Filter.__init__(self, cfg)
    self.short_name = 'UF'
    self.begin_filter_parameter = cfg['usm_begin_param']
    self.num_filter_parameters = 1

  def filter_param_regressor(self, features):
    return tanh_range(*self.cfg['usm_range'])(features)

  def process(self, img, param, defog_A, IcA):
    def make_gaussian_2d_kernel(sigma, dtype=torch.float32):
      radius = 12
      x = torch.arange(-radius, radius+1).to(dtype)
      k = torch.exp(-0.5 * torch.square(x / sigma))
      k = k / torch.sum(k)
      return torch.unsqueeze(k, axis=1) * k

    kernel_i = make_gaussian_2d_kernel(5)
    # print('kernel_i.shape', kernel_i.shape)
    kernel_i = kernel_i[None, None, :, :].repeat(1, 1, 1, 1).to(img)

    # outputs = []
    # for channel_idx in range(3):
    #     data_c = img[:, :, :, channel_idx:(channel_idx + 1)]
    #     data_c = tf.nn.conv2d(data_c, kernel_i, [1, 1, 1, 1], 'SAME')
    #     outputs.append(data_c)

    pad_w = (25 - 1) // 2
    # padded = tf.pad(img, [[0, 0], [pad_w, pad_w], [pad_w, pad_w], [0, 0]], mode='REFLECT')
    padded = F.pad(img, [pad_w, pad_w, pad_w, pad_w])
    outputs = []
    kernel_shape = kernel_i.shape
    for channel_idx in range(3):
        data_c = padded[:, channel_idx:(channel_idx + 1), :, :]
        normal = torch.nn.Conv2d(kernel_shape[0], kernel_shape[1], (kernel_shape[2], kernel_shape[3]), 1, 0, bias=False)
        normal.weight.data = kernel_i
        normal.weight.requires_grad = False
        data_c = normal(data_c)
        outputs.append(data_c)

    output = torch.cat(outputs, dim=1)
    img_out = (img - output) * param[:, :, None, None] + img
    # img_out = (img - output) * 2.5 + img

    return img_out