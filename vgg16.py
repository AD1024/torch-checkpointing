import torch
import torch.utils.checkpoint
import random

class Vgg16(torch.nn.Module):
    def __init__(self):
        self.weights = [torch.randn((64, 3, 3, 3)) * 1e-7, torch.randn((64,)) * 1e-7, torch.randn((64, 64, 3, 3)) * 1e-7, torch.randn((64,)) * 1e-7, torch.randn((128, 64, 3, 3)) * 1e-7, torch.randn((128,)) * 1e-7, torch.randn((128, 128, 3, 3)) * 1e-7, torch.randn((128,)) * 1e-7, torch.randn((256, 128, 3, 3)) * 1e-7, torch.randn((256,)) * 1e-7, torch.randn((256, 256, 3, 3)) * 1e-7, torch.randn((256,)) * 1e-7, torch.randn((256, 256, 3, 3)) * 1e-7, torch.randn((256,)) * 1e-7, torch.randn((512, 256, 3, 3)) * 1e-7, torch.randn((512,)) * 1e-7, torch.randn((512, 512, 3, 3)) * 1e-7, torch.randn((512,)) * 1e-7, torch.randn((512, 512, 3, 3)) * 1e-7, torch.randn((512,)) * 1e-7, torch.randn((512, 512, 3, 3)) * 1e-7, torch.randn((512,)) * 1e-7, torch.randn((512, 512, 3, 3)) * 1e-7, torch.randn((512,)) * 1e-7, torch.randn((512, 512, 3, 3)) * 1e-7, torch.randn((512,)) * 1e-7, torch.randn((4096, 25088)) * 1e-7, torch.randn((4096,)) * 1e-7, torch.randn((4096, 4096)) * 1e-7, torch.randn((4096,)) * 1e-7, torch.randn((1000, 4096)) * 1e-7, torch.randn((1000,)) * 1e-7]

    def jojo_0(self, var_11, var_13, var_220, var_12, var_14):
        var_221 = torch.relu_(var_220)
        var_239 = torch._convolution(var_221, var_11, var_12, [1, 1], [1, 1], [1, 1], False, [0, 0], 1, False, False, True)
        var_240 = torch.relu_(var_239)
        var_258 = torch._convolution(var_240, var_13, var_14, [1, 1], [1, 1], [1, 1], False, [0, 0], 1, False, False, True)
        return var_258

    def jojo_1(self, var_7, var_169, var_8):
        var_187 = torch._convolution(var_169, var_7, var_8, [1, 1], [1, 1], [1, 1], False, [0, 0], 1, False, False, True)
        var_188 = torch.relu_(var_187)
        var_202 = torch.max_pool2d(var_188, [2, 2], [2, 2], [0, 0], [1, 1], False)
        return var_202

    def jojo_2(self, var_5, var_136, var_6):
        var_150 = torch.max_pool2d(var_136, [2, 2], [2, 2], [0, 0], [1, 1], False)
        var_168 = torch._convolution(var_150, var_5, var_6, [1, 1], [1, 1], [1, 1], False, [0, 0], 1, False, False, True)
        return var_168

    def jojo_3(self, var_3, var_4, var_117):
        var_135 = torch._convolution(var_117, var_3, var_4, [1, 1], [1, 1], [1, 1], False, [0, 0], 1, False, False, True)
        return var_135

    def jojo_4(self, var_2, var_0, var_1):
        var_116 = torch._convolution(var_0, var_1, var_2, [1, 1], [1, 1], [1, 1], False, [0, 0], 1, False, False, True)
        return var_116

    def forward_(self, var_0, var_1, var_2, var_3, var_4, var_5, var_6, var_7, var_8, var_9, var_10, var_11, var_12, var_13, var_14, var_15, var_16, var_17, var_18, var_19, var_20, var_21, var_22, var_23, var_24, var_25, var_26, var_27, var_28, var_29, var_30, var_31, var_32):
        var_116 = torch.utils.checkpoint.checkpoint(self.jojo_4, var_2, var_0, var_1)
        var_117 = torch.relu_(var_116)
        var_135 = torch.utils.checkpoint.checkpoint(self.jojo_3, var_3, var_4, var_117)
        var_136 = torch.relu_(var_135)
        var_168 = torch.utils.checkpoint.checkpoint(self.jojo_2, var_5, var_136, var_6)
        var_169 = torch.relu_(var_168)
        var_202 = torch.utils.checkpoint.checkpoint(self.jojo_1, var_7, var_169, var_8)
        var_220 = torch._convolution(var_202, var_9, var_10, [1, 1], [1, 1], [1, 1], False, [0, 0], 1, False, False, True)
        var_258 = torch.utils.checkpoint.checkpoint(self.jojo_0, var_11, var_13, var_220, var_12, var_14)
        var_259 = torch.relu_(var_258)
        var_273 = torch.max_pool2d(var_259, [2, 2], [2, 2], [0, 0], [1, 1], False)
        var_291 = torch._convolution(var_273, var_15, var_16, [1, 1], [1, 1], [1, 1], False, [0, 0], 1, False, False, True)
        var_292 = torch.relu_(var_291)
        var_310 = torch._convolution(var_292, var_17, var_18, [1, 1], [1, 1], [1, 1], False, [0, 0], 1, False, False, True)
        var_311 = torch.relu_(var_310)
        var_329 = torch._convolution(var_311, var_19, var_20, [1, 1], [1, 1], [1, 1], False, [0, 0], 1, False, False, True)
        var_330 = torch.relu_(var_329)
        var_344 = torch.max_pool2d(var_330, [2, 2], [2, 2], [0, 0], [1, 1], False)
        var_362 = torch._convolution(var_344, var_21, var_22, [1, 1], [1, 1], [1, 1], False, [0, 0], 1, False, False, True)
        var_363 = torch.relu_(var_362)
        var_381 = torch._convolution(var_363, var_23, var_24, [1, 1], [1, 1], [1, 1], False, [0, 0], 1, False, False, True)
        var_382 = torch.relu_(var_381)
        var_400 = torch._convolution(var_382, var_25, var_26, [1, 1], [1, 1], [1, 1], False, [0, 0], 1, False, False, True)
        var_401 = torch.relu_(var_400)
        var_415 = torch.max_pool2d(var_401, [2, 2], [2, 2], [0, 0], [1, 1], False)
        var_431 = torch.nn.AdaptiveAvgPool2d([7, 7])(var_415)
        var_434 = torch.flatten(var_431, 1, -1)
        var_435 = torch.t(var_27)
        var_438 = torch.addmm(var_28, var_434, var_435, beta=1, alpha=1)
        var_439 = torch.relu_(var_438)
        var_442 = torch.dropout(var_439, 0.5, False)
        var_443 = torch.t(var_29)
        var_446 = torch.addmm(var_30, var_442, var_443, beta=1, alpha=1)
        var_447 = torch.relu_(var_446)
        var_450 = torch.dropout(var_447, 0.5, False)
        var_451 = torch.t(var_31)
        var_454 = torch.addmm(var_32, var_450, var_451, beta=1, alpha=1)
        return var_454
    
    def forward(self, inputs):
        return self.forward_(inputs, *self.weights)
