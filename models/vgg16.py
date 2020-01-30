import torch
import torch.utils.checkpoint
import random

class Vgg16(torch.nn.Module):
    def __init__(self):
        self.weights = [torch.randn((64, 3, 3, 3)) * 1e-7, torch.randn((64,)) * 1e-7, torch.randn((64, 64, 3, 3)) * 1e-7, torch.randn((64,)) * 1e-7, torch.randn((128, 64, 3, 3)) * 1e-7, torch.randn((128,)) * 1e-7, torch.randn((128, 128, 3, 3)) * 1e-7, torch.randn((128,)) * 1e-7, torch.randn((256, 128, 3, 3)) * 1e-7, torch.randn((256,)) * 1e-7, torch.randn((256, 256, 3, 3)) * 1e-7, torch.randn((256,)) * 1e-7, torch.randn((256, 256, 3, 3)) * 1e-7, torch.randn((256,)) * 1e-7, torch.randn((512, 256, 3, 3)) * 1e-7, torch.randn((512,)) * 1e-7, torch.randn((512, 512, 3, 3)) * 1e-7, torch.randn((512,)) * 1e-7, torch.randn((512, 512, 3, 3)) * 1e-7, torch.randn((512,)) * 1e-7, torch.randn((512, 512, 3, 3)) * 1e-7, torch.randn((512,)) * 1e-7, torch.randn((512, 512, 3, 3)) * 1e-7, torch.randn((512,)) * 1e-7, torch.randn((512, 512, 3, 3)) * 1e-7, torch.randn((512,)) * 1e-7, torch.randn((4096, 25088)) * 1e-7, torch.randn((4096,)) * 1e-7, torch.randn((4096, 4096)) * 1e-7, torch.randn((4096,)) * 1e-7, torch.randn((1000, 4096)) * 1e-7, torch.randn((1000,)) * 1e-7]

    def jojo_0(self, input_vars_12, var_220, input_vars_14, input_vars_11, input_vars_13):
        var_221 = torch.relu_(var_220)
        var_239 = torch._convolution(var_221, input_vars_11, input_vars_12, [1, 1, ], [1, 1, ], [1, 1, ], False, [0, 0, ], 1, False, False, True)
        var_240 = torch.relu_(var_239)
        var_258 = torch._convolution(var_240, input_vars_13, input_vars_14, [1, 1, ], [1, 1, ], [1, 1, ], False, [0, 0, ], 1, False, False, True)
        return var_258

    def jojo_1(self, var_169, input_vars_8, input_vars_7):
        var_187 = torch._convolution(var_169, input_vars_7, input_vars_8, [1, 1, ], [1, 1, ], [1, 1, ], False, [0, 0, ], 1, False, False, True)
        var_188 = torch.relu_(var_187)
        var_202 = torch.max_pool2d(var_188, [2, 2, ], [2, 2, ], [0, 0, ], [1, 1, ], False)
        return var_202

    def jojo_2(self, input_vars_6, input_vars_5, var_136):
        var_150 = torch.max_pool2d(var_136, [2, 2, ], [2, 2, ], [0, 0, ], [1, 1, ], False)
        var_168 = torch._convolution(var_150, input_vars_5, input_vars_6, [1, 1, ], [1, 1, ], [1, 1, ], False, [0, 0, ], 1, False, False, True)
        return var_168

    def jojo_3(self, input_vars_3, var_117, input_vars_4):
        var_135 = torch._convolution(var_117, input_vars_3, input_vars_4, [1, 1, ], [1, 1, ], [1, 1, ], False, [0, 0, ], 1, False, False, True)
        return var_135

    def jojo_4(self, input_vars_1, input_vars_0, input_vars_2):
        var_116 = torch._convolution(input_vars_0, input_vars_1, input_vars_2, [1, 1, ], [1, 1, ], [1, 1, ], False, [0, 0, ], 1, False, False, True)
        return var_116

    def forward_(self, input_vars):
        var_116 = torch.utils.checkpoint.checkpoint(self.jojo_4, input_vars[1], input_vars[0], input_vars[2])
        var_117 = torch.relu_(var_116)
        var_135 = torch.utils.checkpoint.checkpoint(self.jojo_3, input_vars[3], var_117, input_vars[4])
        var_136 = torch.relu_(var_135)
        var_168 = torch.utils.checkpoint.checkpoint(self.jojo_2, input_vars[6], input_vars[5], var_136)
        var_169 = torch.relu_(var_168)
        var_202 = torch.utils.checkpoint.checkpoint(self.jojo_1, var_169, input_vars[8], input_vars[7])
        var_220 = torch._convolution(var_202, input_vars[9], input_vars[10], [1, 1, ], [1, 1, ], [1, 1, ], False, [0, 0, ], 1, False, False, True)
        var_258 = torch.utils.checkpoint.checkpoint(self.jojo_0, input_vars[12], var_220, input_vars[14], input_vars[11], input_vars[13])
        var_259 = torch.relu_(var_258)
        var_273 = torch.max_pool2d(var_259, [2, 2, ], [2, 2, ], [0, 0, ], [1, 1, ], False)
        var_291 = torch._convolution(var_273, input_vars[15], input_vars[16], [1, 1, ], [1, 1, ], [1, 1, ], False, [0, 0, ], 1, False, False, True)
        var_292 = torch.relu_(var_291)
        var_310 = torch._convolution(var_292, input_vars[17], input_vars[18], [1, 1, ], [1, 1, ], [1, 1, ], False, [0, 0, ], 1, False, False, True)
        var_311 = torch.relu_(var_310)
        var_329 = torch._convolution(var_311, input_vars[19], input_vars[20], [1, 1, ], [1, 1, ], [1, 1, ], False, [0, 0, ], 1, False, False, True)
        var_330 = torch.relu_(var_329)
        var_344 = torch.max_pool2d(var_330, [2, 2, ], [2, 2, ], [0, 0, ], [1, 1, ], False)
        var_362 = torch._convolution(var_344, input_vars[21], input_vars[22], [1, 1, ], [1, 1, ], [1, 1, ], False, [0, 0, ], 1, False, False, True)
        var_363 = torch.relu_(var_362)
        var_381 = torch._convolution(var_363, input_vars[23], input_vars[24], [1, 1, ], [1, 1, ], [1, 1, ], False, [0, 0, ], 1, False, False, True)
        var_382 = torch.relu_(var_381)
        var_400 = torch._convolution(var_382, input_vars[25], input_vars[26], [1, 1, ], [1, 1, ], [1, 1, ], False, [0, 0, ], 1, False, False, True)
        var_401 = torch.relu_(var_400)
        var_415 = torch.max_pool2d(var_401, [2, 2, ], [2, 2, ], [0, 0, ], [1, 1, ], False)
        var_431 = torch.nn.AdaptiveAvgPool2d([7, 7, ])(var_415)
        var_434 = torch.flatten(var_431, 1, -1)
        var_435 = torch.t(input_vars[27])
        var_438 = torch.addmm(input_vars[28], var_434, var_435, beta=1, alpha=1)
        var_439 = torch.relu_(var_438)
        var_442 = torch.dropout(var_439, 0.5, False)
        var_443 = torch.t(input_vars[29])
        var_446 = torch.addmm(input_vars[30], var_442, var_443, beta=1, alpha=1)
        var_447 = torch.relu_(var_446)
        var_450 = torch.dropout(var_447, 0.5, False)
        var_451 = torch.t(input_vars[31])
        var_454 = torch.addmm(input_vars[32], var_450, var_451, beta=1, alpha=1)
        return var_454
    
    def forward(self, inputs):
        return self.forward_([ inputs ] + self.weights)
