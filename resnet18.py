import torch
import torch.utils.checkpoint
import random

class ResNet18(torch.nn.Module):
    def __init__(self):
        self.weights = [torch.randn((64, 3, 7, 7)) * 1e-7, torch.randn((64,)) * 1e-7, torch.randn((64,)) * 1e-7, torch.randn((64,)) * 1e-7, torch.randn((64,)) * 1e-7, random.randint(32, 64), torch.randn((64, 64, 3, 3)) * 1e-7, torch.randn((64,)) * 1e-7, torch.randn((64,)) * 1e-7, torch.randn((64,)) * 1e-7, torch.randn((64,)) * 1e-7, random.randint(32, 64), torch.randn((64, 64, 3, 3)) * 1e-7, torch.randn((64,)) * 1e-7, torch.randn((64,)) * 1e-7, torch.randn((64,)) * 1e-7, torch.randn((64,)) * 1e-7, random.randint(32, 64), torch.randn((64, 64, 3, 3)) * 1e-7, torch.randn((64,)) * 1e-7, torch.randn((64,)) * 1e-7, torch.randn((64,)) * 1e-7, torch.randn((64,)) * 1e-7, random.randint(32, 64), torch.randn((64, 64, 3, 3)) * 1e-7, torch.randn((64,)) * 1e-7, torch.randn((64,)) * 1e-7, torch.randn((64,)) * 1e-7, torch.randn((64,)) * 1e-7, random.randint(32, 64), torch.randn((128, 64, 3, 3)) * 1e-7, torch.randn((128,)) * 1e-7, torch.randn((128,)) * 1e-7, torch.randn((128,)) * 1e-7, torch.randn((128,)) * 1e-7, random.randint(32, 64), torch.randn((128, 128, 3, 3)) * 1e-7, torch.randn((128,)) * 1e-7, torch.randn((128,)) * 1e-7, torch.randn((128,)) * 1e-7, torch.randn((128,)) * 1e-7, random.randint(32, 64), torch.randn((128, 64, 1, 1)) * 1e-7, torch.randn((128,)) * 1e-7, torch.randn((128,)) * 1e-7, torch.randn((128,)) * 1e-7, torch.randn((128,)) * 1e-7, random.randint(32, 64), torch.randn((128, 128, 3, 3)) * 1e-7, torch.randn((128,)) * 1e-7, torch.randn((128,)) * 1e-7, torch.randn((128,)) * 1e-7, torch.randn((128,)) * 1e-7, random.randint(32, 64), torch.randn((128, 128, 3, 3)) * 1e-7, torch.randn((128,)) * 1e-7, torch.randn((128,)) * 1e-7, torch.randn((128,)) * 1e-7, torch.randn((128,)) * 1e-7, random.randint(32, 64), torch.randn((256, 128, 3, 3)) * 1e-7, torch.randn((256,)) * 1e-7, torch.randn((256,)) * 1e-7, torch.randn((256,)) * 1e-7, torch.randn((256,)) * 1e-7, random.randint(32, 64), torch.randn((256, 256, 3, 3)) * 1e-7, torch.randn((256,)) * 1e-7, torch.randn((256,)) * 1e-7, torch.randn((256,)) * 1e-7, torch.randn((256,)) * 1e-7, random.randint(32, 64), torch.randn((256, 128, 1, 1)) * 1e-7, torch.randn((256,)) * 1e-7, torch.randn((256,)) * 1e-7, torch.randn((256,)) * 1e-7, torch.randn((256,)) * 1e-7, random.randint(32, 64), torch.randn((256, 256, 3, 3)) * 1e-7, torch.randn((256,)) * 1e-7, torch.randn((256,)) * 1e-7, torch.randn((256,)) * 1e-7, torch.randn((256,)) * 1e-7, random.randint(32, 64), torch.randn((256, 256, 3, 3)) * 1e-7, torch.randn((256,)) * 1e-7, torch.randn((256,)) * 1e-7, torch.randn((256,)) * 1e-7, torch.randn((256,)) * 1e-7, random.randint(32, 64), torch.randn((512, 256, 3, 3)) * 1e-7, torch.randn((512,)) * 1e-7, torch.randn((512,)) * 1e-7, torch.randn((512,)) * 1e-7, torch.randn((512,)) * 1e-7, random.randint(32, 64), torch.randn((512, 512, 3, 3)) * 1e-7, torch.randn((512,)) * 1e-7, torch.randn((512,)) * 1e-7, torch.randn((512,)) * 1e-7, torch.randn((512,)) * 1e-7, random.randint(32, 64), torch.randn((512, 256, 1, 1)) * 1e-7, torch.randn((512,)) * 1e-7, torch.randn((512,)) * 1e-7, torch.randn((512,)) * 1e-7, torch.randn((512,)) * 1e-7, random.randint(32, 64), torch.randn((512, 512, 3, 3)) * 1e-7, torch.randn((512,)) * 1e-7, torch.randn((512,)) * 1e-7, torch.randn((512,)) * 1e-7, torch.randn((512,)) * 1e-7, random.randint(32, 64), torch.randn((512, 512, 3, 3)) * 1e-7, torch.randn((512,)) * 1e-7, torch.randn((512,)) * 1e-7, torch.randn((512,)) * 1e-7, torch.randn((512,)) * 1e-7, random.randint(32, 64), torch.randn((1000, 512)) * 1e-7, torch.randn((1000,)) * 1e-7]

    def jojo_0(self, input_vars_121):
        var_915 = torch.t(input_vars_121)
        return var_915

    def jojo_1(self, input_vars_94, var_786, input_vars_95, input_vars_93, input_vars_92):
        var_791 = torch.batch_norm(var_786, input_vars_92, input_vars_93, input_vars_94, input_vars_95, False, 0.1, 1e-05, True)
        return var_791

    def jojo_2(self, input_vars_89, var_715, input_vars_87, input_vars_85, input_vars_88, var_740, input_vars_86):
        var_759 = torch._convolution(var_740, input_vars_85, None, [1, 1, ], [1, 1, ], [1, 1, ], False, [0, 0, ], 1, False, False, True)
        var_764 = torch.batch_norm(var_759, input_vars_86, input_vars_87, input_vars_88, input_vars_89, False, 0.1, 1e-05, True)
        var_766 = torch.add(var_764, var_715, alpha=1)
        return var_766

    def jojo_3(self, var_714):
        var_715 = torch.relu_(var_714)
        return var_715

    def jojo_4(self, input_vars_69, input_vars_71, input_vars_77, input_vars_68, input_vars_74, input_vars_73, input_vars_70, var_639, input_vars_75, input_vars_76, var_683):
        var_688 = torch.batch_norm(var_683, input_vars_68, input_vars_69, input_vars_70, input_vars_71, False, 0.1, 1e-05, True)
        var_707 = torch._convolution(var_639, input_vars_73, None, [2, 2, ], [0, 0, ], [1, 1, ], False, [0, 0, ], 1, False, False, True)
        var_712 = torch.batch_norm(var_707, input_vars_74, input_vars_75, input_vars_76, input_vars_77, False, 0.1, 1e-05, True)
        return var_688, var_712

    def jojo_5(self, input_vars_56, input_vars_64, var_586, input_vars_65, input_vars_59, input_vars_63, input_vars_58, input_vars_61, input_vars_57, input_vars_50, input_vars_53, input_vars_52, input_vars_51, input_vars_49, input_vars_55, input_vars_62):
        var_587 = torch.relu_(var_586)
        var_606 = torch._convolution(var_587, input_vars_49, None, [1, 1, ], [1, 1, ], [1, 1, ], False, [0, 0, ], 1, False, False, True)
        var_611 = torch.batch_norm(var_606, input_vars_50, input_vars_51, input_vars_52, input_vars_53, False, 0.1, 1e-05, True)
        var_612 = torch.relu_(var_611)
        var_631 = torch._convolution(var_612, input_vars_55, None, [1, 1, ], [1, 1, ], [1, 1, ], False, [0, 0, ], 1, False, False, True)
        var_636 = torch.batch_norm(var_631, input_vars_56, input_vars_57, input_vars_58, input_vars_59, False, 0.1, 1e-05, True)
        var_638 = torch.add(var_636, var_587, alpha=1)
        var_639 = torch.relu_(var_638)
        var_658 = torch._convolution(var_639, input_vars_61, None, [2, 2, ], [1, 1, ], [1, 1, ], False, [0, 0, ], 1, False, False, True)
        var_663 = torch.batch_norm(var_658, input_vars_62, input_vars_63, input_vars_64, input_vars_65, False, 0.1, 1e-05, True)
        return var_639, var_663

    def jojo_6(self, var_459, input_vars_35, var_508, input_vars_40, input_vars_41, input_vars_47, input_vars_45, input_vars_38, input_vars_33, input_vars_43, input_vars_44, input_vars_46, input_vars_31, input_vars_34, input_vars_39, input_vars_32, input_vars_37):
        var_510 = torch.add(var_508, var_459, alpha=1)
        var_511 = torch.relu_(var_510)
        var_530 = torch._convolution(var_511, input_vars_31, None, [2, 2, ], [1, 1, ], [1, 1, ], False, [0, 0, ], 1, False, False, True)
        var_535 = torch.batch_norm(var_530, input_vars_32, input_vars_33, input_vars_34, input_vars_35, False, 0.1, 1e-05, True)
        var_536 = torch.relu_(var_535)
        var_555 = torch._convolution(var_536, input_vars_37, None, [1, 1, ], [1, 1, ], [1, 1, ], False, [0, 0, ], 1, False, False, True)
        var_560 = torch.batch_norm(var_555, input_vars_38, input_vars_39, input_vars_40, input_vars_41, False, 0.1, 1e-05, True)
        var_579 = torch._convolution(var_511, input_vars_43, None, [2, 2, ], [0, 0, ], [1, 1, ], False, [0, 0, ], 1, False, False, True)
        var_584 = torch.batch_norm(var_579, input_vars_44, input_vars_45, input_vars_46, input_vars_47, False, 0.1, 1e-05, True)
        return var_584, var_560

    def jojo_7(self, input_vars_17, input_vars_19, var_407, var_431, input_vars_25, input_vars_20, input_vars_14, input_vars_13, input_vars_23, input_vars_15, input_vars_16, input_vars_22, input_vars_21):
        var_432 = torch.relu_(var_431)
        var_451 = torch._convolution(var_432, input_vars_13, None, [1, 1, ], [1, 1, ], [1, 1, ], False, [0, 0, ], 1, False, False, True)
        var_456 = torch.batch_norm(var_451, input_vars_14, input_vars_15, input_vars_16, input_vars_17, False, 0.1, 1e-05, True)
        var_458 = torch.add(var_456, var_407, alpha=1)
        var_459 = torch.relu_(var_458)
        var_478 = torch._convolution(var_459, input_vars_19, None, [1, 1, ], [1, 1, ], [1, 1, ], False, [0, 0, ], 1, False, False, True)
        var_483 = torch.batch_norm(var_478, input_vars_20, input_vars_21, input_vars_22, input_vars_23, False, 0.1, 1e-05, True)
        var_484 = torch.relu_(var_483)
        var_503 = torch._convolution(var_484, input_vars_25, None, [1, 1, ], [1, 1, ], [1, 1, ], False, [0, 0, ], 1, False, False, True)
        return var_459, var_503

    def jojo_8(self, input_vars_2, input_vars_7, input_vars_3, input_vars_5, input_vars_0, input_vars_4, input_vars_1):
        var_387 = torch._convolution(input_vars_0, input_vars_1, None, [2, 2, ], [3, 3, ], [1, 1, ], False, [0, 0, ], 1, False, False, True)
        var_392 = torch.batch_norm(var_387, input_vars_2, input_vars_3, input_vars_4, input_vars_5, False, 0.1, 1e-05, True)
        var_393 = torch.relu_(var_392)
        var_407 = torch.max_pool2d(var_393, [3, 3, ], [2, 2, ], [1, 1, ], [1, 1, ], False)
        var_426 = torch._convolution(var_407, input_vars_7, None, [1, 1, ], [1, 1, ], [1, 1, ], False, [0, 0, ], 1, False, False, True)
        return var_426, var_407

    def forward_(self, input_vars):
        var_426, var_407 = torch.utils.checkpoint.checkpoint(self.jojo_8, input_vars[2], input_vars[7], input_vars[3], input_vars[5], input_vars[0], input_vars[4], input_vars[1])
        var_431 = torch.batch_norm(var_426, input_vars[8], input_vars[9], input_vars[10], input_vars[11], False, 0.1, 1e-05, True)
        var_459, var_503 = torch.utils.checkpoint.checkpoint(self.jojo_7, input_vars[17], input_vars[19], var_407, var_431, input_vars[25], input_vars[20], input_vars[14], input_vars[13], input_vars[23], input_vars[15], input_vars[16], input_vars[22], input_vars[21])
        var_508 = torch.batch_norm(var_503, input_vars[26], input_vars[27], input_vars[28], input_vars[29], False, 0.1, 1e-05, True)
        var_584, var_560 = torch.utils.checkpoint.checkpoint(self.jojo_6, var_459, input_vars[35], var_508, input_vars[40], input_vars[41], input_vars[47], input_vars[45], input_vars[38], input_vars[33], input_vars[43], input_vars[44], input_vars[46], input_vars[31], input_vars[34], input_vars[39], input_vars[32], input_vars[37])
        var_586 = torch.add(var_560, var_584, alpha=1)
        var_639, var_663 = torch.utils.checkpoint.checkpoint(self.jojo_5, input_vars[56], input_vars[64], var_586, input_vars[65], input_vars[59], input_vars[63], input_vars[58], input_vars[61], input_vars[57], input_vars[50], input_vars[53], input_vars[52], input_vars[51], input_vars[49], input_vars[55], input_vars[62])
        var_664 = torch.relu_(var_663)
        var_683 = torch._convolution(var_664, input_vars[67], None, [1, 1, ], [1, 1, ], [1, 1, ], False, [0, 0, ], 1, False, False, True)
        var_688, var_712 = torch.utils.checkpoint.checkpoint(self.jojo_4, input_vars[69], input_vars[71], input_vars[77], input_vars[68], input_vars[74], input_vars[73], input_vars[70], var_639, input_vars[75], input_vars[76], var_683)
        var_714 = torch.add(var_688, var_712, alpha=1)
        var_715 = torch.utils.checkpoint.checkpoint(self.jojo_3, var_714)
        var_734 = torch._convolution(var_715, input_vars[79], None, [1, 1, ], [1, 1, ], [1, 1, ], False, [0, 0, ], 1, False, False, True)
        var_739 = torch.batch_norm(var_734, input_vars[80], input_vars[81], input_vars[82], input_vars[83], False, 0.1, 1e-05, True)
        var_740 = torch.relu_(var_739)
        var_766 = torch.utils.checkpoint.checkpoint(self.jojo_2, input_vars[89], var_715, input_vars[87], input_vars[85], input_vars[88], var_740, input_vars[86])
        var_767 = torch.relu_(var_766)
        var_786 = torch._convolution(var_767, input_vars[91], None, [2, 2, ], [1, 1, ], [1, 1, ], False, [0, 0, ], 1, False, False, True)
        var_791 = torch.utils.checkpoint.checkpoint(self.jojo_1, input_vars[94], var_786, input_vars[95], input_vars[93], input_vars[92])
        var_792 = torch.relu_(var_791)
        var_811 = torch._convolution(var_792, input_vars[97], None, [1, 1, ], [1, 1, ], [1, 1, ], False, [0, 0, ], 1, False, False, True)
        var_816 = torch.batch_norm(var_811, input_vars[98], input_vars[99], input_vars[100], input_vars[101], False, 0.1, 1e-05, True)
        var_835 = torch._convolution(var_767, input_vars[103], None, [2, 2, ], [0, 0, ], [1, 1, ], False, [0, 0, ], 1, False, False, True)
        var_840 = torch.batch_norm(var_835, input_vars[104], input_vars[105], input_vars[106], input_vars[107], False, 0.1, 1e-05, True)
        var_842 = torch.add(var_816, var_840, alpha=1)
        var_843 = torch.relu_(var_842)
        var_862 = torch._convolution(var_843, input_vars[109], None, [1, 1, ], [1, 1, ], [1, 1, ], False, [0, 0, ], 1, False, False, True)
        var_867 = torch.batch_norm(var_862, input_vars[110], input_vars[111], input_vars[112], input_vars[113], False, 0.1, 1e-05, True)
        var_868 = torch.relu_(var_867)
        var_887 = torch._convolution(var_868, input_vars[115], None, [1, 1, ], [1, 1, ], [1, 1, ], False, [0, 0, ], 1, False, False, True)
        var_892 = torch.batch_norm(var_887, input_vars[116], input_vars[117], input_vars[118], input_vars[119], False, 0.1, 1e-05, True)
        var_894 = torch.add(var_892, var_843, alpha=1)
        var_895 = torch.relu_(var_894)
        var_911 = torch.nn.AdaptiveAvgPool2d([1, 1, ])(var_895)
        var_914 = torch.flatten(var_911, 1, -1)
        var_915 = torch.utils.checkpoint.checkpoint(self.jojo_0, input_vars[121])
        var_918 = torch.addmm(input_vars[122], var_914, var_915, beta=1, alpha=1)
        return var_918
    
    def forward(self, inputs):
        return self.forward_([ inputs ] + self.weights)
