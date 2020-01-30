import torch
import torch.utils.checkpoint
import random

class ResNet34(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = [torch.randn((64, 3, 7, 7)) * 1e-7, torch.randn((64,)) * 1e-7, torch.randn((64,)) * 1e-7, torch.randn((64,)) * 1e-7, torch.randn((64,)) * 1e-7, random.randint(32, 64), torch.randn((64, 64, 3, 3)) * 1e-7, torch.randn((64,)) * 1e-7, torch.randn((64,)) * 1e-7, torch.randn((64,)) * 1e-7, torch.randn((64,)) * 1e-7, random.randint(32, 64), torch.randn((64, 64, 3, 3)) * 1e-7, torch.randn((64,)) * 1e-7, torch.randn((64,)) * 1e-7, torch.randn((64,)) * 1e-7, torch.randn((64,)) * 1e-7, random.randint(32, 64), torch.randn((64, 64, 3, 3)) * 1e-7, torch.randn((64,)) * 1e-7, torch.randn((64,)) * 1e-7, torch.randn((64,)) * 1e-7, torch.randn((64,)) * 1e-7, random.randint(32, 64), torch.randn((64, 64, 3, 3)) * 1e-7, torch.randn((64,)) * 1e-7, torch.randn((64,)) * 1e-7, torch.randn((64,)) * 1e-7, torch.randn((64,)) * 1e-7, random.randint(32, 64), torch.randn((64, 64, 3, 3)) * 1e-7, torch.randn((64,)) * 1e-7, torch.randn((64,)) * 1e-7, torch.randn((64,)) * 1e-7, torch.randn((64,)) * 1e-7, random.randint(32, 64), torch.randn((64, 64, 3, 3)) * 1e-7, torch.randn((64,)) * 1e-7, torch.randn((64,)) * 1e-7, torch.randn((64,)) * 1e-7, torch.randn((64,)) * 1e-7, random.randint(32, 64), torch.randn((128, 64, 3, 3)) * 1e-7, torch.randn((128,)) * 1e-7, torch.randn((128,)) * 1e-7, torch.randn((128,)) * 1e-7, torch.randn((128,)) * 1e-7, random.randint(32, 64), torch.randn((128, 128, 3, 3)) * 1e-7, torch.randn((128,)) * 1e-7, torch.randn((128,)) * 1e-7, torch.randn((128,)) * 1e-7, torch.randn((128,)) * 1e-7, random.randint(32, 64), torch.randn((128, 64, 1, 1)) * 1e-7, torch.randn((128,)) * 1e-7, torch.randn((128,)) * 1e-7, torch.randn((128,)) * 1e-7, torch.randn((128,)) * 1e-7, random.randint(32, 64), torch.randn((128, 128, 3, 3)) * 1e-7, torch.randn((128,)) * 1e-7, torch.randn((128,)) * 1e-7, torch.randn((128,)) * 1e-7, torch.randn((128,)) * 1e-7, random.randint(32, 64), torch.randn((128, 128, 3, 3)) * 1e-7, torch.randn((128,)) * 1e-7, torch.randn((128,)) * 1e-7, torch.randn((128,)) * 1e-7, torch.randn((128,)) * 1e-7, random.randint(32, 64), torch.randn((128, 128, 3, 3)) * 1e-7, torch.randn((128,)) * 1e-7, torch.randn((128,)) * 1e-7, torch.randn((128,)) * 1e-7, torch.randn((128,)) * 1e-7, random.randint(32, 64), torch.randn((128, 128, 3, 3)) * 1e-7, torch.randn((128,)) * 1e-7, torch.randn((128,)) * 1e-7, torch.randn((128,)) * 1e-7, torch.randn((128,)) * 1e-7, random.randint(32, 64), torch.randn((128, 128, 3, 3)) * 1e-7, torch.randn((128,)) * 1e-7, torch.randn((128,)) * 1e-7, torch.randn((128,)) * 1e-7, torch.randn((128,)) * 1e-7, random.randint(32, 64), torch.randn((128, 128, 3, 3)) * 1e-7, torch.randn((128,)) * 1e-7, torch.randn((128,)) * 1e-7, torch.randn((128,)) * 1e-7, torch.randn((128,)) * 1e-7, random.randint(32, 64), torch.randn((256, 128, 3, 3)) * 1e-7, torch.randn((256,)) * 1e-7, torch.randn((256,)) * 1e-7, torch.randn((256,)) * 1e-7, torch.randn((256,)) * 1e-7, random.randint(32, 64), torch.randn((256, 256, 3, 3)) * 1e-7, torch.randn((256,)) * 1e-7, torch.randn((256,)) * 1e-7, torch.randn((256,)) * 1e-7, torch.randn((256,)) * 1e-7, random.randint(32, 64), torch.randn((256, 128, 1, 1)) * 1e-7, torch.randn((256,)) * 1e-7, torch.randn((256,)) * 1e-7, torch.randn((256,)) * 1e-7, torch.randn((256,)) * 1e-7, random.randint(32, 64), torch.randn((256, 256, 3, 3)) * 1e-7, torch.randn((256,)) * 1e-7, torch.randn((256,)) * 1e-7, torch.randn((256,)) * 1e-7, torch.randn((256,)) * 1e-7, random.randint(32, 64), torch.randn((256, 256, 3, 3)) * 1e-7, torch.randn((256,)) * 1e-7, torch.randn((256,)) * 1e-7, torch.randn((256,)) * 1e-7, torch.randn((256,)) * 1e-7, random.randint(32, 64), torch.randn((256, 256, 3, 3)) * 1e-7, torch.randn((256,)) * 1e-7, torch.randn((256,)) * 1e-7, torch.randn((256,)) * 1e-7, torch.randn((256,)) * 1e-7, random.randint(32, 64), torch.randn((256, 256, 3, 3)) * 1e-7, torch.randn((256,)) * 1e-7, torch.randn((256,)) * 1e-7, torch.randn((256,)) * 1e-7, torch.randn((256,)) * 1e-7, random.randint(32, 64), torch.randn((256, 256, 3, 3)) * 1e-7, torch.randn((256,)) * 1e-7, torch.randn((256,)) * 1e-7, torch.randn((256,)) * 1e-7, torch.randn((256,)) * 1e-7, random.randint(32, 64), torch.randn((256, 256, 3, 3)) * 1e-7, torch.randn((256,)) * 1e-7, torch.randn((256,)) * 1e-7, torch.randn((256,)) * 1e-7, torch.randn((256,)) * 1e-7, random.randint(32, 64), torch.randn((256, 256, 3, 3)) * 1e-7, torch.randn((256,)) * 1e-7, torch.randn((256,)) * 1e-7, torch.randn((256,)) * 1e-7, torch.randn((256,)) * 1e-7, random.randint(32, 64), torch.randn((256, 256, 3, 3)) * 1e-7, torch.randn((256,)) * 1e-7, torch.randn((256,)) * 1e-7, torch.randn((256,)) * 1e-7, torch.randn((256,)) * 1e-7, random.randint(32, 64), torch.randn((256, 256, 3, 3)) * 1e-7, torch.randn((256,)) * 1e-7, torch.randn((256,)) * 1e-7, torch.randn((256,)) * 1e-7, torch.randn((256,)) * 1e-7, random.randint(32, 64), torch.randn((256, 256, 3, 3)) * 1e-7, torch.randn((256,)) * 1e-7, torch.randn((256,)) * 1e-7, torch.randn((256,)) * 1e-7, torch.randn((256,)) * 1e-7, random.randint(32, 64), torch.randn((512, 256, 3, 3)) * 1e-7, torch.randn((512,)) * 1e-7, torch.randn((512,)) * 1e-7, torch.randn((512,)) * 1e-7, torch.randn((512,)) * 1e-7, random.randint(32, 64), torch.randn((512, 512, 3, 3)) * 1e-7, torch.randn((512,)) * 1e-7, torch.randn((512,)) * 1e-7, torch.randn((512,)) * 1e-7, torch.randn((512,)) * 1e-7, random.randint(32, 64), torch.randn((512, 256, 1, 1)) * 1e-7, torch.randn((512,)) * 1e-7, torch.randn((512,)) * 1e-7, torch.randn((512,)) * 1e-7, torch.randn((512,)) * 1e-7, random.randint(32, 64), torch.randn((512, 512, 3, 3)) * 1e-7, torch.randn((512,)) * 1e-7, torch.randn((512,)) * 1e-7, torch.randn((512,)) * 1e-7, torch.randn((512,)) * 1e-7, random.randint(32, 64), torch.randn((512, 512, 3, 3)) * 1e-7, torch.randn((512,)) * 1e-7, torch.randn((512,)) * 1e-7, torch.randn((512,)) * 1e-7, torch.randn((512,)) * 1e-7, random.randint(32, 64), torch.randn((512, 512, 3, 3)) * 1e-7, torch.randn((512,)) * 1e-7, torch.randn((512,)) * 1e-7, torch.randn((512,)) * 1e-7, torch.randn((512,)) * 1e-7, random.randint(32, 64), torch.randn((512, 512, 3, 3)) * 1e-7, torch.randn((512,)) * 1e-7, torch.randn((512,)) * 1e-7, torch.randn((512,)) * 1e-7, torch.randn((512,)) * 1e-7, random.randint(32, 64), torch.randn((1000, 512)) * 1e-7, torch.randn((1000,)) * 1e-7]
        self.layer_0 = torch.nn.AdaptiveAvgPool2d([1, 1])

    def jojo_0(self, input_vars_202, input_vars_209, input_vars_207, input_vars_206, input_vars_205, var_1539, var_1495, input_vars_200, input_vars_203, input_vars_208, input_vars_201):
        var_1544 = torch.batch_norm(var_1539, input_vars_200, input_vars_201, input_vars_202, input_vars_203, False, 0.1, 1e-05, True)
        var_1546 = torch.add(var_1544, var_1495, alpha=1)
        var_1547 = torch.relu_(var_1546)
        var_1566 = torch._convolution(var_1547, input_vars_205, None, [1, 1, ], [1, 1, ], [1, 1, ], False, [0, 0, ], 1, False, False, True)
        var_1571 = torch.batch_norm(var_1566, input_vars_206, input_vars_207, input_vars_208, input_vars_209, False, 0.1, 1e-05, True)
        return var_1571, var_1547

    def jojo_1(self, input_vars_195, var_1514, input_vars_197, input_vars_194, input_vars_196):
        var_1519 = torch.batch_norm(var_1514, input_vars_194, input_vars_195, input_vars_196, input_vars_197, False, 0.1, 1e-05, True)
        var_1520 = torch.relu_(var_1519)
        return var_1520

    def jojo_2(self, input_vars_184, input_vars_175, input_vars_179, input_vars_183, input_vars_176, input_vars_187, input_vars_185, input_vars_177, input_vars_178, input_vars_188, input_vars_182, input_vars_189, input_vars_191, input_vars_190, var_1367, input_vars_181, var_1416):
        var_1418 = torch.add(var_1416, var_1367, alpha=1)
        var_1419 = torch.relu_(var_1418)
        var_1438 = torch._convolution(var_1419, input_vars_175, None, [2, 2, ], [1, 1, ], [1, 1, ], False, [0, 0, ], 1, False, False, True)
        var_1443 = torch.batch_norm(var_1438, input_vars_176, input_vars_177, input_vars_178, input_vars_179, False, 0.1, 1e-05, True)
        var_1444 = torch.relu_(var_1443)
        var_1463 = torch._convolution(var_1444, input_vars_181, None, [1, 1, ], [1, 1, ], [1, 1, ], False, [0, 0, ], 1, False, False, True)
        var_1468 = torch.batch_norm(var_1463, input_vars_182, input_vars_183, input_vars_184, input_vars_185, False, 0.1, 1e-05, True)
        var_1487 = torch._convolution(var_1419, input_vars_187, None, [2, 2, ], [0, 0, ], [1, 1, ], False, [0, 0, ], 1, False, False, True)
        var_1492 = torch.batch_norm(var_1487, input_vars_188, input_vars_189, input_vars_190, input_vars_191, False, 0.1, 1e-05, True)
        var_1494 = torch.add(var_1468, var_1492, alpha=1)
        var_1495 = torch.relu_(var_1494)
        return var_1495

    def jojo_3(self, input_vars_166, input_vars_165, input_vars_167, input_vars_163, var_1366, input_vars_164):
        var_1367 = torch.relu_(var_1366.clone())
        var_1386 = torch._convolution(var_1367, input_vars_163, None, [1, 1, ], [1, 1, ], [1, 1, ], False, [0, 0, ], 1, False, False, True)
        var_1391 = torch.batch_norm(var_1386, input_vars_164, input_vars_165, input_vars_166, input_vars_167, False, 0.1, 1e-05, True)
        var_1392 = torch.relu_(var_1391)
        return var_1367, var_1392

    def jojo_4(self, input_vars_158, var_1359, input_vars_160, input_vars_159, input_vars_161):
        var_1364 = torch.batch_norm(var_1359, input_vars_158, input_vars_159, input_vars_160, input_vars_161, False, 0.1, 1e-05, True)
        return var_1364

    def jojo_5(self, input_vars_154, input_vars_147, var_1307, input_vars_152, input_vars_153, var_1263, input_vars_149, input_vars_148, input_vars_146, input_vars_155, input_vars_151):
        var_1312 = torch.batch_norm(var_1307, input_vars_146, input_vars_147, input_vars_148, input_vars_149, False, 0.1, 1e-05, True)
        var_1314 = torch.add(var_1312, var_1263, alpha=1)
        var_1315 = torch.relu_(var_1314)
        var_1334 = torch._convolution(var_1315, input_vars_151, None, [1, 1, ], [1, 1, ], [1, 1, ], False, [0, 0, ], 1, False, False, True)
        var_1339 = torch.batch_norm(var_1334, input_vars_152, input_vars_153, input_vars_154, input_vars_155, False, 0.1, 1e-05, True)
        return var_1315, var_1339

    def jojo_6(self, var_1287):
        var_1288 = torch.relu_(var_1287.clone())
        return var_1288

    def jojo_7(self, var_1263, input_vars_139):
        var_1282 = torch._convolution(var_1263, input_vars_139, None, [1, 1, ], [1, 1, ], [1, 1, ], False, [0, 0, ], 1, False, False, True)
        return var_1282

    def jojo_8(self, input_vars_128, input_vars_130, input_vars_127, input_vars_124, input_vars_134, input_vars_135, var_1183, input_vars_122, input_vars_133, input_vars_137, input_vars_121, input_vars_129, var_1159, input_vars_136, input_vars_123, input_vars_131, input_vars_125):
        var_1184 = torch.relu_(var_1183.clone())
        var_1203 = torch._convolution(var_1184, input_vars_121, None, [1, 1, ], [1, 1, ], [1, 1, ], False, [0, 0, ], 1, False, False, True)
        var_1208 = torch.batch_norm(var_1203, input_vars_122, input_vars_123, input_vars_124, input_vars_125, False, 0.1, 1e-05, True)
        var_1210 = torch.add(var_1208, var_1159, alpha=1)
        var_1211 = torch.relu_(var_1210)
        var_1230 = torch._convolution(var_1211, input_vars_127, None, [1, 1, ], [1, 1, ], [1, 1, ], False, [0, 0, ], 1, False, False, True)
        var_1235 = torch.batch_norm(var_1230, input_vars_128, input_vars_129, input_vars_130, input_vars_131, False, 0.1, 1e-05, True)
        var_1236 = torch.relu_(var_1235)
        var_1255 = torch._convolution(var_1236, input_vars_133, None, [1, 1, ], [1, 1, ], [1, 1, ], False, [0, 0, ], 1, False, False, True)
        var_1260 = torch.batch_norm(var_1255, input_vars_134, input_vars_135, input_vars_136, input_vars_137, False, 0.1, 1e-05, True)
        var_1262 = torch.add(var_1260, var_1211, alpha=1)
        return var_1262

    def jojo_9(self, input_vars_110, input_vars_112, input_vars_101, input_vars_109, input_vars_113, input_vars_98, input_vars_95, input_vars_93, input_vars_115, input_vars_92, input_vars_107, input_vars_91, input_vars_100, input_vars_94, input_vars_97, var_1056, input_vars_103, var_1031, input_vars_104, input_vars_105, input_vars_99, input_vars_111, input_vars_106):
        var_1075 = torch._convolution(var_1056, input_vars_91, None, [1, 1, ], [1, 1, ], [1, 1, ], False, [0, 0, ], 1, False, False, True)
        var_1080 = torch.batch_norm(var_1075, input_vars_92, input_vars_93, input_vars_94, input_vars_95, False, 0.1, 1e-05, True)
        var_1082 = torch.add(var_1080, var_1031, alpha=1)
        var_1083 = torch.relu_(var_1082)
        var_1102 = torch._convolution(var_1083, input_vars_97, None, [2, 2, ], [1, 1, ], [1, 1, ], False, [0, 0, ], 1, False, False, True)
        var_1107 = torch.batch_norm(var_1102, input_vars_98, input_vars_99, input_vars_100, input_vars_101, False, 0.1, 1e-05, True)
        var_1108 = torch.relu_(var_1107)
        var_1127 = torch._convolution(var_1108, input_vars_103, None, [1, 1, ], [1, 1, ], [1, 1, ], False, [0, 0, ], 1, False, False, True)
        var_1132 = torch.batch_norm(var_1127, input_vars_104, input_vars_105, input_vars_106, input_vars_107, False, 0.1, 1e-05, True)
        var_1151 = torch._convolution(var_1083, input_vars_109, None, [2, 2, ], [0, 0, ], [1, 1, ], False, [0, 0, ], 1, False, False, True)
        var_1156 = torch.batch_norm(var_1151, input_vars_110, input_vars_111, input_vars_112, input_vars_113, False, 0.1, 1e-05, True)
        var_1158 = torch.add(var_1132, var_1156, alpha=1)
        var_1159 = torch.relu_(var_1158)
        var_1178 = torch._convolution(var_1159, input_vars_115, None, [1, 1, ], [1, 1, ], [1, 1, ], False, [0, 0, ], 1, False, False, True)
        return var_1159, var_1178

    def jojo_10(self, input_vars_89, input_vars_88, input_vars_87, input_vars_86, var_1028, var_979, input_vars_85):
        var_1030 = torch.add(var_1028, var_979, alpha=1)
        var_1031 = torch.relu_(var_1030)
        var_1050 = torch._convolution(var_1031, input_vars_85, None, [1, 1, ], [1, 1, ], [1, 1, ], False, [0, 0, ], 1, False, False, True)
        var_1055 = torch.batch_norm(var_1050, input_vars_86, input_vars_87, input_vars_88, input_vars_89, False, 0.1, 1e-05, True)
        return var_1055, var_1031

    def jojo_11(self, input_vars_77, input_vars_75, input_vars_69, input_vars_67, input_vars_70, var_952, input_vars_71, input_vars_74, input_vars_79, input_vars_73, var_927, input_vars_76, input_vars_68):
        var_971 = torch._convolution(var_952, input_vars_67, None, [1, 1, ], [1, 1, ], [1, 1, ], False, [0, 0, ], 1, False, False, True)
        var_976 = torch.batch_norm(var_971, input_vars_68, input_vars_69, input_vars_70, input_vars_71, False, 0.1, 1e-05, True)
        var_978 = torch.add(var_976, var_927, alpha=1)
        var_979 = torch.relu_(var_978)
        var_998 = torch._convolution(var_979, input_vars_73, None, [1, 1, ], [1, 1, ], [1, 1, ], False, [0, 0, ], 1, False, False, True)
        var_1003 = torch.batch_norm(var_998, input_vars_74, input_vars_75, input_vars_76, input_vars_77, False, 0.1, 1e-05, True)
        var_1004 = torch.relu_(var_1003)
        var_1023 = torch._convolution(var_1004, input_vars_79, None, [1, 1, ], [1, 1, ], [1, 1, ], False, [0, 0, ], 1, False, False, True)
        return var_1023, var_979

    def jojo_12(self, input_vars_52, input_vars_49, var_850, input_vars_46, input_vars_61, input_vars_43, input_vars_57, input_vars_47, input_vars_55, input_vars_51, input_vars_58, input_vars_62, input_vars_56, input_vars_63, input_vars_50, input_vars_65, input_vars_44, input_vars_53, input_vars_64, input_vars_59, input_vars_45):
        var_851 = torch.relu_(var_850.clone())
        var_870 = torch._convolution(var_851, input_vars_43, None, [2, 2, ], [1, 1, ], [1, 1, ], False, [0, 0, ], 1, False, False, True)
        var_875 = torch.batch_norm(var_870, input_vars_44, input_vars_45, input_vars_46, input_vars_47, False, 0.1, 1e-05, True)
        var_876 = torch.relu_(var_875)
        var_895 = torch._convolution(var_876, input_vars_49, None, [1, 1, ], [1, 1, ], [1, 1, ], False, [0, 0, ], 1, False, False, True)
        var_900 = torch.batch_norm(var_895, input_vars_50, input_vars_51, input_vars_52, input_vars_53, False, 0.1, 1e-05, True)
        var_919 = torch._convolution(var_851, input_vars_55, None, [2, 2, ], [0, 0, ], [1, 1, ], False, [0, 0, ], 1, False, False, True)
        var_924 = torch.batch_norm(var_919, input_vars_56, input_vars_57, input_vars_58, input_vars_59, False, 0.1, 1e-05, True)
        var_926 = torch.add(var_900, var_924, alpha=1)
        var_927 = torch.relu_(var_926)
        var_946 = torch._convolution(var_927, input_vars_61, None, [1, 1, ], [1, 1, ], [1, 1, ], False, [0, 0, ], 1, False, False, True)
        var_951 = torch.batch_norm(var_946, input_vars_62, input_vars_63, input_vars_64, input_vars_65, False, 0.1, 1e-05, True)
        return var_951, var_927

    def jojo_13(self, input_vars_38, input_vars_37, var_824, input_vars_41, input_vars_39, input_vars_40):
        var_843 = torch._convolution(var_824, input_vars_37, None, [1, 1, ], [1, 1, ], [1, 1, ], False, [0, 0, ], 1, False, False, True)
        var_848 = torch.batch_norm(var_843, input_vars_38, input_vars_39, input_vars_40, input_vars_41, False, 0.1, 1e-05, True)
        return var_848

    def jojo_14(self, input_vars_34, input_vars_31, var_798, input_vars_33, input_vars_32, input_vars_35):
        var_799 = torch.relu_(var_798.clone())
        var_818 = torch._convolution(var_799, input_vars_31, None, [1, 1, ], [1, 1, ], [1, 1, ], False, [0, 0, ], 1, False, False, True)
        var_823 = torch.batch_norm(var_818, input_vars_32, input_vars_33, input_vars_34, input_vars_35, False, 0.1, 1e-05, True)
        return var_799, var_823

    def jojo_15(self, input_vars_25, input_vars_26, var_771, input_vars_28, input_vars_29, input_vars_27):
        var_772 = torch.relu_(var_771.clone())
        var_791 = torch._convolution(var_772, input_vars_25, None, [1, 1, ], [1, 1, ], [1, 1, ], False, [0, 0, ], 1, False, False, True)
        var_796 = torch.batch_norm(var_791, input_vars_26, input_vars_27, input_vars_28, input_vars_29, False, 0.1, 1e-05, True)
        return var_796

    def jojo_16(self, input_vars_4, input_vars_5, input_vars_14, input_vars_17, input_vars_11, input_vars_1, input_vars_9, input_vars_2, input_vars_8, input_vars_10, input_vars_19, input_vars_13, input_vars_0, input_vars_15, input_vars_16, input_vars_7, input_vars_3):
        var_675 = torch._convolution(input_vars_0, input_vars_1, None, [2, 2, ], [3, 3, ], [1, 1, ], False, [0, 0, ], 1, False, False, True)
        var_680 = torch.batch_norm(var_675, input_vars_2, input_vars_3, input_vars_4, input_vars_5, False, 0.1, 1e-05, True)
        var_681 = torch.relu_(var_680)
        var_695 = torch.max_pool2d(var_681, [3, 3, ], [2, 2, ], [1, 1, ], [1, 1, ], False)
        var_714 = torch._convolution(var_695, input_vars_7, None, [1, 1, ], [1, 1, ], [1, 1, ], False, [0, 0, ], 1, False, False, True)
        var_719 = torch.batch_norm(var_714, input_vars_8, input_vars_9, input_vars_10, input_vars_11, False, 0.1, 1e-05, True)
        var_720 = torch.relu_(var_719)
        var_739 = torch._convolution(var_720, input_vars_13, None, [1, 1, ], [1, 1, ], [1, 1, ], False, [0, 0, ], 1, False, False, True)
        var_744 = torch.batch_norm(var_739, input_vars_14, input_vars_15, input_vars_16, input_vars_17, False, 0.1, 1e-05, True)
        var_746 = torch.add(var_744, var_695, alpha=1)
        var_747 = torch.relu_(var_746)
        var_766 = torch._convolution(var_747, input_vars_19, None, [1, 1, ], [1, 1, ], [1, 1, ], False, [0, 0, ], 1, False, False, True)
        return var_766, var_747

    def forward_(self, input_vars):
        var_766, var_747 = torch.utils.checkpoint.checkpoint(self.jojo_16, input_vars[4], input_vars[5], input_vars[14], input_vars[17], input_vars[11], input_vars[1], input_vars[9], input_vars[2], input_vars[8], input_vars[10], input_vars[19], input_vars[13], input_vars[0], input_vars[15], input_vars[16], input_vars[7], input_vars[3])
        var_771 = torch.batch_norm(var_766, input_vars[20], input_vars[21], input_vars[22], input_vars[23], False, 0.1, 1e-05, True)
        var_796 = torch.utils.checkpoint.checkpoint(self.jojo_15, input_vars[25], input_vars[26], var_771, input_vars[28], input_vars[29], input_vars[27])
        var_798 = torch.add(var_796, var_747, alpha=1)
        var_799, var_823 = torch.utils.checkpoint.checkpoint(self.jojo_14, input_vars[34], input_vars[31], var_798, input_vars[33], input_vars[32], input_vars[35])
        var_824 = torch.relu_(var_823)
        var_848 = torch.utils.checkpoint.checkpoint(self.jojo_13, input_vars[38], input_vars[37], var_824, input_vars[41], input_vars[39], input_vars[40])
        var_850 = torch.add(var_848, var_799, alpha=1)
        var_951, var_927 = torch.utils.checkpoint.checkpoint(self.jojo_12, input_vars[52], input_vars[49], var_850, input_vars[46], input_vars[61], input_vars[43], input_vars[57], input_vars[47], input_vars[55], input_vars[51], input_vars[58], input_vars[62], input_vars[56], input_vars[63], input_vars[50], input_vars[65], input_vars[44], input_vars[53], input_vars[64], input_vars[59], input_vars[45])
        var_952 = torch.relu_(var_951)
        var_1023, var_979 = torch.utils.checkpoint.checkpoint(self.jojo_11, input_vars[77], input_vars[75], input_vars[69], input_vars[67], input_vars[70], var_952, input_vars[71], input_vars[74], input_vars[79], input_vars[73], var_927, input_vars[76], input_vars[68])
        var_1028 = torch.batch_norm(var_1023, input_vars[80], input_vars[81], input_vars[82], input_vars[83], False, 0.1, 1e-05, True)
        var_1055, var_1031 = torch.utils.checkpoint.checkpoint(self.jojo_10, input_vars[89], input_vars[88], input_vars[87], input_vars[86], var_1028, var_979, input_vars[85])
        var_1056 = torch.relu_(var_1055)
        var_1159, var_1178 = torch.utils.checkpoint.checkpoint(self.jojo_9, input_vars[110], input_vars[112], input_vars[101], input_vars[109], input_vars[113], input_vars[98], input_vars[95], input_vars[93], input_vars[115], input_vars[92], input_vars[107], input_vars[91], input_vars[100], input_vars[94], input_vars[97], var_1056, input_vars[103], var_1031, input_vars[104], input_vars[105], input_vars[99], input_vars[111], input_vars[106])
        var_1183 = torch.batch_norm(var_1178, input_vars[116], input_vars[117], input_vars[118], input_vars[119], False, 0.1, 1e-05, True)
        var_1262 = torch.utils.checkpoint.checkpoint(self.jojo_8, input_vars[128], input_vars[130], input_vars[127], input_vars[124], input_vars[134], input_vars[135], var_1183, input_vars[122], input_vars[133], input_vars[137], input_vars[121], input_vars[129], var_1159, input_vars[136], input_vars[123], input_vars[131], input_vars[125])
        var_1263 = torch.relu_(var_1262)
        var_1282 = torch.utils.checkpoint.checkpoint(self.jojo_7, var_1263, input_vars[139])
        var_1287 = torch.batch_norm(var_1282, input_vars[140], input_vars[141], input_vars[142], input_vars[143], False, 0.1, 1e-05, True)
        var_1288 = torch.utils.checkpoint.checkpoint(self.jojo_6, var_1287)
        var_1307 = torch._convolution(var_1288, input_vars[145], None, [1, 1, ], [1, 1, ], [1, 1, ], False, [0, 0, ], 1, False, False, True)
        var_1315, var_1339 = torch.utils.checkpoint.checkpoint(self.jojo_5, input_vars[154], input_vars[147], var_1307, input_vars[152], input_vars[153], var_1263, input_vars[149], input_vars[148], input_vars[146], input_vars[155], input_vars[151])
        var_1340 = torch.relu_(var_1339)
        var_1359 = torch._convolution(var_1340, input_vars[157], None, [1, 1, ], [1, 1, ], [1, 1, ], False, [0, 0, ], 1, False, False, True)
        var_1364 = torch.utils.checkpoint.checkpoint(self.jojo_4, input_vars[158], var_1359, input_vars[160], input_vars[159], input_vars[161])
        var_1366 = torch.add(var_1364, var_1315, alpha=1)
        var_1367, var_1392 = torch.utils.checkpoint.checkpoint(self.jojo_3, input_vars[166], input_vars[165], input_vars[167], input_vars[163], var_1366, input_vars[164])
        var_1411 = torch._convolution(var_1392, input_vars[169], None, [1, 1, ], [1, 1, ], [1, 1, ], False, [0, 0, ], 1, False, False, True)
        var_1416 = torch.batch_norm(var_1411, input_vars[170], input_vars[171], input_vars[172], input_vars[173], False, 0.1, 1e-05, True)
        var_1495 = torch.utils.checkpoint.checkpoint(self.jojo_2, input_vars[184], input_vars[175], input_vars[179], input_vars[183], input_vars[176], input_vars[187], input_vars[185], input_vars[177], input_vars[178], input_vars[188], input_vars[182], input_vars[189], input_vars[191], input_vars[190], var_1367, input_vars[181], var_1416)
        var_1514 = torch._convolution(var_1495, input_vars[193], None, [1, 1, ], [1, 1, ], [1, 1, ], False, [0, 0, ], 1, False, False, True)
        var_1520 = torch.utils.checkpoint.checkpoint(self.jojo_1, input_vars[195], var_1514, input_vars[197], input_vars[194], input_vars[196])
        var_1539 = torch._convolution(var_1520, input_vars[199], None, [1, 1, ], [1, 1, ], [1, 1, ], False, [0, 0, ], 1, False, False, True)
        var_1571, var_1547 = torch.utils.checkpoint.checkpoint(self.jojo_0, input_vars[202], input_vars[209], input_vars[207], input_vars[206], input_vars[205], var_1539, var_1495, input_vars[200], input_vars[203], input_vars[208], input_vars[201])
        var_1572 = torch.relu_(var_1571)
        var_1591 = torch._convolution(var_1572, input_vars[211], None, [1, 1, ], [1, 1, ], [1, 1, ], False, [0, 0, ], 1, False, False, True)
        var_1596 = torch.batch_norm(var_1591, input_vars[212], input_vars[213], input_vars[214], input_vars[215], False, 0.1, 1e-05, True)
        var_1598 = torch.add(var_1596, var_1547, alpha=1)
        var_1599 = torch.relu_(var_1598)
        var_1615 = self.layer_0(var_1599)
        var_1618 = torch.flatten(var_1615, 1, -1)
        var_1619 = torch.t(input_vars[217])
        var_1622 = torch.addmm(input_vars[218], var_1618, var_1619, beta=1, alpha=1)
        return var_1622
    
    def forward(self, inputs):
        return self.forward_([ inputs.requires_grad_(True) ] + self.weights)
