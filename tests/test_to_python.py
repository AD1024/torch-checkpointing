import unittest

class ToPythonTest(unittest.TestCase):

    def run_resnet18(self):
        model_resnet18 = torchvision.models.resnet18()
        inputs_resnet18 = torch.zeros([64, 3, 7, 7])
        output = checkpointing.auto_checkpoint(model_resnet18, inputs_resnet18, 1638400)
        print('CheckpointedResnet18', src_builder.to_python_src(output.params, output.start, output.graph, output.checkpoints))
    
    def run_resnet50(self):
        model_resnet50  = torchvision.models.resnet50()
        inputs_resnet50 = torch.zeros([1, 3, 224, 224])
        output = checkpointing.auto_checkpoint(model_resnet50, inputs_resnet50, 1638400)
        print('CheckpointedResnet50', src_builder.to_python_src(output.params, output.start, output.graph, output.checkpoints)
    
    def run_vgg16(self):
        model_vgg = torchvision.models.vgg16()
        inputs_vgg = torch.zeros([1, 3, 224, 224])
        output = checkpointing.auto_checkpoint(model_vgg, inputs_vgg, 1638400)
        print('CheckpointedVgg16', src_builder.to_python_src(output.params, output.start, output.graph, output.checkpoints))

if __name__ == '__main__':
    unittest.main()