from utils import checkpointing, src_builder
import torch
import torchvision

if __name__ == '__main__':
    model_resnet50  = torchvision.models.resnet50()
    inputs_resnet50 = torch.zeros([1, 3, 224, 224])
    output = checkpointing.auto_checkpoint(model_resnet50, inputs_resnet50, 1638400, verbose=True)
    print(src_builder.to_python_src(output.params, output.start, output.graph))