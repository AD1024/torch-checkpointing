from utils import checkpointing
from utils import src_builder
import torch
import torchvision

if __name__ == '__main__':
    model_resnet50  = torchvision.models.resnet50()
    inputs_resnet50 = torch.zeros([64, 3, 224, 224])
    output = checkpointing.auto_checkpoint(model_resnet50, inputs_resnet50, 163840000)
    with open('resnet50.py', 'w') as fp:
        fp.write(src_builder.to_python_src('ResNet50', output.params, output.start,\
                                           output.graph, output.checkpoints))
