from utils import checkpointing
from utils import src_builder
import torch
import torchvision

if __name__ == '__main__':
    model_vgg = torchvision.models.vgg16()
    inputs_vgg = torch.zeros([32, 3, 224, 224])
    output = checkpointing.auto_checkpoint(model_vgg, inputs_vgg, 512000000, verbose=True)
    with open('vgg16.py', 'w') as fp:
        fp.write(to_python_src('JoJo', output.params, output.start,\
                                           output.graph, output.checkpoints))