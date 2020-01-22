from utils import checkpointing, src_builder
import torch
import torchvision

if __name__ == '__main__':
    model_vgg = torchvision.models.vgg16()
    inputs_vgg = torch.zeros([1, 3, 224, 224])
    output = checkpointing.auto_checkpoint(model_vgg, inputs_vgg, 16384000, verbose=True)
    print(src_builder.to_python_src(output.params, output.start, output.graph))