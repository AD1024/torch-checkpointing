from utils import checkpointing, src_builder

if __name__ == '__main__':
    output = checkpointing.auto_checkpoint(checkpointing.torchvision.models.resnet18(), checkpointing.torch.zeros([64, 3, 7, 7]), 1638400)
    src_builder.to_python_src(output.start, output.graph)