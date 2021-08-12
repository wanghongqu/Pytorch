import torch

tensor = torch.Tensor([[2,2,4],[1,2,3]])
print(torch.argmax(tensor.softmax(axis=-1),dim=-1) == torch.Tensor([2,2]))
