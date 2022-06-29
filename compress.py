
import torch
from model import CNN
from dataloaders import get_dataloaders
from mnist_utils import mnist_evaluate_model
from compression_tactics import spatial_svd_auto_mode, channel_pruning_auto_mode

# Get MNIST dataloaders
train_loader, eval_loader = get_dataloaders(batch_size_train=64, batch_size_test=1000)


model = CNN()
model.load_state_dict(torch.load("./weights_cnn.pt"))

# Spatial SVD
spatial_svd_model, spatial_svd_stats = spatial_svd_auto_mode(model, eval_loader, mnist_evaluate_model)
torch.save(spatial_svd_model, "spatial_svd_weights.pt")

channel_pruned_model, channel_pruned_stats = channel_pruning_auto_mode(spatial_svd_model, train_loader, mnist_evaluate_model)