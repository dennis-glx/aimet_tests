
import torch
from model import CNN
from dataloaders import get_dataloaders
from mnist_utils import mnist_evaluate_model
from compression_tactics import spatial_svd_auto_mode, channel_pruning_auto_mode
from aimet_torch.visualize_serialized_data import VisualizeCompression
from aimet_common.utils import start_bokeh_server_session
# Get MNIST dataloaders
train_loader, eval_loader = get_dataloaders(batch_size_train=64, batch_size_test=1000)

visualization_url, process = start_bokeh_server_session(5006)
model = CNN()
model.load_state_dict(torch.load("./weights_cnn.pt"))

# Spatial SVD
spatial_svd_model, spatial_svd_stats = spatial_svd_auto_mode(model, eval_loader, mnist_evaluate_model)
torch.save(spatial_svd_model, "spatial_svd_weights.pt")

channel_pruned_model, channel_pruned_stats = channel_pruning_auto_mode(spatial_svd_model, train_loader, mnist_evaluate_model)

comp_ratios_file_path = './data/greedy_selection_comp_ratios_list.pkl'
eval_scores_path = './data/greedy_selection_eval_scores_dict.pkl'

compression_visualizations = VisualizeCompression(visualization_url)
compression_visualizations.display_eval_scores(eval_scores_path)
compression_visualizations.display_comp_ratio_plot(comp_ratios_file_path)