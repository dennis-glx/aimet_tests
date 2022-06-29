import torch
from decimal import Decimal

# Compression-related imports
from aimet_common.defs import CostMetric, CompressionScheme, GreedySelectionParameters 
from aimet_torch.defs import SpatialSvdParameters, ChannelPruningParameters
from aimet_torch.compress import ModelCompressor


def spatial_svd_auto_mode(model:torch.nn.Module, eval_loader, evaluate_model):
    # Specify the necessary parameters
    greedy_params = GreedySelectionParameters(target_comp_ratio=Decimal(0.4),
                                              num_comp_ratio_candidates=10)
    auto_params = SpatialSvdParameters.AutoModeParams(greedy_params,
                                                      modules_to_ignore=[model.conv1])

    params = SpatialSvdParameters(mode=SpatialSvdParameters.Mode.auto,
                                  params=auto_params, multiplicity=8)

    # Single call to compress the model
    results = ModelCompressor.compress_model(model,
                                             eval_callback=evaluate_model,
                                             eval_iterations=1000,
                                             input_shape=(1, 1, 28, 28),
                                             compress_scheme=CompressionScheme.spatial_svd,
                                             cost_metric=CostMetric.mac,
                                             parameters=params,
                                             visualization_url=None)

    compressed_model, stats = results
    print(compressed_model)
    print(stats)     # Stats object can be pretty-printed easily

    return compressed_model, stats

def channel_pruning_auto_mode(model:torch.nn.Module, train_loader, evaluate_model):

    # Specify the necessary parameters
    greedy_params = GreedySelectionParameters(target_comp_ratio=Decimal(0.4),
                                              num_comp_ratio_candidates=10)
    auto_params = ChannelPruningParameters.AutoModeParams(greedy_params,
                                                          modules_to_ignore=[model.conv1])

    params = ChannelPruningParameters(data_loader=train_loader,
                                      num_reconstruction_samples=500,
                                      allow_custom_downsample_ops=True,
                                      mode=ChannelPruningParameters.Mode.auto,
                                      params=auto_params)

    # Single call to compress the model
    results = ModelCompressor.compress_model(model,
                                             eval_callback=evaluate_model,
                                             eval_iterations=1000,
                                             input_shape=(1, 1, 28, 28),
                                             compress_scheme=CompressionScheme.channel_pruning,
                                             cost_metric=CostMetric.mac,
                                             parameters=params,
                                             visualization_url=None)

    compressed_model, stats = results
    print(compressed_model)
    print(stats)     # Stats object can be pretty-printed easily

    return compressed_model, stats