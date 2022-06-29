"""
This file demonstrates the use of quantization using AIMET Cross Layer Equalization (CLE)
and Bias Correction (BC) technique.
"""
import os
import copy 
import logging
import argparse
from datetime import datetime
from functools import partial
import torch
import torch.utils.data as torch_data

# imports for AIMET
import aimet_common
from aimet_torch import bias_correction
from aimet_torch.cross_layer_equalization import equalize_model
from aimet_torch.quantsim import QuantParams, QuantizationSimModel
from aimet_torch import visualize_model
from aimet_common.utils import start_bokeh_server_session
# Custom imports
from model import CNN
from dataloaders import get_dataloaders
from mnist_utils import mnist_evaluate_model


logger = logging.getLogger('TorchCLE-BC')
formatter = logging.Formatter('%(asctime)s : %(name)s - %(levelname)s - %(message)s')
logging.basicConfig(format=formatter)


###
# This script utilizes AIMET to apply Cross Layer Equalization and Bias Correction on a resnet18
# pretrained model with the ImageNet data set. This is intended as a working example to show
# how AIMET APIs can be invoked.

# Scenario parameters:
#    - AIMET quantization accuracy using simulation model
#       - Quant Scheme: 'tf_enhanced'
#       - rounding_mode: 'nearest'
#       - default_output_bw: 8, default_param_bw: 8
#       - Encoding computation using 5 batches of data
#    - AIMET Bias Correction
#       - Quant Scheme: 'tf_enhanced'
#       - rounding_mode: 'nearest'
#       - num_quant_samples: 16
#       - num_bias_correct_samples: 16
#       - ops_to_ignore: None
#    - Input shape: [1, 3, 224, 224]
###
mnist_train_loader, mnist_test_loader = get_dataloaders(batch_size_train=64, batch_size_test=1000)

class MNISTDataPipeline:
    """
    Provides APIs for model quantization using evaluation and finetuning.
    """

    def __init__(self, _config: argparse.Namespace):
        """
        :param _config:
        """
        self._config = _config

    def evaluate(self, model: torch.nn.Module, iterations: int = None, use_cuda: bool = False) -> float:
        """
        Evaluate the specified model using the specified number of samples from the validation set.
        :param model: The model to be evaluated.
        :param iterations: The number of batches of the dataset.
        :param use_cuda: If True then use a GPU for inference.
        :return: The accuracy for the sample with the maximum accuracy.
        """
        return mnist_evaluate_model(model, iterations, use_cuda)


def calculate_quantsim_accuracy(model: torch.nn.Module, evaluator: aimet_common.defs.EvalFunction,
                                use_cuda: bool = False) -> float:
    """
    Calculates quantized model accuracy (INT8) using AIMET QuantizationSim
    :param model: the loaded model
    :param evaluator: the Eval function to use for evaluation
    :param use_cuda: True, if model is placed on GPU
    :return: quantized accuracy of model
    """
    input_shape = (1, 1, 28, 28,)
    if use_cuda:
        dummy_input = torch.rand(input_shape).cuda()
    else:
        dummy_input = torch.rand(input_shape)

    # Number of batches to use for computing encodings
    # Only 5 batches are used here to speed up the process, also the
    # number of images in these 5 batches should be sufficient for
    # compute encodings
    iterations = 5

    quantsim = QuantizationSimModel(model=model, quant_scheme='tf_enhanced',
                                    dummy_input=dummy_input, rounding_mode='nearest',
                                    default_output_bw=8, default_param_bw=8, in_place=False)

    quantsim.compute_encodings(forward_pass_callback=partial(evaluator, use_cuda=use_cuda),
                               forward_pass_callback_args=iterations)

    accuracy = evaluator(quantsim.model, eval_iterations=500, use_cuda=use_cuda)

    return accuracy


def apply_cross_layer_equalization(model: torch.nn.Module, input_shape: tuple):
    """
    Applies CLE on the model and calculates model accuracy on quantized simulator
    Applying CLE on the model inplace consists of:
        - Batch Norm Folding
        - Converts any ReLU6 layers to ReLU layers
        - Cross Layer Scaling
        - High Bias Fold
        - Converts any ReLU6 into ReLU
    :param model: the loaded model
    :param input_shape: the shape of the input to the model
    :return:
    """
    equalize_model(model, input_shape)


def apply_bias_correction(model: torch.nn.Module, data_loader: torch_data.DataLoader):
    """
    Applies Bias-Correction on the model.
    :param model: The model to quantize
    :param evaluator: Evaluator used during quantization
    :param dataloader: DataLoader used during quantization
    :param logdir: Log directory used for storing log files
    :return: None
    """
    # Rounding mode can be 'nearest' or 'stochastic'
    rounding_mode = 'nearest'

    # Number of samples used during quantization
    num_quant_samples = 16

    # Number of samples used for bias correction
    num_bias_correct_samples = 16

    params = QuantParams(weight_bw=8, act_bw=8, round_mode=rounding_mode, quant_scheme='tf_enhanced')

    # Perform Bias Correction
    bias_correction.correct_bias(model, params, num_quant_samples=num_quant_samples,
                                 data_loader=data_loader, num_bias_correct_samples=num_bias_correct_samples)


def cle_bc_example(config: argparse.Namespace):
    """
    Example code that shows the following
    1. Instantiates Data Pipeline for evaluation
    2. Loads the pretrained resnet18 Pytorch model
    3. Calculates Model accuracy
        3.1. Calculates floating point accuracy
        3.2. Calculates Quant Simulator accuracy
    4. Applies AIMET CLE and BC
        4.1. Applies AIMET CLE and calculates QuantSim accuracy
        4.2. Applies AIMET BC and calculates QuantSim accuracy
    :param config: This argparse.Namespace config expects following parameters:
                   tfrecord_dir: Path to a directory containing ImageNet TFRecords.
                                This folder should conatin files starting with:
                                'train*': for training records and 'validation*': for validation records
                   use_cuda: A boolean var to indicate to run the test on GPU.
                   logdir: Path to a directory for logging.
    """

    # Instantiate Data Pipeline for evaluation and training
    visualization_url, process = start_bokeh_server_session(5006)
    data_pipeline = MNISTDataPipeline(config)

    # Load the pretrained resnet18 model
    # model = models.resnet18(pretrained=True)
    # if config.use_cuda:
    #     model.to(torch.device('cuda'))
    
    model = CNN()
    model.load_state_dict(torch.load("./weights_cnn.pt"))

    original_model = CNN()
    original_model.load_state_dict(torch.load("./weights_cnn.pt"))

    # Calculate FP32 accuracy
    accuracy = mnist_evaluate_model(model, eval_iterations=500, use_cuda=config.use_cuda)
    logger.info("Original Model Top-1 accuracy = %.2f", accuracy)

    # Applying cross-layer equalization (CLE)
    # Note that this API will equalize the model in-place
    apply_cross_layer_equalization(model=model, input_shape=(1, 1, 28, 28))

    # Calculate quantized (INT8) accuracy after CLE
    accuracy = calculate_quantsim_accuracy(model=model, evaluator=mnist_evaluate_model, use_cuda=config.use_cuda)
    logger.info("Quantized (INT8) Model Top-1 Accuracy After CLE = %.2f", accuracy)

    # Applying Bias Correction
    # Bias Correction needs representative data samples (a small subset of either the training or validation data)
    data_loader = mnist_test_loader
    
    # Note that this API will bias-correct the model in-place
    apply_bias_correction(model=model, data_loader=data_loader)

    # Calculating accuracy on Quant Simulator
    accuracy = calculate_quantsim_accuracy(model=model, evaluator=mnist_evaluate_model, use_cuda=config.use_cuda)
    logger.info("Quantized (INT8) Model Top-1 Accuracy After Bias Correction = %.2f", accuracy)

    # Save the quantized model
    torch.save(model, "model_cle_bc.pt")

    logger.info("Cross Layer Equalization (CLE) and Bias Correction (BC) complete")
    
    visualize_model.visualize_changes_after_optimization(original_model, model, "./visualisations")


if __name__ == '__main__':
    default_logdir = os.path.join("benchmark_output", "CLE_BC" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))

    parser = argparse.ArgumentParser(description='Apply Cross Layer Equalization and Bias Correction on pretrained '
                                                 'custom model and evaluate on MNIST dataset')

    parser.add_argument('--dataset_dir', type=str,
                        required=True,
                        help="Path to a directory containing ImageNet dataset.\n\
                              This folder should conatin at least 2 subfolders:\n\
                              'train': for training dataset and 'val': for validation dataset")

    parser.add_argument('--use_cuda', action='store_true',
                        required=False,
                        help='Add this flag to run the test on GPU.')

    parser.add_argument('--logdir', type=str,
                        default=default_logdir,
                        help="Path to a directory for logging. "
                             "Default value is 'benchmark_output/weight_svd_<Y-m-d-H-M-S>'")

    _config = parser.parse_args()
    print(_config.use_cuda)
    os.makedirs(_config.logdir, exist_ok=True)

    fileHandler = logging.FileHandler(os.path.join(_config.logdir, "test.log"))
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)

    if _config.use_cuda and not torch.cuda.is_available():
        logger.error('use_cuda is selected but no cuda device found.')
        raise RuntimeError("Found no CUDA Device while use_cuda is selected")

    cle_bc_example(_config)