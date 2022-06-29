import torch, torch.nn.functional as F
from dataloaders import get_dataloaders

def mnist_evaluate_model(model: torch.nn.Module, eval_iterations: int, use_cuda: bool = False) -> float:
    """
    This is intended to be the user-defined model evaluation function.
    AIMET requires the above signature. So if the user's eval function does not
    match this signature, please create a simple wrapper.

    Note: Honoring the number of iterations is not absolutely necessary.
    However if all evaluations run over an entire epoch of validation data,
    the runtime for AIMET compression will obviously be higher.

    :param model: Model to evaluate
    :param eval_iterations: Number of iterations to use for evaluation.
            None for entire epoch.
    :param use_cuda: If true, evaluate using gpu acceleration
    :return: single float number (accuracy) representing model's performance
    
    """
    train_loader, eval_loader = get_dataloaders(batch_size_train=64, batch_size_test=1000)

    model.eval()
    test_loss = 0
    correct   = 0
    predictions = []

    with torch.no_grad():
        for data, target in eval_loader:
            output     = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred       = output.data.max(1, keepdim = True)[1]
            correct   += pred.eq(target.data.view_as(pred)).sum()

        test_loss /= len(eval_loader.dataset)
        accuracy = 100. * correct / len(eval_loader.dataset)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(eval_loader.dataset), accuracy))

    return float(accuracy)
