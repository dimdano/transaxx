from collections import defaultdict, deque, OrderedDict
import copy
import datetime
import hashlib
import time
import torch
import torch.distributed as dist
from torchvision.datasets import CIFAR10, ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms as T

#np.random.seed(args.seed)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic=True

import errno
import os
from tqdm import tqdm
import timeit
import numpy as np

from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib

from layers.adapt_linear_layer import AdaPT_Linear
from layers.adapt_convolution_layer import AdaptConv2D

def replace_linear_layers(model, custom_linear_class, axx_list, total_macs, total_params, layer_count = [0], returned_power = [0], initialize = True):
    """
    Recursively replaces all linear layers in with a custom linear layer class, preserving their weights and biases.
    layer_count argument is single value list so that is passed by reference to count id of layers
    """
    if(layer_count[0]>=len(axx_list)):
        return
    for name, module in model.named_children():
        name_flag = ("head" not in name and "reduction" not in name)
        if name_flag and initialize and (isinstance(module, torch.nn.Linear) or isinstance(module, AdaPT_Linear)):
            # Determine the device where the new tensor should be created   
            device = module.weight.device
            setattr(model, name, custom_linear_class(module.in_features, module.out_features,
                                       axx_mult = axx_list[layer_count[0]]['axx_mult'],
                                       quant_bits = axx_list[layer_count[0]]['quant_bits'],
                                       fake_quant= axx_list[layer_count[0]]['fake_quant']).to(device))
            getattr(model, name).set_axx_kernel()
            getattr(model, name).weight.data.copy_(module.weight.data)
            if module.bias is not None:
                getattr(model, name).bias.data.copy_(module.bias.data)
            layer_count[0] += 1
            
        elif name_flag and not initialize and isinstance(module, AdaPT_Linear): 
            # ** important ** we assume weights/biases are already loaded and we keep same quant_bits
            # ** so that further calibration is not needed
            #if(hasattr(module, "axx_mult")):
            #    quantizer_t = module.quantizer
            #    quantizer_w_t = module.quantizer_w 
            #change axx kernel name and extension in current module
            module.axx_mult = axx_list[layer_count[0]]['axx_mult']
            module.quant_bits = axx_list[layer_count[0]]['quant_bits']
            module.fake_quant= axx_list[layer_count[0]]['fake_quant']
            module.set_axx_kernel()
            #if(hasattr(module, "axx_mult")):
            #    getattr(model, name).quantizer = quantizer_t    
            #    getattr(model, name).quantizer_w = quantizer_w_t  
            module.flops_power_mem_percent(total_macs, total_params, axx_list[layer_count[0]]['axx_power'])  
            returned_power[0] += module.power_percentage
            layer_count[0] += 1  
        elif isinstance(module, torch.nn.Module):
            replace_linear_layers(module, custom_linear_class, axx_list, total_macs, total_params, layer_count=layer_count, returned_power=returned_power, initialize=initialize)

def replace_conv_layers(model, custom_conv_class, axx_list, total_macs, total_params, layer_count = [0], returned_power = [0], initialize = True):
    """
    Recursively replaces all conv2d layers in with a custom linear layer class, preserving their weights and biases.
    layer_count argument is single value list so that is passed by reference to count id of layers
    """
    if(layer_count[0]>=len(axx_list)):
        return
    for name, module in model.named_children():
        name_flag = ("head" not in name and "reduction" not in name)
        if name_flag and initialize and (isinstance(module, torch.nn.Conv2d) or isinstance(module, AdaptConv2D)):
            # Determine the device where the new tensor should be created   
            device = module.weight.device
            setattr(model, name, custom_conv_class(module.in_channels, module.out_channels, module.kernel_size, module.stride, module.padding, module.dilation, module.groups, module.bias,
                                         axx_mult = axx_list[layer_count[0]]['axx_mult'],
                                         quant_bits = axx_list[layer_count[0]]['quant_bits'],
                                         fake_quant= axx_list[layer_count[0]]['fake_quant']).to(device))
            getattr(model, name).set_axx_kernel()
            getattr(model, name).weight.data.copy_(module.weight.data)
            if module.bias is not None:
                getattr(model, name).bias.data.copy_(module.bias.data)
            layer_count[0] += 1
            
        elif name_flag and not initialize and isinstance(module, AdaptConv2D): 
            # ** important ** we assume weights/biases are already loaded and we keep same quant_bits
            # ** so that further calibration is not needed
            #if(hasattr(module, "axx_mult")):
            #    quantizer_t = module.quantizer
            #    quantizer_w_t = module.quantizer_w 
            #change axx kernel name and extension in current module
            module.axx_mult = axx_list[layer_count[0]]['axx_mult']
            module.quant_bits = axx_list[layer_count[0]]['quant_bits']
            module.fake_quant= axx_list[layer_count[0]]['fake_quant']
            module.set_axx_kernel()
            #if(hasattr(module, "axx_mult")):
            #    getattr(model, name).quantizer = quantizer_t    
            #    getattr(model, name).quantizer_w = quantizer_w_t  
            module.flops_power_mem_percent(total_macs, total_params, axx_list[layer_count[0]]['axx_power'])  
            returned_power[0] += module.power_percentage
            layer_count[0] += 1  
        elif isinstance(module, torch.nn.Module):
            replace_conv_layers(module, custom_conv_class, axx_list, total_macs, total_params, layer_count=layer_count, returned_power=returned_power, initialize=initialize)

def evaluate_cifar10(model, data, device='cuda'):
      
    correct = 0
    total = 0  
    model.eval()
    
    start_time = timeit.default_timer()
    with torch.no_grad():
        for iteraction, (images, labels) in tqdm(enumerate(data), total=len(data)):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(timeit.default_timer() - start_time)
    print('Accuracy of the network on the 10000 test images: %.4f %%' % (
        100 * correct / total))
    return 100 * correct / total
    
def cifar10_data_loader (data_path, batch_size=128):
    """
    This function takes path of cifar10 dataset and returns the test data and a calib data (10% of train data)
    """
    
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2471, 0.2435, 0.2616)

    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )

    test_dataset = CIFAR10(root=data_path, train=False, download=True, transform=transform)
    train_dataset = CIFAR10(root=data_path, train=True, download=True, transform=transform)

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=8,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
    )
    
    #sample 10% of train data to create calib dataset
    evens = list(range(0, len(train_dataset), 10))
    calib_dataset = torch.utils.data.Subset(train_dataset, evens)
    calib_dataloader = DataLoader(
        calib_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        drop_last=True,
        pin_memory=True,
    )
        
    return test_dataloader, calib_dataloader


def imagenet_data_loader (data_path, batch_size=128):
    
    val_dir = os.path.join(data_path + '/val')
    calib_dir = os.path.join(data_path + '/train_tiny')

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    val_data = ImageFolder(val_dir, T.Compose([
        T.Resize (256),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize,
    ]))
    
    calib_data = ImageFolder(calib_dir, T.Compose([
        T.Resize (256),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize,
    ]))

    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=batch_size, shuffle=False,
        num_workers=16, pin_memory=True)
    
    # calib_loader is used for calibration purposes and is a subset of train dataset
    #evens = list(range(0, len(calib_data), 50))
    #subset = torch.utils.data.Subset(calib_data, evens)
    calib_loader = torch.utils.data.DataLoader(
        calib_data,
        batch_size=batch_size, shuffle=False,
        num_workers=16, pin_memory=True)
    
    return val_loader, calib_loader



def evaluate_imagenet(model, data, criterion, print_freq=100, device = 'cuda'):
    with torch.no_grad():
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to evaluate mode
        model.eval()
        print("Starting evaluation...")

        for i, (input, target) in tqdm(enumerate(data), total=len(data)):
            target = target.to(device)
            input = input.to(device)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))
            losses.update(loss.data.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            #print("iteration ", print_freq)
            if (i+1) % print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(data), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

        return top1.avg, top5.avg
  

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count  
 

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res    



def collect_stats(model, data_loader, num_batches, device="cuda"):
    """Feed data to the network and collect statistic"""

    # Enable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()

    for i, (image, _) in tqdm(enumerate(data_loader), total=num_batches):
        model(image.to(device))
        if i >= num_batches:
            break

    # Disable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()


def compute_amax(model, device="cuda", **kwargs):
    # Load calib result
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax()
                else:
                    module.load_calib_amax(**kwargs)
            print(f"{name:40}: {module}")
    model.to(device)
    
    
class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {}'.format(header, total_time_str))


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        return res


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    elif hasattr(args, "rank"):
        pass
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    setup_for_distributed(args.rank == 0)


def average_checkpoints(inputs):
    """Loads checkpoints from inputs and returns a model with averaged weights. Original implementation taken from:
    https://github.com/pytorch/fairseq/blob/a48f235636557b8d3bc4922a6fa90f3a0fa57955/scripts/average_checkpoints.py#L16

    Args:
      inputs (List[str]): An iterable of string paths of checkpoints to load from.
    Returns:
      A dict of string keys mapping to various values. The 'model' key
      from the returned dict should correspond to an OrderedDict mapping
      string parameter names to torch Tensors.
    """
    params_dict = OrderedDict()
    params_keys = None
    new_state = None
    num_models = len(inputs)
    for fpath in inputs:
        with open(fpath, "rb") as f:
            state = torch.load(
                f,
                map_location=(
                    lambda s, _: torch.serialization.default_restore_location(s, "cpu")
                ),
            )
        # Copies over the settings from the first checkpoint
        if new_state is None:
            new_state = state
        model_params = state["model"]
        model_params_keys = list(model_params.keys())
        if params_keys is None:
            params_keys = model_params_keys
        elif params_keys != model_params_keys:
            raise KeyError(
                "For checkpoint {}, expected list of params: {}, "
                "but found: {}".format(f, params_keys, model_params_keys)
            )
        for k in params_keys:
            p = model_params[k]
            if isinstance(p, torch.HalfTensor):
                p = p.float()
            if k not in params_dict:
                params_dict[k] = p.clone()
                # NOTE: clone() is needed in case of p is a shared parameter
            else:
                params_dict[k] += p
    averaged_params = OrderedDict()
    for k, v in params_dict.items():
        averaged_params[k] = v
        if averaged_params[k].is_floating_point():
            averaged_params[k].div_(num_models)
        else:
            averaged_params[k] //= num_models
    new_state["model"] = averaged_params
    return new_state


def store_model_weights(model, checkpoint_path, checkpoint_key='model', strict=True):
    """
    This method can be used to prepare weights files for new models. It receives as
    input a model architecture and a checkpoint from the training script and produces
    a file with the weights ready for release.

    Examples:
        from torchvision import models as M

        # Classification
        model = M.mobilenet_v3_large(pretrained=False)
        print(store_model_weights(model, './class.pth'))

        # Quantized Classification
        model = M.quantization.mobilenet_v3_large(pretrained=False, quantize=False)
        model.fuse_model()
        model.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
        _ = torch.quantization.prepare_qat(model, inplace=True)
        print(store_model_weights(model, './qat.pth'))

        # Object Detection
        model = M.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=False, pretrained_backbone=False)
        print(store_model_weights(model, './obj.pth'))

        # Segmentation
        model = M.segmentation.deeplabv3_mobilenet_v3_large(pretrained=False, pretrained_backbone=False, aux_loss=True)
        print(store_model_weights(model, './segm.pth', strict=False))

    Args:
        model (pytorch.nn.Module): The model on which the weights will be loaded for validation purposes.
        checkpoint_path (str): The path of the checkpoint we will load.
        checkpoint_key (str, optional): The key of the checkpoint where the model weights are stored.
            Default: "model".
        strict (bool): whether to strictly enforce that the keys
            in :attr:`state_dict` match the keys returned by this module's
            :meth:`~torch.nn.Module.state_dict` function. Default: ``True``

    Returns:
        output_path (str): The location where the weights are saved.
    """
    # Store the new model next to the checkpoint_path
    checkpoint_path = os.path.abspath(checkpoint_path)
    output_dir = os.path.dirname(checkpoint_path)

    # Deep copy to avoid side-effects on the model object.
    model = copy.deepcopy(model)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Load the weights to the model to validate that everything works
    # and remove unnecessary weights (such as auxiliaries, etc)
    model.load_state_dict(checkpoint[checkpoint_key], strict=strict)

    tmp_path = os.path.join(output_dir, str(model.__hash__()))
    torch.save(model.state_dict(), tmp_path)

    sha256_hash = hashlib.sha256()
    with open(tmp_path, "rb") as f:
        # Read and update hash string value in blocks of 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
        hh = sha256_hash.hexdigest()

    output_path = os.path.join(output_dir, "weights-" + str(hh[:8]) + ".pth")
    os.replace(tmp_path, output_path)

    return output_path
