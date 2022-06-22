# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Input/output checkpointing."""

import os
import random
import re
import sys
import numpy as np

import torch
from torch.nn.parallel import DistributedDataParallel as torchDDP

import torch.nn.functional as F

from megatron import mpu, get_args
from megatron import get_args
from megatron import print_rank_0

_CHECKPOINT_VERSION = None

def set_checkpoint_version(value):
    global _CHECKPOINT_VERSION
    assert _CHECKPOINT_VERSION is None, \
        "checkpoint version already set"
    _CHECKPOINT_VERSION = value

def get_checkpoint_version():
    global _CHECKPOINT_VERSION
    return _CHECKPOINT_VERSION

def check_checkpoint_args(checkpoint_args):
    """Ensure fixed arguments for a model are the same for the input
    arguments and the one retreived frm checkpoint."""
    args = get_args()

    def _compare(arg_name):
        checkpoint_value = getattr(checkpoint_args, arg_name)
        args_value = getattr(args, arg_name)
        error_message = '{} value from checkpoint ({}) is not equal to the ' \
                        'input argument value ({}).'.format(
                            arg_name, checkpoint_value, args_value)
        assert checkpoint_value == args_value, error_message

    _compare('num_layers')
    _compare('hidden_size')
    _compare('num_attention_heads')
    _compare('max_position_embeddings')
    _compare('make_vocab_size_divisible_by')
    _compare('padded_vocab_size')
    _compare('tokenizer_type')
    _compare('model_parallel_size')


def ensure_directory_exists(filename):
    """Build filename's path if it does not already exists."""
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_checkpoint_name(checkpoints_path, iteration,
                        release=False, mp_rank=None):
    """A unified checkpoint name."""
    if release:
        directory = 'release'
    else:
        directory = 'iter_{:07d}'.format(iteration)
    return os.path.join(checkpoints_path, directory,
                        'mp_rank_{:02d}'.format(
                            mpu.get_model_parallel_rank() if mp_rank is None
                            else mp_rank),
                        'model_optim_rng.pt')


def get_checkpoint_tracker_filename(checkpoints_path):
    """Tracker file rescords the latest chckpoint during
    training to restart from."""
    return os.path.join(checkpoints_path, 'latest_checkpointed_iteration.txt')


def save_ds_checkpoint(iteration, model, args):
    """Save a model checkpoint."""

    sd = {}
    sd['iteration'] = iteration
    sd['tokens'] = args.tokens
    sd['checkpoint_version'] = 2.0
    sd['args'] = args

    # rng states.
    if not args.no_save_rng:
        sd['random_rng_state'] = random.getstate()
        sd['np_rng_state'] = np.random.get_state()
        sd['torch_rng_state'] = torch.get_rng_state()
        sd['cuda_rng_state'] = torch.cuda.get_rng_state()
        sd['rng_tracker_states'] = mpu.get_cuda_rng_tracker().get_states()

    #megatron model uses state_dict_for_save_checkpointing instead of the standard state_dict
    #state_dict is used by deepspeed for module saving so it needs to point to the right function
    original_state_dict = model.module.state_dict
    model.module.state_dict = model.module.state_dict_for_save_checkpoint
    try:
        model.save_checkpoint(args.save, client_state=sd)
    finally:
        model.module.state_dict = original_state_dict


def save_checkpoint(iteration, model, optimizer, lr_scheduler):
    """Save a model checkpoint."""
    args = get_args()
    # args.save = 'rewrite'

    if args.deepspeed:
        save_ds_checkpoint(iteration, model, args)
    else:
        # Only rank zero of the data parallel writes to the disk.
        if isinstance(model, torchDDP):
            model = model.module
        if mpu.get_data_parallel_rank() == 0:

            # Arguments, iteration, and model.
            state_dict = {}
            state_dict['args'] = args
            state_dict['checkpoint_version'] = 2.0
            state_dict['iteration'] = iteration
            state_dict['tokens'] = args.tokens
            state_dict['model'] = model.state_dict_for_save_checkpoint()

            # Optimizer stuff.
            if not args.no_save_optim:
                if optimizer is not None:
                    state_dict['optimizer'] = optimizer.state_dict()
                if lr_scheduler is not None:
                    state_dict['lr_scheduler'] = lr_scheduler.state_dict()

            # RNG states.
            if not args.no_save_rng:
                state_dict['random_rng_state'] = random.getstate()
                state_dict['np_rng_state'] = np.random.get_state()
                state_dict['torch_rng_state'] = torch.get_rng_state()
                state_dict['cuda_rng_state'] = torch.cuda.get_rng_state()
                state_dict['rng_tracker_states'] \
                    = mpu.get_cuda_rng_tracker().get_states()

            # Save.
            checkpoint_name = get_checkpoint_name(args.save, iteration)
            print('global rank {} is saving checkpoint at iteration {:7d} to {}'.
                format(torch.distributed.get_rank(), iteration,
                        checkpoint_name))
            ensure_directory_exists(checkpoint_name)
            torch.save(state_dict, checkpoint_name)
            print('  successfully saved {}'.format(checkpoint_name))

    # Wait so everyone is done (necessary)
    torch.distributed.barrier()
    # And update the latest iteration
    if torch.distributed.get_rank() == 0:
        tracker_filename = get_checkpoint_tracker_filename(args.save)
        with open(tracker_filename, 'w') as f:
            f.write(str(iteration))
    # Wait so everyone is done (not necessary)
    torch.distributed.barrier()


def load_checkpoint(model, optimizer, lr_scheduler, load_arg='load'):
    """Load a model checkpoint and return the iteration."""
    args = get_args()
    if args.load_release_checkpoint:
        load_checkpoint_new(model, optimizer, lr_scheduler)
        return 0

    load_dir = getattr(args, load_arg)

    if isinstance(model, torchDDP):
        model = model.module
    # Read the tracker file and set the iteration.
    tracker_filename = get_checkpoint_tracker_filename(load_dir)

    # If no tracker file, return iretation zero.
    if not os.path.isfile(tracker_filename):
        print_rank_0('WARNING: could not find the metadata file {} '.format(
            tracker_filename))
        print_rank_0('    will not load any checkpoints and will start from '
                     'random')
        return 0

    # Otherwise, read the tracker file and either set the iteration or
    # mark it as a release checkpoint.
    iteration = 0
    release = False
    with open(tracker_filename, 'r') as f:
        metastring = f.read().strip()
        try:
            iteration = int(metastring)
        except ValueError:
            release = metastring == 'release'
            if not release:
                print_rank_0('ERROR: Invalid metadata file {}. Exiting'.format(
                    tracker_filename))
                sys.exit()

    assert iteration > 0 or release, 'error parsing metadata file {}'.format(
        tracker_filename)

    if args.deepspeed:
        checkpoint_name, state_dict = model.load_checkpoint(load_dir)

        if checkpoint_name is None:
            if mpu.get_data_parallel_rank() == 0:
                print("Unable to load checkpoint.")
            return iteration

    else:
        # Checkpoint.
        checkpoint_name = get_checkpoint_name(load_dir, iteration, release)
        if mpu.get_data_parallel_rank() == 0:
            print('global rank {} is loading checkpoint {}'.format(
                torch.distributed.get_rank(), checkpoint_name))

        # Load the checkpoint.
        try:
            state_dict = torch.load(checkpoint_name, map_location='cpu')
        except ModuleNotFoundError:
            # For backward compatibility.
            print_rank_0(' > deserializing using the old code structure ...')
            sys.modules['fp16.loss_scaler'] = sys.modules[
                'megatron.fp16.loss_scaler']
            state_dict = torch.load(checkpoint_name, map_location='cpu')
            sys.modules.pop('fp16.loss_scaler', None)
        except BaseException:
            print_rank_0('could not load the checkpoint')
            sys.exit()
            # Model.

        # print('>>>', model.state_dict().keys())
        # print('<<<', state_dict['model'].keys())
        if 'model' in state_dict:
            model.load_state_dict(state_dict['model'])
        else:
            # This is a HACK to load deepspeed checkpoint's model state even if not initialized with deepspeed
            model.load_state_dict(state_dict['module'])

        # Optimizer.
        if not release and not args.finetune and not args.no_load_optim:
            try:
                if optimizer is not None:
                    optimizer.load_state_dict(state_dict['optimizer'])
                if lr_scheduler is not None:
                    lr_scheduler.load_state_dict(state_dict['lr_scheduler'])
            except KeyError:
                print_rank_0(
                    'Unable to load optimizer from checkpoint {}. '
                    'Specify --no-load-optim or --finetune to prevent '
                    'attempting to load the optimizer state, '
                    'exiting ...'.format(checkpoint_name))
                sys.exit()

    # set checkpoint version
    set_checkpoint_version(state_dict.get('checkpoint_version', 0))

    # Set iteration.
    if args.finetune or release:
        iteration = 0
    else:
        try:
            iteration = state_dict['iteration']
            if 'tokens' in state_dict:
                args.tokens = state_dict['tokens']
        except KeyError:
            try:  # Backward compatible with older checkpoints
                iteration = state_dict['total_iters']
            except KeyError:
                print_rank_0('A metadata file exists but unable to load '
                             'iteration from checkpoint {}, exiting'.format(
                                 checkpoint_name))
                sys.exit()


    # Check arguments.
    if 'args' in state_dict:
        checkpoint_args = state_dict['args']
        check_checkpoint_args(checkpoint_args)
    else:
        print_rank_0('could not find arguments in the checkpoint ...')

    # rng states.
    if not release and not args.finetune and not args.no_load_rng:
        try:
            random.setstate(state_dict['random_rng_state'])
            np.random.set_state(state_dict['np_rng_state'])
            torch.set_rng_state(state_dict['torch_rng_state'])
            torch.cuda.set_rng_state(state_dict['cuda_rng_state'])
            mpu.get_cuda_rng_tracker().set_states(
                state_dict['rng_tracker_states'])
        except KeyError:
            print_rank_0('Unable to load optimizer from checkpoint {}. '
                         'Specify --no-load-rng or --finetune to prevent '
                         'attempting to load the optimizer state, '
                         'exiting ...'.format(checkpoint_name))
            sys.exit()

    torch.distributed.barrier()
    if mpu.get_data_parallel_rank() == 0:
        print('  successfully loaded {}'.format(checkpoint_name))

    return iteration


def load_ict_checkpoint(model, only_query_model=False, only_block_model=False, from_realm_chkpt=False):
    """selectively load ICT models for indexing/retrieving from ICT or REALM checkpoints"""

    args = get_args()

    if isinstance(model, torchDDP):
        model = model.module

    load_path = args.load if from_realm_chkpt else args.ict_load

    tracker_filename = get_checkpoint_tracker_filename(load_path)
    with open(tracker_filename, 'r') as f:
        iteration = int(f.read().strip())

    # assert iteration > 0
    checkpoint_name = get_checkpoint_name(load_path, iteration, False)
    if mpu.get_data_parallel_rank() == 0:
        print('global rank {} is loading checkpoint {}'.format(
            torch.distributed.get_rank(), checkpoint_name))

    state_dict = torch.load(checkpoint_name, map_location='cpu')
    ict_state_dict = state_dict['model']
    if from_realm_chkpt and mpu.get_data_parallel_rank() == 0:
        print(" loading ICT state dict from REALM", flush=True)
        ict_state_dict = ict_state_dict['retriever']['ict_model']

    if only_query_model:
        ict_state_dict.pop('context_model')
    if only_block_model:
        ict_state_dict.pop('question_model')

    model.load_state_dict(ict_state_dict)
    torch.distributed.barrier()

    if mpu.get_data_parallel_rank() == 0:
        print(' successfully loaded {}'.format(checkpoint_name))

    return model


def load_checkpoint_new(model, optimizer, lr_scheduler):
    args = get_args()
    load_dir = args.load
    mp_rank = mpu.get_model_parallel_rank()
    mp_size = args.model_parallel_size
    if mp_size == 1:
        mp_str = ''
    else:
        mp_str = f' [mp {mp_rank:02d} / {mp_size}]'

    torch.distributed.barrier()
    print_rank_0(f'> Start loading from release checkpoint from folder {load_dir}')

    state_dict = model.state_dict()
    is_loaded = dict.fromkeys(state_dict.keys(), False)

    for name_pt in os.listdir(load_dir):
        if not re.fullmatch(r'layer_\d\d-model_00-model_states\.pt', name_pt):
            print(f'>> Found {name_pt}, skipping it')
            continue

        fname = os.path.join(load_dir, name_pt)
        print(f'>> Loading {name_pt} on CPU{mp_str}')
        part = torch.load(fname, map_location='cpu')

        for key, weight in part.items():
            key_converted = map_key(key, name_pt, args.num_layers)
            if key_converted is None or key_converted not in state_dict:
                print(f'>>> Skip {key} (converted: {key_converted}) which is not in state_dict{mp_str}')
                continue

            print(f'>>> Setting {key} to {key_converted}{mp_str}')
            old_shape = weight.shape
            weight = pad_weight_if_needed(key, weight, args._intermediate_pad)
            if old_shape != weight.shape:
                print(f'>>>> Pad {key} from {old_shape} to {weight.shape}')

            tensor = state_dict[key_converted]
            if weight.shape == tensor.shape:
                tensor.copy_(weight)
            else:
                assert mp_size > 1, f"mp is 1, but loaded {key} shape: {weight.shape}, state_dict {key_converted}: {tensor.shape}"
                assert len(weight.shape) == len(tensor.shape)
                diff_dim = None
                for i in range(len(weight.shape)):
                    if weight.shape[i] != tensor.shape[i]:
                        assert diff_dim is None, f"Loaded {key} shape: {weight.shape}, state_dict {key_converted}: {tensor.shape}"
                        diff_dim = i
                num_partitions = mpu.divide(weight.shape[diff_dim], tensor.shape[diff_dim])
                weights = split_into_partitions(weight, num_partitions, diff_dim)
                weight_mp = weights[mp_rank]
                assert weight_mp.shape == tensor.shape, f"Split didn't help on {key}: checkpoint is {weight.shape}, state_dict is {tensor.shape}, split is {weight_mp.shape}{mp_str}"
                tensor.copy_(weight_mp)
            is_loaded[key_converted] = True

    for key, flag in is_loaded.items():
        if not flag:
            print(f'> !!! {key} has not been found in checkpoint{mp_str}')
    torch.distributed.barrier()
    print('> Finish loading from release checkpoint')


def map_key(key, name_pt, num_layers):
    '''Map state dict key from checkpoint to the current model'''
    num_pt = int(name_pt[6:8])
    if num_pt == 0:
        return f'language_model.embedding.{key}'
    if num_pt == 1:
        return f'language_model.projector.{key}'
    num_state = num_pt - 3
    if 0 <= num_state < num_layers:
        return f'language_model.transformer.layers.{num_state}.{key}'
    if num_pt == num_layers + 4:
        return f'language_model.output_layer.{key}'


def split_into_partitions(tensor, num_partitions, partition_dim, stride=1):

    per_partition_size = mpu.utils.divide(tensor.size(partition_dim), num_partitions)
    per_partition_per_stride_size = mpu.utils.divide(per_partition_size, stride)

    partitions_list = torch.split(tensor, per_partition_per_stride_size, dim=partition_dim)

    partitions = []
    for i in range(num_partitions):
        partition = torch.cat(partitions_list[i::num_partitions], dim=partition_dim)
        partitions.append(partition)

    return partitions


def pad_weight_if_needed(key, weight, intermediate_pad):
    if intermediate_pad == 0:
        return weight
    if 'dense_ffn_gate' in key or 'dense_ffn_hidden' in key:
        if 'weight' in key:
            return F.pad(weight, (0, 0, 0, intermediate_pad))
        else:
            return F.pad(weight, (0, intermediate_pad))
    if 'dense_ffn_output.weight' in key:
        return F.pad(weight, (0, intermediate_pad))
    return weight
