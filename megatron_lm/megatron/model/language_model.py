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

"""Transformer based language model."""

import torch
import torch.nn.functional as F

from megatron import get_args
from megatron import mpu
from megatron.module import MegatronModule
from megatron.model.transformer import ParallelTransformer, LayerNorm
from megatron.model.utils import get_linear_layer
from megatron.model.utils import init_method_normal, scaled_init_method_normal

import deepspeed

def parallel_lm_logits(input_, word_embeddings_weight, parallel_output,
                       bias=None):
    """LM logits using word embedding weights."""
    # Parallel logits.
    input_parallel = mpu.copy_to_model_parallel_region(input_)
    # Matrix multiply.
    if bias is None:
        logits_parallel = F.linear(input_parallel, word_embeddings_weight)
    else:
        logits_parallel = F.linear(input_parallel, word_embeddings_weight, bias)
    # Gather if needed.
    if parallel_output:
        return logits_parallel

    return mpu.gather_from_model_parallel_region(logits_parallel)


def get_language_model(attention_mask_func, num_tokentypes, add_pooler,
                       init_method=None, scaled_init_method=None):
    """Build language model and return along with the key to save."""
    args = get_args()

    if init_method is None:
        init_method = init_method_normal(args.init_method_std)

    if scaled_init_method is None:
        scaled_init_method = scaled_init_method_normal(args.init_method_std, args.num_layers)

    # Language model.
    language_model = TransformerLanguageModel(
        attention_mask_func=attention_mask_func,
        init_method=init_method,
        output_layer_init_method=scaled_init_method,
        num_tokentypes=num_tokentypes,
        add_pooler=add_pooler)
    # key used for checkpoints.
    language_model_key = 'language_model'

    return language_model, language_model_key


class Pooler(MegatronModule):
    """Pooler layer.

    Pool hidden states of a specific token (for example start of the
    sequence) and add a linear transformation followed by a tanh.

    Arguments:
        hidden_size: hidden size
        init_method: weight initialization method for the linear layer.
            bias is set to zero.
    """

    def __init__(self, hidden_size, init_method):
        super(Pooler, self).__init__()
        self.dense = get_linear_layer(hidden_size, hidden_size, init_method)

    def forward(self, hidden_states, sequence_index=0):
        # hidden_states: [b, s, h]
        # sequence_index: index of the token to pool.
        pooled = hidden_states[:, sequence_index, :]
        pooled = self.dense(pooled)
        pooled = torch.tanh(pooled)
        return pooled


class Embedding(MegatronModule):
    """Language model embeddings.

    Arguments:
        embedding_size: hidden size
        vocab_size: vocabulary size
        max_sequence_length: maximum size of sequence. This
                             is used for positional embedding
        embedding_dropout_prob: dropout probability for embeddings
        init_method: weight initialization method
        num_tokentypes: size of the token-type embeddings. 0 value
                        will ignore this embedding
        scattered_embeddings: perform elementwise-operations on
                              partitioned embedding activations.
                              introduces minor dropout differences
                              betwen MP configurations.
    """

    def __init__(self,
                 embedding_size,
                 vocab_size,
                 max_sequence_length,
                 pos_encoding_type,
                 embedding_dropout_prob,
                 init_method,
                 num_tokentypes=0,
                 scattered_embeddings=False):
        super(Embedding, self).__init__()

        self.embedding_size = embedding_size
        self.init_method = init_method
        self.num_tokentypes = num_tokentypes
        self.scattered_embeddings = scattered_embeddings
        self.pos_encoding_type = pos_encoding_type

        # Word embeddings (parallel).
        self.word_embeddings = mpu.VocabParallelEmbedding(
            vocab_size,
            self.embedding_size,
            init_method=self.init_method
        )
        self._word_embeddings_key = 'word_embeddings'

        # Position embedding (serial).
        if pos_encoding_type == 'trainable_absolute':
            self.position_embeddings = torch.nn.Embedding(
                max_sequence_length, self.embedding_size)
            self._position_embeddings_key = 'position_embeddings'

            with deepspeed.zero.GatheredParameters(self.position_embeddings.weight,
                                                modifier_rank=0):
                # Initialize the position embeddings.
                self.init_method(self.position_embeddings.weight)

        # Token type embedding.
        # Add this as an optional field that can be added through
        # method call so we can load a pretrain model without
        # token types and add them as needed.
        self._tokentype_embeddings_key = 'tokentype_embeddings'
        if self.num_tokentypes > 0:
            self.tokentype_embeddings = torch.nn.Embedding(self.num_tokentypes,
                                                           self.embedding_size)
            with deepspeed.zero.GatheredParameters(self.tokentype_embeddings.weight,
                                                   modifier_rank=0):
                # Initialize the token-type embeddings.
                self.init_method(self.tokentype_embeddings.weight)
        else:
            self.tokentype_embeddings = None

        # Embeddings dropout
        self.embedding_dropout = torch.nn.Dropout(embedding_dropout_prob)

    def add_tokentype_embeddings(self, num_tokentypes):
        """Add token-type embedding. This function is provided so we can add
        token-type embeddings in case the pretrained model does not have it.
        This allows us to load the model normally and then add this embedding.
        """
        if self.tokentype_embeddings is not None:
            raise Exception('tokentype embeddings is already initialized')
        if torch.distributed.get_rank() == 0:
            print('adding embedding for {} tokentypes'.format(num_tokentypes),
                  flush=True)
        self.num_tokentypes = num_tokentypes
        self.tokentype_embeddings = torch.nn.Embedding(num_tokentypes,
                                                       self.embedding_size)
        with deepspeed.zero.GatheredParameters(self.tokentype_embeddings.weight,
                                               modifier_rank=0):
            # Initialize the token-type embeddings.
            self.init_method(self.tokentype_embeddings.weight)

    def forward(self, input_ids, position_ids, tokentype_ids=None):
        if self.scattered_embeddings:
            scatter = mpu.scatter_to_model_parallel_region
            gather = mpu.gather_from_model_parallel_region
        else:
            # do nothing
            scatter = lambda x: x
            gather = lambda x: x

        # Embeddings.
        words_embeddings = scatter(self.word_embeddings(input_ids))

        if self.pos_encoding_type == 'trainable_absolute':
            position_embeddings = scatter(self.position_embeddings(position_ids))
            embeddings = words_embeddings + position_embeddings
        else:
            embeddings = words_embeddings

        if tokentype_ids is not None:
            assert self.tokentype_embeddings is not None
            embeddings = embeddings + scatter(self.tokentype_embeddings(tokentype_ids))
        else:
            assert self.tokentype_embeddings is None

        # Dropout.
        embeddings = gather(self.embedding_dropout(embeddings))

        return embeddings

    def state_dict_for_save_checkpoint(self, destination=None, prefix='',
                                       keep_vars=False):
        """For easy load."""

        state_dict_ = {}
        state_dict_[self._word_embeddings_key] \
            = self.word_embeddings.state_dict(destination, prefix, keep_vars)
        if self.pos_encoding_type == 'trainable_absolute':
            state_dict_[self._position_embeddings_key] \
                = self.position_embeddings.state_dict(
                    destination, prefix, keep_vars)
        if self.num_tokentypes > 0:
            state_dict_[self._tokentype_embeddings_key] \
                = self.tokentype_embeddings.state_dict(
                    destination, prefix, keep_vars)

        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        # Word embedding.
        if self._word_embeddings_key in state_dict:
            state_dict_ = state_dict[self._word_embeddings_key]
        else:
            # for backward compatibility.
            state_dict_ = {}
            for key in state_dict.keys():
                if 'word_embeddings' in key:
                    state_dict_[key.split('word_embeddings.')[1]] \
                        = state_dict[key]
        self.word_embeddings.load_state_dict(state_dict_, strict=strict)

        # Position embedding.
        if self._position_embeddings_key in state_dict:
            state_dict_ = state_dict[self._position_embeddings_key]
        else:
            # for backward compatibility.
            state_dict_ = {}
            for key in state_dict.keys():
                if 'position_embeddings' in key:
                    state_dict_[key.split('position_embeddings.')[1]] \
                        = state_dict[key]
        self.position_embeddings.load_state_dict(state_dict_, strict=strict)

        # Tokentype embedding.
        if self.num_tokentypes > 0:
            state_dict_ = {}
            if self._tokentype_embeddings_key in state_dict:
                state_dict_ = state_dict[self._tokentype_embeddings_key]
            else:
                # for backward compatibility.
                for key in state_dict.keys():
                    if 'tokentype_embeddings' in key:
                        state_dict_[key.split('tokentype_embeddings.')[1]] \
                            = state_dict[key]
            if len(state_dict_.keys()) > 0:
                self.tokentype_embeddings.load_state_dict(state_dict_,
                                                          strict=strict)
            else:
                print('***WARNING*** expected tokentype embeddings in the '
                      'checkpoint but could not find it', flush=True)


class Projector(MegatronModule):
    def __init__(self):
        super().__init__()

        args = get_args()
        self.embedding_size = args.embedding_size
        self.hidden_size = args.hidden_size
        self.apply_residual_connection_post_layernorm = args.apply_residual_connection_post_layernorm

        if not self.apply_residual_connection_post_layernorm:
            self.input_layernorm = LayerNorm(
                args.embedding_size,
                eps=args.layernorm_epsilon
            )

        if args.embedding_size != args.hidden_size:
            self.register_buffer(
                "projector",
                torch.eye(args.embedding_size, args.hidden_size).to(args.params_dtype),
                persistent=False,
            )

    def forward(self, data):
        if self.apply_residual_connection_post_layernorm:
            hidden_states = data
        else:
            hidden_states = self.input_layernorm(data)

        if self.embedding_size != self.hidden_size:
            hidden_states = hidden_states @ self.projector

        return hidden_states


class OutputLayer(MegatronModule):
    def __init__(self, init_method):
        super().__init__()
        args = get_args()

        self.input_layer_norm = LayerNorm(
            args.hidden_size,
            eps=args.layernorm_epsilon
        )

        self.dense = mpu.RowParallelLinear(
            args.hidden_size,
            args.embedding_size,
            input_is_parallel=False,
            init_method=init_method,
            skip_bias_add=False,
        )

        self.activation_func = F.gelu

        self.output_layer_norm = LayerNorm(
            args.embedding_size,
            eps=args.layernorm_epsilon
        )

        self.output_bias = torch.nn.Parameter(
            torch.zeros(
                mpu.divide(
                    args.padded_vocab_size,
                    mpu.get_model_parallel_world_size(),
                )
            )
        )
    
        
    def forward(self, input_data):
        if isinstance(input_data, torch.Tensor):
            hidden_states = input_data
        else:
            assert len(input_data) == 2, f"Unknown presents format, output of transformer of len {len(input_data)} is {input_data}"
            hidden_states = input_data[0]
            presents = input_data[1]

        output = self.input_layer_norm(hidden_states)
        output, _ = self.dense(output)
        output = self.activation_func(output)
        output = self.output_layer_norm(output)
        output = [output, self.output_bias]

        if isinstance(input_data, torch.Tensor):
            return output
        else:
            return [output, presents]


class TransformerLanguageModel(MegatronModule):
    """Transformer language model.

    Arguments:
        transformer_hparams: transformer hyperparameters
        attention_mask_func: a function that takes `unmaksed-attention-scores`
            with size [b, np, s, s] and an `attention-mask` and will apply
            the masking. The function should return a masked score of the
            same size [b, np, s, s].
          masked-attention-scores = attention_mask_func(
                                     unmaksed-attention-scores, attention-mask)
        vocab_size: vocabulary size
        max_sequence_length: maximum size of sequence. This
                             is used for positional embedding
        embedding_dropout_prob: dropout probability for embeddings
        num_tokentypes: size of the token-type embeddings. 0 value
                        will ignore this embedding
    """

    def __init__(self,
                 attention_mask_func,
                 init_method,
                 output_layer_init_method,
                 num_tokentypes=0,
                 add_pooler=False):
        super(TransformerLanguageModel, self).__init__()
        args = get_args()

        self.embedding_size = args.embedding_size
        self.hidden_size = args.hidden_size
        self.num_tokentypes = num_tokentypes
        self.init_method = init_method
        self.add_pooler = add_pooler

        # Embeddings
        self.embedding = Embedding(self.embedding_size,
                                   args.padded_vocab_size,
                                   args.max_position_embeddings,
                                   args.pos_encoding_type,
                                   args.hidden_dropout,
                                   self.init_method,
                                   self.num_tokentypes,
                                   scattered_embeddings=args.scattered_embeddings)
        self._embedding_key = 'embedding'

        self.projector = Projector()

        # Transformer
        self.transformer = ParallelTransformer(
            attention_mask_func, self.init_method, 
            output_layer_init_method)
        self._transformer_key = 'transformer'

        self.output_layer = OutputLayer(init_method=self.init_method)

        # Pooler
        if self.add_pooler:
            self.pooler = Pooler(self.hidden_size, self.init_method)
            self._pooler_key = 'pooler'

    def forward(self, input_ids, position_ids, attention_mask,
                tokentype_ids=None, layer_past=None, get_key_value=False,
                pooling_sequence_index=0):

        # Embeddings.
        embedding_output = self.embedding(input_ids, position_ids,
                                          tokentype_ids=tokentype_ids)

        # Projector!
        projector_output = self.projector(embedding_output)

        # Transformer.
        transformer_output = self.transformer(projector_output,
                                              attention_mask,
                                              layer_past=layer_past,
                                              get_key_value=get_key_value)
        
        # OutputLayer!
        output = self.output_layer(transformer_output)

        if self.add_pooler:
            pooled_output = self.pooler(output,
                                        pooling_sequence_index)
            return output, pooled_output

        return output

    def state_dict_for_save_checkpoint(self, destination=None, prefix='',
                                       keep_vars=False):
        """For easy load."""

        state_dict_ = {}
        state_dict_[self._embedding_key] \
            = self.embedding.state_dict_for_save_checkpoint(
                destination, prefix, keep_vars)
        state_dict_[self._transformer_key] \
            = self.transformer.state_dict_for_save_checkpoint(
                destination, prefix, keep_vars)
        if self.add_pooler:
            state_dict_[self._pooler_key] \
                = self.pooler.state_dict_for_save_checkpoint(
                    destination, prefix, keep_vars)

        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        # Embedding.
        if self._embedding_key in state_dict:
            state_dict_ = state_dict[self._embedding_key]
        else:
            # for backward compatibility.
            state_dict_ = {}
            for key in state_dict.keys():
                if '_embeddings' in key:
                    state_dict_[key] = state_dict[key]
        self.embedding.load_state_dict(state_dict_, strict=strict)

        # Transformer.
        if self._transformer_key in state_dict:
            state_dict_ = state_dict[self._transformer_key]
        else:
            # for backward compatibility.
            state_dict_ = {}
            for key in state_dict.keys():
                if 'transformer.' in key:
                    state_dict_[key.split('transformer.')[1]] = state_dict[key]
        self.transformer.load_state_dict(state_dict_, strict=strict)

        # Pooler.
        if self.add_pooler:
            assert 'pooler' in state_dict, \
                'could not find data for pooler in the checkpoint'
            self.pooler.load_state_dict(state_dict[self._pooler_key],
                                        strict=strict)
