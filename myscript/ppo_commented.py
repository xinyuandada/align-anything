# Copyright 2022 The HuggingFace Team. All rights reserved.
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
import inspect
import math
import os
import time
import typing
import warnings
from contextlib import nullcontext
from typing import Callable, List, Optional, Union

import datasets
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, gather_object, is_deepspeed_available
from datasets import Dataset
from huggingface_hub import whoami
from packaging import version
from torch.optim import Adam
from transformers import (
    DataCollatorForLanguageModeling,
    PreTrainedTokenizer,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
)

from ..core import (
    WANDB_PADDING,
    PPODecorators,
    clip_by_value,
    convert_to_scalar,
    entropy_from_logits,
    flatten_dict,
    logprobs_from_logits,
    masked_mean,
    masked_var,
    masked_whiten,
    set_seed,
    stack_dicts,
    stats_to_np,
)
from ..import_utils import is_npu_available, is_torch_greater_2_0, is_xpu_available
from ..models import SUPPORTED_ARCHITECTURES, PreTrainedModelWrapper, create_reference_model
from . import AdaptiveKLController, BaseTrainer, FixedKLController, PPOConfig, RunningMoments


if is_deepspeed_available():
    import deepspeed

MODEL_CARD_TEMPLATE = """---
license: apache-2.0
tags:
- trl
- ppo
- transformers
- reinforcement-learning
---

# {model_name}

这是一个 [TRL 语言模型](https://github.com/huggingface/trl)，它通过强化学习进行了微调，
以根据某个价值、函数或人类反馈来引导模型的输出。该模型可用于文本生成。

## 使用方法

要使用此模型进行推理，首先请安装 TRL 库：

```bash
python -m pip install trl

from transformers import pipeline

generator = pipeline("text-generation", model="{model_id}")
outputs = generator("你好，我的 llama 很可爱")
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead

tokenizer = AutoTokenizer.from_pretrained("{model_id}")
model = AutoModelForCausalLMWithValueHead.from_pretrained("{model_id}")

inputs = tokenizer("你好，我的 llama 很可爱", return_tensors="pt")
outputs = model(**inputs, labels=inputs["input_ids"])
"""

class PPOTrainer(BaseTrainer):
"""
PPOTrainer 使用近端策略优化（Proximal Policy Optimization）来优化语言模型。
注意，这个训练器的设计很大程度上受到了 OpenAI 最初的“从反馈中学习摘要”工作的启发：
https://github.com/openai/summarize-from-feedback


属性:
    **config** (`PPOConfig`) -- PPOTrainer 的配置对象。更多细节请查阅 `PPOConfig` 的文档。
    **model** (`PreTrainedModelWrapper`) -- 需要优化的模型，是一个带有value head的 Hugging Face transformer 模型。
        更多细节请查阅 `PreTrainedModelWrapper` 的文档。
    **ref_model** (`PreTrainedModelWrapper`, *可选*) -- 用于 KL 惩罚的参考模型，是一个带有因果语言建模头的 Hugging Face
        transformer 模型。更多细节请查阅 `PreTrainedModelWrapper` 的文档。如果未提供参考模型，
        训练器将创建一个与待优化模型结构相同且共享层级的参考模型。
    **tokenizer** (`PreTrainedTokenizerBase`) -- 用于数据编码的分词器。
        更多细节请查阅 `transformers.PreTrainedTokenizer` 和 `transformers.PreTrainedTokenizerFast` 的文档。
    **dataset** (Union[`torch.utils.data.Dataset`, `datasets.Dataset`], *可选*) -- PyTorch 数据集或 Hugging
        Face 数据集。这用于创建一个 PyTorch dataloader。如果未提供数据集，dataloader 必须在训练器外部创建，
        用户需要设计自己的 dataloader，并确保使用的批次大小与配置对象中指定的大小相同。
    **optimizer** (`torch.optim.Optimizer`, *可选*) -- 用于训练的优化器。如果未提供优化器，
        训练器将创建一个 Adam 优化器，学习率由配置对象指定。
    **data_collator** (DataCollatorForLanguageModeling, *可选*) -- 用于训练的数据整理器，并传递给 dataloader。
    **num_shared_layers** (int, *可选*) -- 如果未传递参考模型，则此参数指定模型和参考模型之间共享的层数。
        如果未提供数值，则所有层都将被共享。
    **lr_scheduler** (`torch.optim.lr_scheduler`, *可选*) -- 用于训练的学习率调度器。
"""

_tag_names = ["trl", "ppo"]

def __init__(
    self,
    config: Optional[PPOConfig] = None,
    model: Optional[PreTrainedModelWrapper] = None,
    ref_model: Optional[PreTrainedModelWrapper] = None,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    dataset: Optional[Union[torch.utils.data.Dataset, Dataset]] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    data_collator: Optional[typing.Callable] = None,
    num_shared_layers: Optional[int] = None,
    lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
):
    """
    初始化 PPOTrainer。

    参数:
        config (`PPOConfig`):
            PPOTrainer 的配置对象。更多细节请查阅 `PPOConfig` 的文档。
        model (`PreTrainedModelWrapper`):
            带有价值头的 Hugging Face transformer 模型。
        ref_model (`PreTrainedModelWrapper`):
            带有因果语言建模头的 Hugging Face transformer 模型。用于 KL 惩罚。
        tokenizer (`transformers.PreTrainedTokenizerBase`):
            Hugging Face 分词器。
        dataset (Optional[Union[`torch.utils.data.Dataset`, `datasets.Dataset`]]):
            PyTorch 数据集或 Hugging Face 数据集。如果传递的是 Hugging Face 数据集，
            该数据集将通过移除模型不使用的列来进行预处理。如果未传递任何数据集，
            在多 GPU 设置下会发出警告。
        optimizer (Optional[`torch.optim.Optimizer`]):
            用于训练的优化器。如果为 `None`，则默认使用 `Adam`。
        data_collator (Optional[function]):
            数据整理函数。
        num_shared_layers (Optional[int]):
            模型和参考模型之间共享的层数。如果为 `None`，则所有层都共享。
            仅在 `ref_model` 为 `None` 时使用。
        lr_scheduler (Optional[`torch.optim.lr_scheduler`]):
            用于训练的学习率调度器。
    """
    super().__init__(config)

    # 为可复现实验设置初始种子
    set_seed(config.seed)

    # 步骤0：检查位置参数的有效性
    if not isinstance(config, PPOConfig):
        raise ValueError(f"config 必须是 PPOConfig，但得到的是 {type(config)}")
    if not isinstance(tokenizer, (PreTrainedTokenizerBase)):
        raise ValueError(
            f"tokenizer 必须是 PreTrainedTokenizerBase，例如 PreTrainedTokenizer 或 PreTrainedTokenizerFast，但得到的是 {type(tokenizer)}"
        )
    if not isinstance(model, (SUPPORTED_ARCHITECTURES)):
        raise ValueError(
            f"model 必须是 PreTrainedModelWrapper，但得到的是 {type(model)} - 支持的架构有: {SUPPORTED_ARCHITECTURES}"
        )
    
    # 步骤1：初始化 Accelerator
    self.accelerator = Accelerator(
        log_with=config.log_with,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        project_config=ProjectConfiguration(**config.project_kwargs),
        **config.accelerator_kwargs,
    )

    # 步骤1.1：由 accelerator 填充的运行时变量
    config.world_size = self.accelerator.num_processes
    config.global_backward_batch_size = config.backward_batch_size * config.world_size
    config.global_batch_size = config.batch_size * config.world_size

    self.model = model
    self.model_params = filter(lambda p: p.requires_grad, self.model.parameters())
    self.is_encoder_decoder = hasattr(self.model, "is_encoder_decoder")
    self.is_peft_model = getattr(self.model, "is_peft_model", False)
    config.is_encoder_decoder = self.is_encoder_decoder
    config.is_peft_model = self.is_peft_model

    is_using_tensorboard = config.log_with is not None and config.log_with == "tensorboard"
    self.accelerator.init_trackers(
        config.tracker_project_name,
        config=dict(trl_ppo_trainer_config=config.to_dict()) if not is_using_tensorboard else config.to_dict(),
        init_kwargs=config.tracker_kwargs,
    )
    self.is_using_text_environment = getattr(config, "use_text_environment", False)

    if isinstance(ref_model, SUPPORTED_ARCHITECTURES):
        self.ref_model = ref_model
        if num_shared_layers is not None:
            warnings.warn(
                "当提供了 ref_model 时，num_shared_layers 会被忽略。模型和参考模型使用了两个不同的模型，"
                "没有任何层是共享的。",
                UserWarning,
            )
    elif ref_model is None and not self.is_peft_model:
        self.ref_model = create_reference_model(self.model, num_shared_layers=num_shared_layers)
    elif self.is_peft_model:
        self.ref_model = None
    else:
        raise ValueError(
            f"ref_model 必须是 PreTrainedModelWrapper 或 `None`，但得到的是 {type(ref_model)} - 支持的架构有: "
            f"{SUPPORTED_ARCHITECTURES} "
        )
    self.optional_peft_ctx = (
        self.accelerator.unwrap_model(self.model).pretrained_model.disable_adapter
        if self.is_peft_model
        else nullcontext
    )

    if not (isinstance(tokenizer, PreTrainedTokenizer) or isinstance(tokenizer, PreTrainedTokenizerFast)):
        raise ValueError(
            "tokenizer 必须是 transformers.PreTrainedTokenizer 或 transformers.PreTrainedTokenizerFast"
        )
    self.tokenizer = tokenizer

    if dataset is not None and not (isinstance(dataset, torch.utils.data.Dataset) or isinstance(dataset, Dataset)):
        raise ValueError("dataset 必须是 torch.utils.data.Dataset 或 datasets.Dataset")
    elif dataset is None:
        warnings.warn(
            "没有提供数据集。请确保在训练前将 config.batch_size 设置为正确的值。",
            UserWarning,
        )
    self.dataset = dataset
    self._signature_columns = None
    if self.dataset is not None:
        self.dataloader = self.prepare_dataloader(self.dataset, data_collator)
    elif self.dataset is None and self.accelerator.num_processes > 1:
        warnings.warn(
            "没有提供数据集。在多 GPU 设置中，这会导致错误。您应该"
            "自己使用 `dataloader = ppo_trainer.accelerator.prepare(dataloader)`"
            " 和 `torch.utils.data.DataLoader` 来准备您的 dataloader，"
            "或者向 `PPOTrainer` 传递一个数据集。请参考文档获取更多细节。",
            UserWarning,
        )
        self.dataloader = None
    else:
        self.dataloader = None

    # 步骤3：初始化优化器和数据整理器
    self.data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
    if optimizer is None:
        self.optimizer = Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config.learning_rate,
        )
    else:
        self.optimizer = optimizer

    self.lr_scheduler = lr_scheduler
    if self.lr_scheduler is not None:
        lr_scheduler_class = (
            torch.optim.lr_scheduler._LRScheduler
            if not is_torch_greater_2_0()
            else torch.optim.lr_scheduler.LRScheduler
        )

        if not isinstance(self.lr_scheduler, lr_scheduler_class):
            raise ValueError(
                "lr_scheduler 必须是 torch.optim.lr_scheduler._LRScheduler 或 torch.optim.lr_scheduler.LRScheduler (适用于 torch >= 2.0)"
            )

    if self.config.adap_kl_ctrl:
        self.kl_ctl = AdaptiveKLController(self.config.init_kl_coef, self.config.target, self.config.horizon)
    else:
        self.kl_ctl = FixedKLController(self.config.init_kl_coef)

    # DeepSpeed 集成的安全检查
    is_deepspeed_used = self.accelerator.distributed_type == "DEEPSPEED" and hasattr(
        self.accelerator.state, "deepspeed_plugin"
    )

    (
        self.model,
        self.optimizer,
        self.data_collator,
        self.dataloader,
        self.lr_scheduler,
    ) = self.accelerator.prepare(
        self.model,
        self.optimizer,
        self.data_collator,
        self.dataloader,
        self.lr_scheduler,
    )
    if is_deepspeed_used:
        # 量化模型已经被设置在正确的设备上
        if not self.is_peft_model and not (
            getattr(self.ref_model.pretrained_model, "is_loaded_in_8bit", False)
            or getattr(self.ref_model.pretrained_model, "is_loaded_in_4bit", False)
        ):
            self.ref_model = self._prepare_deepspeed(self.ref_model)
    else:
        self.ref_model = self.accelerator.prepare(self.ref_model)

    # 在分布式设置中，只有主进程需要执行日志记录
    # 查阅: https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html
    # 或: https://discuss.pytorch.org/t/use-distributed-data-parallel-correctly/82500/11
    self.is_distributed = self.accelerator.num_processes > 1

    # 初始化当前步数
    self.current_step = 0

    # 初始化用于将模型推送到 Hub 的变量
    if config.push_to_hub_if_best_kwargs:
        if "repo_id" not in config.push_to_hub_if_best_kwargs:
            raise ValueError("您必须指定 repo_id 才能将模型推送到 Hub！")
        self.push_to_hub_kwargs = config.push_to_hub_if_best_kwargs
        self.compare_step = 0
        self.highest_reward = torch.tensor(-float("inf"))

    # 流水线并行（PP）的后处理
    if not getattr(self.model, "is_sequential_parallel", False):
        self.current_device = self.accelerator.device
    else:
        if is_xpu_available():
            self.current_device = torch.device("xpu:0")
        elif is_npu_available():
            self.current_device = torch.device("npu:0")
        else:
            self.current_device = torch.device("cuda:0")

    PPODecorators.optimize_device_cache = self.config.optimize_device_cache

    self.running = RunningMoments(self.accelerator)

def _filter_kwargs(self, kwargs, target_func):
    """
    过滤目标函数支持的关键字参数。

    参数:
        kwargs (dict):
            关键字参数
        target_func (function):
            目标函数
    """
    return {k: v for k, v in kwargs.items() if k in inspect.signature(target_func).parameters.keys()}

def prepare_dataloader(self, dataset: Union[torch.utils.data.Dataset, Dataset], data_collator=None):
    """
    准备用于训练的 dataloader。

    参数:
        dataset (Union[`torch.utils.data.Dataset`, `datasets.Dataset`]):
            PyTorch 数据集或 Hugging Face 数据集。如果传递的是 Hugging Face 数据集，
            该数据集将通过移除模型不使用的列来进行预处理。
        data_collator (Optional[function]):
            数据整理函数。

    返回:
        `torch.utils.data.DataLoader`: PyTorch dataloader
    """
    if isinstance(dataset, Dataset):
        dataset = self._remove_unused_columns(dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=self.config.batch_size,
        collate_fn=data_collator,
        shuffle=True,
        drop_last=True,
    )
    return dataloader

# 改编自 transformers.Trainer._set_signature_columns_if_needed
def _set_signature_columns_if_needed(self):
    if self._signature_columns is None:
        # 检查模型 forward 方法的签名，只保留它接受的参数。
        signature = inspect.signature(self.model.forward)
        self._signature_columns = list(signature.parameters.keys())
        # label => sentiment | 我们需要 query 和 response 用于日志记录
        self._signature_columns += ["label", "query", "response"]

# 改编自 transformers.Trainer._remove_unused_columns
def _remove_unused_columns(self, dataset: "Dataset"):
    if not self.config.remove_unused_columns:
        return dataset
    self._set_signature_columns_if_needed()
    signature_columns = self._signature_columns

    ignored_columns = list(set(dataset.column_names) - set(signature_columns))

    columns = [k for k in signature_columns if k in dataset.column_names]

    if version.parse(datasets.__version__) < version.parse("1.4.0"):
        dataset.set_format(
            type=dataset.format["type"],
            columns=columns,
            format_kwargs=dataset.format["format_kwargs"],
        )
        return dataset
    else:
        return dataset.remove_columns(ignored_columns)

def generate(
    self,
    query_tensor: Union[torch.Tensor, List[torch.Tensor]],
    length_sampler: Optional[Callable] = None,
    batch_size: int = 4,
    return_prompt: bool = True,
    generate_ref_response: bool = False,
    **generation_kwargs,
):
    """
    根据给定的 query tensor 使用模型生成回答。
    调用模型的 `generate` 方法。

    参数:
        query_tensor (`torch.LongTensor`):
            一个形状为 (`seq_len`) 的张量，包含 query tokens，或一个由形状为 (`seq_len`) 的张量组成的列表。
        length_sampler (`Callable`, *可选*):
            一个返回新生成 token 数量的可调用对象。
        batch_size (`int`, *可选*):
            用于生成的批次大小，默认为 `4`。
        return_prompt (`bool`, *可选*):
            如果设为 `False`，则不返回提示，只返回新生成的 tokens，默认为 `True`。
        generate_ref_response (`bool`, *可选*):
            如果设为 `True`，同时也会生成参考模型的回答，默认为 `False`。
        generation_kwargs (dict[str, Any]):
            生成的关键字参数。

    返回:
        `torch.LongTensor`: 一个形状为 (`batch_size`, `gen_len`) 的张量，包含回答的 tokens。
    """
    if generate_ref_response:
        ref_model = self.model if self.is_peft_model else self.ref_model
    if isinstance(query_tensor, List):
        response = self._generate_batched(
            self.model,
            query_tensor,
            length_sampler=length_sampler,
            batch_size=batch_size,
            return_prompt=return_prompt,
            **generation_kwargs,
        )
        if generate_ref_response:
            with self.optional_peft_ctx():
                ref_response = self._generate_batched(
                    ref_model,
                    query_tensor,
                    length_sampler=length_sampler,
                    batch_size=batch_size,
                    return_prompt=return_prompt,
                    **generation_kwargs,
                )

    else:
        if len(query_tensor.shape) == 2:
            raise ValueError(
                "query_tensor 必须是形状为 (`seq_len`) 的张量或由形状为 (`seq_len`) 的张量组成的列表"
            )

        if length_sampler is not None:
            generation_kwargs["max_new_tokens"] = length_sampler()
        response = self.accelerator.unwrap_model(self.model).generate(
            input_ids=query_tensor.unsqueeze(dim=0), **generation_kwargs
        )
        if generate_ref_response:
            with self.optional_peft_ctx():
                ref_response = ref_model.generate(input_ids=query_tensor.unsqueeze(dim=0), **generation_kwargs)

        if not return_prompt and not self.is_encoder_decoder:
            response = response[:, query_tensor.shape[0] :]
            if generate_ref_response:
                ref_response = ref_response[:, query_tensor.shape[0] :]

    if generate_ref_response:
        return response, ref_response
    return response

def _generate_batched(
    self,
    model: PreTrainedModelWrapper,
    query_tensors: List[torch.Tensor],
    length_sampler: Optional[Callable] = None,
    batch_size: int = 4,
    return_prompt: bool = True,
    pad_to_multiple_of: Optional[int] = None,
    remove_padding: bool = True,
    **generation_kwargs,
):
    outputs = []

    padding_side_default = self.tokenizer.padding_side
    if not self.is_encoder_decoder:
        self.tokenizer.padding_side = "left"

    # 以防样本数量少于批次大小
    batch_size = min(len(query_tensors), batch_size)

    for i in range(0, len(query_tensors), batch_size):
        if length_sampler is not None:
            generation_kwargs["max_new_tokens"] = length_sampler()

        # 防止当 query tensors 数量不是批次大小的整数倍时溢出
        end_index = min(len(query_tensors), i + batch_size)

        batch = query_tensors[i:end_index]
        batch_mask = [torch.ones_like(element) for element in batch]
        inputs = {"input_ids": batch, "attention_mask": batch_mask}

        padded_inputs = self.tokenizer.pad(
            inputs,
            padding=True,
            max_length=None,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors="pt",
        ).to(self.current_device)

        generations = self.accelerator.unwrap_model(model).generate(**padded_inputs, **generation_kwargs)

        for generation, mask in zip(generations, padded_inputs["attention_mask"]):
            if not self.is_encoder_decoder:
                output = generation[(1 - mask).sum() :]  # 移除填充
            else:
                output = generation

            if not return_prompt and not self.is_encoder_decoder:
                output = output[(mask).sum() :]  # 移除提示

            if remove_padding and self.tokenizer.eos_token_id in output:
                pad_mask = output == self.tokenizer.eos_token_id
                pad_start = torch.nonzero(pad_mask, as_tuple=False)[0, 0].item()
                output = output[: pad_start + 1]  # 在末尾保留 eos 符

            outputs.append(output)

    self.tokenizer.padding_side = padding_side_default
    return outputs

def _step_safety_checker(
    self,
    batch_size: int,
    queries: List[torch.LongTensor],
    responses: List[torch.LongTensor],
    scores: List[torch.FloatTensor],
    masks: Optional[List[torch.LongTensor]] = None,
):
    """
    检查输入数据对于训练是否有效。

    参数:
        batch_size (int):
            配置文件中的批次大小。
        queries (List[`torch.LongTensor`]):
            包含编码后 query 的张量列表，形状为 (`query_length`)。
        responses (List[`torch.LongTensor`]):
            包含编码后 response 的张量列表，形状为 (`response_length`)。
        scores (List[`torch.FloatTensor`]):
            包含分数的张量列表。
        masks (List[`torch.LongTensor`], *可选*):
            包含掩码的可选张量列表，形状为 (`query_length` + `response_length`)。
    返回:
        `tuple`: 处理后的输入数据。
    """
    for name, tensor_list in zip(["queries", "responses", "scores"], [queries, responses, scores]):
        if not isinstance(tensor_list, list):
            raise ValueError(f"{name} 必须是张量列表 - 但得到的是 {type(tensor_list)}")
        if not isinstance(tensor_list[0], torch.Tensor):
            raise ValueError(f"{name} 中的元素必须是张量 - 但得到的是 {type(tensor_list[0])}")
        if batch_size is not None and len(tensor_list) != batch_size:
            raise ValueError(
                f"批次大小 ({batch_size}) 与样本数量不匹配 - 但 {name} 的数量是 {len(tensor_list)}"
            )

    # 将 queries, scores 和 responses 移动到正确的设备上
    queries = [tensor.to(self.current_device) for tensor in queries]
    responses = [tensor.to(self.current_device) for tensor in responses]
    scores = [tensor.to(self.current_device) for tensor in scores]
    masks = [tensor.to(self.current_device) for tensor in masks] if masks is not None else None

    # 如果需要，压缩 scores 的维度
    for i, score in enumerate(scores):
        if score.dim() > 1:
            raise ValueError(f"Scores 必须是一维的 - 但对于 {score} 得到了 {score.dim()} 维")
        elif score.dim() == 1:
            scores[i] = score.squeeze()

    return queries, responses, scores, masks

@PPODecorators.empty_device_cache()
def step(
    self,
    queries: List[torch.LongTensor],  # 用于从旧模型（离线策略）生成回答的提示列表
    responses: List[torch.LongTensor], # 由旧模型（离线策略）生成的回答列表
    scores: List[torch.FloatTensor], # 与每个回答相关联的奖励列表。每个回答一个奖励（而不是每个 token 一个奖励）
    response_masks: Optional[List[torch.LongTensor]] = None,
):
    """
    根据给定的 queries、模型 responses 和 rewards，运行一个 PPO 优化步骤。

    参数:
        queries (List[`torch.LongTensor`]):
            包含编码后 query 的张量列表，形状为 (`query_length`)。
        responses (List[`torch.LongTensor`]):
            包含编码后 response 的张量列表，形状为 (`response_length`)。
        scores (List[`torch.FloatTensor`]):
            包含分数的张量列表。
        response_masks (List[`torch.FloatTensor`], *可选*)):
            包含 response tokens 掩码的张量列表。

    返回:
        `dict[str, Any]`: 训练统计数据的摘要
    """
    bs = self.config.batch_size

    # queries: 提示的 input_ids
    # responses: 回答的 input_ids
    # scores: 来自奖励模型的分数（每个回答一个）
    # 验证输入张量（检查类型、形状等）
    queries, responses, scores, response_masks = self._step_safety_checker(
        bs, queries, responses, scores, response_masks
    )
    
    # 表示给予回答的奖励。每个回答一个标量值。
    # 形状: (batch_size)
    scores = torch.tensor(scores, device=self.current_device)
    
    # if self.config.use_score_scaling:
    #     # 分数缩放
    #     scores_mean, scores_std = self.running.update(scores)
    #     tensor_to_kwargs = dict(dtype=scores.dtype, device=scores.device)
    #     score_scaling_factor = self.running.std.to(**tensor_to_kwargs) + torch.finfo(scores.dtype).eps
    #     if self.config.use_score_norm:
    #         scores = (scores - self.running.mean.to(**tensor_to_kwargs)) / score_scaling_factor
    #     else:
    #         scores /= score_scaling_factor

    # if self.config.score_clip is not None:
    #     # 分数裁剪
    #     scores_dtype = scores.dtype
    #     scores = torch.clip(scores.float(), -self.config.score_clip, self.config.score_clip).to(dtype=scores_dtype)

    # # 如果我们想把最好的模型推送到 hub
    # if hasattr(self, "highest_reward"):
    #     if self.compare_step % self.config.compare_steps == 0:
    #         curr_mean_reward = scores.mean()
    #         # 如果这是有史以来最好的奖励
    #         if curr_mean_reward > self.highest_reward:
    #             self.highest_reward = curr_mean_reward
    #             # 将模型推送到 hub
    #             self.push_to_hub(**self.push_to_hub_kwargs)
    #     self.compare_step += 1

    timing = dict()
    t0 = time.time()

    t = time.time()

    # 连接 query 和 response 以创建 input_ids 张量
    # 同时生成注意力掩码（用于填充）。添加填充是为了让所有的 query+response 可以在同一个张量中连接
    # 包含 input_ids 和 attention_mask 的字典。
    # input_ids 的形状: (batch_size, seq_len)
    # attention_mask 的形状: (batch_size, seq_len)。注意力掩码仅用于屏蔽填充 token。
    model_inputs = self.prepare_model_inputs(queries, responses)

    # if self.is_distributed:
    #     pad_first = self.tokenizer.padding_side == "left"
    #
    #     model_inputs["input_ids"] = self.accelerator.pad_across_processes(
    #         model_inputs["input_ids"],
    #         dim=1,
    #         pad_index=self.tokenizer.pad_token_id,
    #         pad_first=pad_first,
    #     )
    #     model_inputs["attention_mask"] = self.accelerator.pad_across_processes(
    #         model_inputs["attention_mask"], dim=1, pad_index=0, pad_first=pad_first
    #     )
    #     if self.is_encoder_decoder:
    #         model_inputs["decoder_input_ids"] = self.accelerator.pad_across_processes(
    #             model_inputs["decoder_input_ids"],
    #             dim=1,
    #             pad_index=self.tokenizer.pad_token_id,
    #             pad_first=pad_first,
    #         )
    #         model_inputs["decoder_attention_mask"] = self.accelerator.pad_across_processes(
    #             model_inputs["decoder_attention_mask"],
    #             dim=1,
    #             pad_index=0,
    #             pad_first=pad_first,
    #         )
model_inputs_names = list(model_inputs.keys())

        full_kl_penalty = self.config.kl_penalty == "full" # 在我们的案例中，这将是 False。

        # 由于来自离线模型的给定轨迹(trajectories)在每个位置(动作)上都没有对数概率(logprobs)和价值估计(value estimations)，我们需要计算它们。

        with torch.no_grad():
            # 计算每个句子所有 token 的对数概率。
            # masks 指示要使用哪些对数概率（排除 query tokens 和 padding tokens）。
            # all_logprobs: (批次大小, 序列长度 - 1)，其中序列长度是 query+response 的最大长度。
            # values: (批次大小, 序列长度 - 1), masks: (批次大小, 序列长度 - 1)
            all_logprobs, logits_or_none, values, masks = self.batched_forward_pass(
                self.model,
                queries,
                responses,
                model_inputs,
                response_masks=response_masks,
                return_logits=full_kl_penalty,
            )

            with self.optional_peft_ctx():
                # 同时也获取相对于参考模型（冻结模型）的对数概率。
                ref_logprobs, ref_logits_or_none, _, _ = self.batched_forward_pass(
                    self.model if self.is_peft_model else self.ref_model,
                    queries,
                    responses,
                    model_inputs,
                    return_logits=full_kl_penalty,
                )

        timing["time/ppo/forward_pass"] = time.time() - t

        with torch.no_grad():
            t = time.time()
            if full_kl_penalty:
                # === 未使用 === #
                active_full_logprobs = logprobs_from_logits(logits_or_none, None, gather=False)
                ref_full_logprobs = logprobs_from_logits(ref_logits_or_none, None, gather=False)

                rewards, non_score_reward, kls = self.compute_rewards(
                    scores, active_full_logprobs, ref_full_logprobs, masks
                )
            else:
                # 使用分数（来自奖励模型）和对数概率来生成奖励。
                # rewards: (批次大小, 序列长度 - 1)
                rewards, non_score_reward, kls = self.compute_rewards(scores, all_logprobs, ref_logprobs, masks)
            timing["time/ppo/compute_rewards"] = time.time() - t

            t = time.time()
            # 使用奖励和价值，通过 GAE 计算优势(advantage)。
            # values: (批次大小, 序列长度 - 1)
            # rewards: (批次大小, 序列长度-1)
            # returns (Q-values): (批次大小, 序列长度-1)
            values, advantages, returns = self.compute_advantages(values, rewards, masks)
            timing["time/ppo/compute_advantages"] = time.time() - t

        # 这代表了使用旧策略（离线）采样的所有轨迹（我们的轨迹存储）。
        # 向上转换为 float32 以避免数据集问题。
        batch_dict = {
            "queries": queries,
            "responses": responses,
            "logprobs": all_logprobs.to(torch.float32),
            "values": values.to(torch.float32),
            "masks": masks,
            "advantages": advantages,
            "returns": returns,
        }
        batch_dict.update(model_inputs)

        # ======================================
        # 阶段 2：使用 PPO 优化模型
        # ======================================

        t = time.time()
        all_stats = []
        early_stop = False
        for _ in range(self.config.ppo_epochs):
            if early_stop:
                break
            b_inds = np.random.permutation(bs) # 将轨迹随机排序
            for backward_batch_start in range(0, bs, self.config.backward_batch_size):
                backward_batch_end = backward_batch_start + self.config.backward_batch_size
                # 获取要从轨迹存储中检索的项目
                backward_batch_inds = b_inds[backward_batch_start:backward_batch_end]

                # 从从轨迹中提取的宏批次(macro-batch)中提取一个小批次(mini-batch)
                for mini_batch_start in range(0, self.config.backward_batch_size, self.config.mini_batch_size):
                    mini_batch_end = mini_batch_start + self.config.mini_batch_size
                    mini_batch_inds = backward_batch_inds[mini_batch_start:mini_batch_end]


                    # 这是将用于优化模型的采样小批次
                    mini_batch_dict = {
                        "logprobs": batch_dict["logprobs"][mini_batch_inds],
                        "values": batch_dict["values"][mini_batch_inds],
                        "masks": batch_dict["masks"][mini_batch_inds],
                        # 技巧：queries 和 responses 是不规则的（长度不一）。
                        "queries": [batch_dict["queries"][i] for i in mini_batch_inds],
                        "responses": [batch_dict["responses"][i] for i in mini_batch_inds],
                        "advantages": batch_dict["advantages"][mini_batch_inds],
                        "returns": batch_dict["returns"][mini_batch_inds],
                    }

                    for k in model_inputs_names:
                        mini_batch_dict[k] = batch_dict[k][mini_batch_inds]
                    with self.accelerator.accumulate(self.model):
                        model_inputs = {k: mini_batch_dict[k] for k in model_inputs_names}

                        # 计算在线模型（新策略）的对数概率、logits 和价值
                        logprobs, logits, vpreds, _ = self.batched_forward_pass(
                            self.model,
                            mini_batch_dict["queries"],
                            mini_batch_dict["responses"],
                            model_inputs,
                            return_logits=True,
                        )

                        # 使用旧策略的对数概率和新策略的对数概率执行一个训练步骤
                        train_stats = self.train_minibatch(
                            mini_batch_dict["logprobs"],
                            mini_batch_dict["values"],
                            logprobs,
                            logits,
                            vpreds,
                            mini_batch_dict["masks"],
                            mini_batch_dict["advantages"],
                            mini_batch_dict["returns"],
                        )
                        all_stats.append(train_stats)

            # 通常，提早停止是在 epoch 级别完成的
            if self.config.early_stopping:
                policykl = train_stats["policy/policykl"]
                early_stop = self._early_stop(policykl)
                if early_stop:
                    break

        timing["time/ppo/optimize_step"] = time.time() - t

        t = time.time()
        train_stats = stack_dicts(all_stats)

        # 重塑 advantages/ratios，使它们不被平均。
        train_stats["policy/advantages"] = torch.flatten(train_stats["policy/advantages"]).unsqueeze(0)
        train_stats["policy/advantages"] = torch.nan_to_num(train_stats["policy/advantages"], WANDB_PADDING)
        train_stats["policy/ratio"] = torch.flatten(train_stats["policy/ratio"]).unsqueeze(0)

        stats = self.record_step_stats(
            scores=scores,
            logprobs=all_logprobs,
            ref_logprobs=ref_logprobs,
            non_score_reward=non_score_reward,
            train_stats=train_stats,
            kl_coef=self.kl_ctl.value,
            masks=masks,
            queries=queries,
            responses=responses,
            kls=kls,
        )
        # 从所有进程收集/归约(Reduce)统计数据
        if self.is_distributed:
            stats = self.gather_stats(stats)
        stats = stats_to_np(stats)
        timing["time/ppo/calc_stats"] = time.time() - t
        stats["ppo/learning_rate"] = self.optimizer.param_groups[0]["lr"]

        # 更新 KL 控制 - 将批次大小乘以进程数
        self.kl_ctl.update(
            stats["objective/kl"],
            self.config.batch_size * self.accelerator.num_processes,
        )

        # 记录总 ppo 时间
        timing["time/ppo/total"] = time.time() - t0
        stats.update(timing)

        # 为 tensorboard 和其他日志记录器后处理统计数据
        if self.config.log_with != "wandb":
            stats = convert_to_scalar(stats)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return stats

    def _early_stop(self, policykl):
        r"""
        处理提早停止的逻辑。如果策略 KL 大于目标 KL，则将梯度清零并
        跳过优化步骤。
        这也处理了多 GPU 的情况，其中策略 KL 是在所有进程中平均的。

        Args:
            policy_kl (torch.Tensor):
                策略 KL

        Returns:
            `bool`: 是否要提早停止
        """
        early_stop = False
        if not self.config.early_stopping:
            return early_stop

        if not self.is_distributed and policykl > 1.5 * self.config.target_kl:
            self.optimizer.zero_grad()
            early_stop = True
        elif self.is_distributed:
            import torch.distributed as dist

            # 等待所有进程完成
            dist.barrier()

            # 对 policykl 进行 all gather
            dist.all_reduce(policykl, dist.ReduceOp.SUM)
            policykl /= self.accelerator.num_processes

            if policykl > 1.5 * self.config.target_kl:
                self.optimizer.zero_grad()
                early_stop = True
        return early_stop

    def gather_stats(self, stats):
        """
        从所有进程收集统计数据。在分布式训练的上下文中很有用。

        Args:
            stats (dict[str, Any]):
                一个要收集的统计数据字典。统计数据应包含 torch 张量。

        Returns:
            `dict[str, Any]`: 一个包含收集后张量的统计数据字典。
        """
        import torch.distributed as dist

        # 等待所有进程完成
        dist.barrier()

        for k, v in stats.items():
            if isinstance(v, torch.Tensor):
                dist.all_reduce(v.to(self.accelerator.device), dist.ReduceOp.SUM)
                v /= self.accelerator.num_processes
            stats[k] = v
        return stats

    def prepare_model_inputs(self, queries: torch.Tensor, responses: torch.Tensor):
        if self.is_encoder_decoder:
            input_data = self.data_collator(
                [{"input_ids": q, "attention_mask": torch.ones_like(q)} for q in queries]
            ).to(self.current_device)

            decoder_inputs = self.data_collator(
                [{"input_ids": r, "attention_mask": torch.ones_like(r)} for r in responses]
            ).to(self.current_device)

            input_data["decoder_input_ids"] = decoder_inputs["input_ids"]
            input_data["decoder_attention_mask"] = decoder_inputs["attention_mask"]
        else:
            input_ids = [torch.cat([q, r]) for q, r in zip(queries, responses)]
            input_data = self.data_collator(
                [{"input_ids": ids, "attention_mask": torch.ones_like(ids)} for ids in input_ids]
            ).to(self.current_device)

        input_data.pop("labels", None)  # 我们不想计算语言模型损失
        return input_data

    @PPODecorators.empty_device_cache()
    def batched_forward_pass(
        self,
        model: PreTrainedModelWrapper,
        queries: torch.Tensor,
        responses: torch.Tensor,
        model_inputs: dict,
        return_logits: bool = False,
        response_masks: Optional[torch.Tensor] = None,
    ):
        """
        分批计算模型输出。

        Args:
            queries (`torch.LongTensor`):
                包含编码后查询的张量列表，形状 (`batch_size`, `query_length`)
            responses (`torch.LongTensor`):
                包含编码后回应的张量列表，形状 (`batch_size`, `response_length`)
            return_logits (`bool`, *optional*, defaults to `False`):
                是否返回 all_logits。如果不需要 logits，则设置为 `False` 以减少内存消耗。
        Returns:
            (tuple):
                - all_logprobs (`torch.FloatTensor`): 回应的对数概率，
                    形状 (`batch_size`, `response_length`)
                - all_ref_logprobs (`torch.FloatTensor`): 回应的对数概率，
                    形状 (`batch_size`, `response_length`)
                - all_values (`torch.FloatTensor`): 回应的价值，形状 (`batch_size`, `response_length`)
        """
        bs = len(queries)
        fbs = self.config.mini_batch_size
        all_logprobs = []
        all_logits = []
        all_masks = []
        all_values = []

        model.eval()

        # 由于每个批次可能很大，可能无法放入内存，我们通过将批次分成大小为 `fbs` 的较小批次来计算 logits 和对数概率

        for i in range(math.ceil(bs / fbs)):
            # 获​​取当前小批次（大小为 `fbs`）的输入张量

            input_kwargs = {key: value[i * fbs : (i + 1) * fbs] for key, value in model_inputs.items()}
            query_batch = queries[i * fbs : (i + 1) * fbs]
            response_batch = responses[i * fbs : (i + 1) * fbs]
            if response_masks is not None:
                response_masks_batch = response_masks[i * fbs : (i + 1) * fbs]

            # 获取输入中每个 token 对应的 logits 和来自 ValueHead 的相应价值。
            # 输入是查询和回应的串联。
            # logits: (批次, 序列长度, 词汇大小),
            # values: (批次, 序列长度)
            logits, _, values = model(**input_kwargs)

            if self.is_encoder_decoder:
                input_ids = input_kwargs["decoder_input_ids"]
                attention_mask = input_kwargs["decoder_attention_mask"]
            else:
                input_ids = input_kwargs["input_ids"]
                attention_mask = input_kwargs["attention_mask"]

            # 计算每个 token 的对数概率。
            # 这可以通过模型为每个 token 输出的 logits 获得（并应用 softmax）。
            # logits: (批次大小, 序列长度 - 1)
            logprobs = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])

            masks = torch.zeros_like(attention_mask)
            masks[:, :-1] = attention_mask[:, 1:] # 指示我们拥有对数概率的 token

            for j in range(len(query_batch)):
                if self.is_encoder_decoder:
                    # 在编码器-解码器模型中，解码器句子总是在填充后的索引 1 开始
                    start = 1
                    end = attention_mask[j, :].sum() - 1
                else:
                    # logprobs 从第一个回应 token 开始
                    start = len(query_batch[j]) - 1
                    if attention_mask[j, 0] == 0:  # 偏移左侧填充
                        start += attention_mask[j, :].nonzero()[0]
                    # 对应于整个（查询+回应）序列中结束位置的索引
                    end = start + len(response_batch[j])
                    if response_masks is not None:
                        response_masks_batch[j] = torch.cat(
                            (torch.zeros_like(query_batch[j]), response_masks_batch[j])
                        )[1:]

                # 所有我们没有对数概率的 token 都被遮蔽掉
                # 遮蔽掉第一个回应 token 之前的任何 token（即遮蔽掉提示 token）
                masks[j, :start] = 0
                # 遮蔽掉回应 token 之后的任何 token（即遮蔽掉任何填充 token）
                masks[j, end:] = 0

                if response_masks is not None:
                    masks[j, start:end] = masks[j, start:end] * response_masks_batch[j][start:end]

            if return_logits:
                all_logits.append(logits)
            else:
                del logits
            all_values.append(values)
            all_logprobs.append(logprobs)
            all_masks.append(masks)

        return (
            torch.cat(all_logprobs),
            torch.cat(all_logits)[:, :-1] if return_logits else None,
            torch.cat(all_values)[:, :-1],
            torch.cat(all_masks)[:, :-1],
        )

    @PPODecorators.empty_device_cache()
    def train_minibatch(
        self,
        old_logprobs: torch.FloatTensor, # 旧策略（离线）下的对数概率
        values: torch.FloatTensor, # 旧策略（离线）下的价值
        logprobs: torch.FloatTensor, # 新策略（在线）下的对数概率
        logits: torch.FloatTensor, # 新策略（在线）下的 logits
        vpreds: torch.FloatTensor, # 新策略（在线）下的价值
        mask: torch.LongTensor, # 指示对数概率对应于哪些 token
        advantages: torch.FloatTensor, # 在旧策略（离线）下计算的优势
        returns: torch.FloatTensor, # 在旧策略（离线）下计算的状态-动作值（Q-values）
    ):
        """
        训练一个 PPO 小批次

        Args:
            logprobs (`torch.FloatTensor`):
                模型的对数概率，形状 [mini_batch_size, response_length]
            values (`torch.FloatTensor`):
                价值头的价值，形状 [mini_batch_size, response_length]
            query (`torch.LongTensor`):
                编码后的查询，形状 [mini_batch_size, query_length]
            response (`torch.LongTensor`):
                编码后的回应，形状 [mini_batch_size, response_length]
            model_input (`torch.LongTensor`):
                串联的查询和回应，形状 [mini_batch_size, query_length+response_length]

        Returns:
            train_stats (dict[str, `torch.Tensor`]):
                训练统计数据的字典
        """
        self.model.train()
        loss_p, loss_v, train_stats = self.loss(
            old_logprobs, values, logits, vpreds, logprobs, mask, advantages, returns
        )
        loss = loss_p + loss_v # 损失是策略梯度损失和价值损失的总和
        self.accelerator.backward(loss)
        if self.config.max_grad_norm is not None:
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.model_params, self.config.max_grad_norm)
        self.optimizer.step()
        # 我们每次都调用 optimizer.zero_grad()，让 `accelerator` 处理梯度累积
        # 参考 https://huggingface.co/docs/accelerate/usage_guides/gradient_accumulation#the-finished-code
        self.optimizer.zero_grad()
        return train_stats

    def compute_rewards(
        self,
        scores: torch.FloatTensor,
        logprobs: torch.FloatTensor,
        ref_logprobs: torch.FloatTensor,
        masks: torch.LongTensor,
    ):
        """
        从分数和 KL 惩罚计算每个 token 的奖励。

        Args:
            scores (`torch.FloatTensor`):
                来自奖励模型的分数，形状 (`batch_size`)
            logprobs (`torch.FloatTensor`):
                模型的对数概率，形状 (`batch_size`, `response_length`)
            ref_logprobs (`torch.FloatTensor`):
                参考模型的对数概率，形状 (`batch_size`, `response_length`)

        Returns:
            `torch.FloatTensor`: 每个 token 的奖励，形状 (`batch_size`, `response_length`)
            `torch.FloatTensor`: 非分数奖励，形状 (`batch_size`, `response_length`)
            `torch.FloatTensor`: KL 惩罚，形状 (`batch_size`, `response_length`)
        """
        rewards, non_score_rewards, kls = [], [], []
        for score, logprob, ref_logprob, mask in zip(scores, logprobs, ref_logprobs, masks):
            # 计算 KL 惩罚（来自对数概率的差异）
            # 形状: (Seq_Len) - 表示每个 token 的对数概率差异（冻结模型 vs 微调模型）
            kl = self._kl_penalty(logprob, ref_logprob)
            kls.append(kl)
            non_score_reward = -self.kl_ctl.value * kl
            non_score_rewards.append(non_score_reward)
            reward = non_score_reward.clone()
            last_non_masked_index = mask.nonzero()[-1]

            # 奖励最初用 -KL 惩罚初始化。然后我们只将奖励模型给出的分数加到回应的最后一个生成 token 上
            # 基本上，我们是用 KL 惩罚来惩罚奖励模型给出的奖励（回应与冻结模型的差异程度）
            # 形状: (Seq_Len)
            reward[last_non_masked_index] += score
            rewards.append(reward)
        return torch.stack(rewards), torch.stack(non_score_rewards), torch.stack(kls)

    def _kl_penalty(self, logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor) -> torch.FloatTensor:
        if self.config.kl_penalty == "kl":
            return logprob - ref_logprob

        if self.config.kl_penalty == "abs":
            return (logprob - ref_logprob).abs()

        if self.config.kl_penalty == "mse":
            return 0.5 * (logprob - ref_logprob).square()

        if self.config.kl_penalty == "full":
            # 由于这个问题需要翻转？:https://github.com/pytorch/pytorch/issues/57459
            return F.kl_div(ref_logprob, logprob, log_target=True, reduction="none").sum(-1)

        raise NotImplementedError

    def compute_advantages(
        self,
        values: torch.FloatTensor,
        rewards: torch.FloatTensor,
        mask: torch.FloatTensor,
    ):
        lastgaelam = 0
        advantages_reversed = []
        gen_len = rewards.shape[-1]

        values = values * mask
        rewards = rewards * mask

        # if self.config.whiten_rewards:
        #     rewards = masked_whiten(rewards, mask, shift_mean=False)

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0 # 在时间 (t+1) 评估的价值函数
            delta = rewards[:, t] + self.config.gamma * nextvalues - values[:, t] # 根据 GAE 公式: delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
            lastgaelam = delta + self.config.gamma * self.config.lam * lastgaelam # 保存 GAE 以供下次迭代使用
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1]).transpose(0, 1) # 反转优势并将它们堆叠起来

        returns = advantages + values # 因为 Advantage = Q - V，我们可以计算 Q = Advantage + V。Q 值对于训练价值函数估计是必要的。
        advantages = masked_whiten(advantages, mask)
        advantages = advantages.detach()
        return values, advantages, returns

    def loss(
        self,
        old_logprobs: torch.FloatTensor, # 旧策略（离线）下的对数概率
        values: torch.FloatTensor, # 旧策略（离线）下的价值
        logits: torch.FloatTensor, # 新策略（在线）下的 logits
        vpreds: torch.FloatTensor, # 新策略（在线）下的价值
        logprobs: torch.FloatTensor, # 新策略（在线）下的对数概率
        mask: torch.LongTensor, # 对数概率对应于哪些 token
        advantages: torch.FloatTensor, # 使用旧策略（离线）计算的优势
        returns: torch.FloatTensor, # 使用旧策略（离线）计算的狀態-動作值（Q-values）
    ):
        """
        计算策略和价值损失。

        Args:
            old_logprobs (`torch.FloatTensor`):
                模型的对数概率，形状 (`batch_size`, `response_length`)
            values (`torch.FloatTensor`):
                价值头的价值，形状 (`batch_size`, `response_length`)
            rewards (`torch.FloatTensor`):
                来自奖励模型的奖励，形状 (`batch_size`, `response_length`)
            logits (`torch.FloatTensor`):
                模型的 logits，形状 (`batch_size`, `response_length`, `vocab_size`)
            v_pred (`torch.FloatTensor`):
                价值头的价值，形状 (`batch_size`, `response_length`)
            logprobs (`torch.FloatTensor`):
                模型的对数概率，形状 (`batch_size`, `response_length`)
        """

        vpredclipped = clip_by_value(
            vpreds,
            values - self.config.cliprange_value,
            values + self.config.cliprange_value,
        )

        # 价值头的损失
        vf_losses1 = (vpreds - returns) ** 2 # 这是根据幻灯片中公式的损失。(V(s) - Q(s, a))^2
        vf_losses2 = (vpredclipped - returns) ** 2
        vf_loss = 0.5 * masked_mean(torch.max(vf_losses1, vf_losses2), mask)
        vf_clipfrac = masked_mean(torch.gt(vf_losses2, vf_losses1).float(), mask)

        # 新策略和旧策略的对数概率之比
        ratio = torch.exp(logprobs - old_logprobs)

        # 负号是因为我们想要最大化目标函数，但优化器是最小化损失
        pg_losses = -advantages * ratio # 根据公式，对数概率的比率乘以优势
        pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - self.config.cliprange, 1.0 + self.config.cliprange)

        # 使用 "max" 而不是 "min" 是因为我们想要最大化目标函数，但优化器是最小化损失
        pg_loss = masked_mean(torch.max(pg_losses, pg_losses2), mask) # 策略梯度损失
        pg_clipfrac = masked_mean(torch.gt(pg_losses2, pg_losses).float(), mask)

        loss = pg_loss + self.config.vf_coef * vf_loss

        avg_ratio = masked_mean(ratio, mask).item()
        if avg_ratio > self.config.ratio_threshold:
            warnings.warn(
                f"批次的平均比率 ({avg_ratio:.2f}) 超过阈值 {self.config.ratio_threshold:.2f}。跳过此批次。"
            )
            pg_loss = pg_loss * 0.0
            vf_loss = vf_loss * 0.0
            loss = loss * 0.0
        # 熵，用来强制模型进行探索
        entropy = masked_mean(entropy_from_logits(logits), mask)

        approxkl = 0.5 * masked_mean((logprobs - old_logprobs) ** 2, mask)
        policykl = masked_mean(old_logprobs - logprobs, mask)

        return_mean, return_var = masked_mean(returns, mask), masked_var(returns, mask)
        value_mean, value_var = masked_mean(values, mask), masked_var(values, mask)

        stats = dict(
            loss=dict(policy=pg_loss.detach(), value=vf_loss.detach(), total=loss.detach()),
            policy=dict(
                entropy=entropy.detach(),
                approxkl=approxkl.detach(),
                policykl=policykl.detach(),
                clipfrac=pg_clipfrac.detach(), # 策略梯度的裁剪分数
                advantages=advantages.detach(), # 优势函数
                advantages_mean=masked_mean(advantages, mask).detach(), # 掩码后的优势函数均值
                ratio=ratio.detach(), # 策略比率
            ),
            returns=dict(mean=return_mean.detach(), var=return_var.detach()), # 返回值的均值和方差
            val=dict(
                vpred=masked_mean(vpreds, mask).detach(), # 掩码后的价值预测
                error=masked_mean((vpreds - returns) ** 2, mask).detach(), # 掩码后的价值预测误差（均方误差）
                clipfrac=vf_clipfrac.detach(), # 价值函数的裁剪分数
                mean=value_mean.detach(), # 价值的均值
                var=value_var.detach(), # 价值的方差
            ),
        )
        return pg_loss, self.config.vf_coef * vf_loss, flatten_dict(stats)

    def record_step_stats(self, kl_coef: float, **data):
        """
        记录训练步骤统计数据。

        Args:
            kl_coef (`float`):
                KL散度系数
            data (`dict`):
                训练步骤数据的字典

        Returns:
            stats (`dict`):
                训练步骤统计数据的字典
        """
        mask = data.pop("masks") # 弹出掩码数据

        kls = data.pop("kls") # 弹出KL散度数据
        kl_list = ((kls) * mask).sum(axis=-1) # 计算掩码后的KL散度列表
        mean_kl = kl_list.mean() # 计算KL散度的均值
        mean_entropy = (-data["logprobs"] * mask).sum(axis=-1).mean() # 计算掩码后的熵的均值

        mean_non_score_reward = masked_mean(
            data["non_score_reward"], mask
        )  # non_score_reward 的大小为 `batch_size`, `response_length`，计算非分数奖励的均值
        mean_scores = data["scores"].mean()  # scores 的大小为 `batch_size`，计算分数的均值
        std_scores = data["scores"].std() # 计算分数的标准差

        if mean_kl.item() < -1.0:
            # 警告用户
            warnings.warn(
                f"KL散度开始变为负值: {mean_kl.item():.2f} - 这可能是训练失败的前兆。"
                "有时这发生是因为生成 kwargs 设置不正确。请确保"
                "生成 kwargs 设置正确，或检查您的训练超参数。"
            )

        stats = {
            "objective/kl": mean_kl, # 目标/KL散度
            "objective/kl_dist": kl_list, # 目标/KL散度分布
            "objective/logprobs": data["logprobs"], # 目标/对数概率
            "objective/ref_logprobs": data["ref_logprobs"], # 目标/参考对数概率
            "objective/kl_coef": kl_coef, # 目标/KL散度系数
            "objective/entropy": mean_entropy, # 目标/熵
            "ppo/mean_non_score_reward": mean_non_score_reward, # PPO/非分数奖励均值
            "ppo/mean_scores": mean_scores, # PPO/分数均值
            "ppo/std_scores": std_scores, # PPO/分数标准差
        }

        # 记录文本属性
        query_lens = torch.tensor([len(query) for query in data["queries"]], dtype=torch.float) # 查询长度
        response_lens = torch.tensor([len(response) for response in data["responses"]], dtype=torch.float) # 响应长度

        stats["tokens/queries_len_mean"] = torch.mean(query_lens).cpu().numpy().item() # token/查询长度均值
        stats["tokens/queries_len_std"] = torch.std(query_lens).cpu().numpy().item() # token/查询长度标准差
        stats["tokens/queries_dist"] = query_lens.cpu().numpy() # token/查询长度分布
        stats["tokens/responses_len_mean"] = torch.mean(response_lens).cpu().numpy().item() # token/响应长度均值
        stats["tokens/responses_len_std"] = torch.std(response_lens).cpu().numpy().item() # token/响应长度标准差
        stats["tokens/responses_dist"] = response_lens.cpu().numpy() # token/响应长度分布

        for k, v in data["train_stats"].items():
            stats[f"ppo/{k}"] = torch.mean(v, axis=0) # PPO/训练统计数据
        stats["ppo/val/var_explained"] = 1 - stats["ppo/val/error"] / stats["ppo/returns/var"] # PPO/价值/解释方差
        return stats

    def log_stats(
        self,
        stats: dict,
        batch: dict,
        rewards: List[torch.FloatTensor],
        columns_to_log: typing.Iterable[str] = ("query", "response"),
    ):
        """
        一个记录所有训练统计数据的函数。在每个 epoch 结束时调用。

        Args:
            stats (dict[str, Any]):
                训练统计数据的字典。
            batch (dict[str, Any]):
                批处理数据的字典，包含查询和响应。
            rewards (`List[torch.FloatTensor]`):
                奖励张量。
        """

        # 收集所有统计数据
        if not isinstance(rewards, torch.Tensor):
            rewards = torch.tensor(rewards).to(self.current_device) # 将奖励转换为张量并移动到当前设备
        rewards = self.accelerator.gather(rewards).flatten() # 收集奖励并展平

        if self.config.log_with == "wandb":
            import wandb

            if any(column_to_log not in batch.keys() for column_to_log in columns_to_log):
                raise ValueError(f"要记录的列 {columns_to_log} 不存在于批处理 {batch.keys()} 中。")

            batch_list = [batch[column_to_log] for column_to_log in columns_to_log] # 提取要记录的列
            if self.is_distributed:
                gathered_batch_list = []
                for b in batch_list:
                    flattened = gather_object(b) # 收集对象并展平
                    gathered_batch_list.append(flattened)
                batch_list = gathered_batch_list

        # 仅在主进程中记录
        if self.accelerator.is_main_process:
            logs = {}

            # 记录统计数据
            if "query" not in batch.keys() and "response" not in batch.keys():
                # 警告用户，游戏日志将不会被记录
                warnings.warn(
                    "游戏日志将不会被记录，因为批处理不包含 'query' 和 'response' 键。 "
                )
            elif self.config.log_with == "wandb":
                table_rows = [list(r) for r in zip(*batch_list, rewards.cpu().tolist())] # 构建表格行
                logs.update({"game_log": wandb.Table(columns=[*columns_to_log, "reward"], rows=table_rows)}) # 更新游戏日志

            logs.update(stats) # 更新统计数据

            # 对于 bf16 torch 张量手动转换为 fp32
            for k, v in logs.items():
                if isinstance(v, torch.Tensor) and v.dtype == torch.bfloat16:
                    logs[k] = v.float()

            logs["env/reward_mean"] = torch.mean(rewards).cpu().numpy().item() # 环境/奖励均值
            logs["env/reward_std"] = torch.std(rewards).cpu().numpy().item() # 环境/奖励标准差
            logs["env/reward_dist"] = rewards.cpu().numpy() # 环境/奖励分布

            if self.config.log_with == "tensorboard":
                # 更新当前步骤
                self.current_step += 1

            self.accelerator.log(
                logs,
                step=self.current_step if self.config.log_with == "tensorboard" else None,
            )

    def create_model_card(self, path: str, model_name: Optional[str] = "TRL Model") -> None:
        """为 TRL 模型创建并保存模型卡片。

        Args:
            path (`str`): 保存模型卡片的路径。
            model_name (`str`, *optional*): 模型的名称，默认为 `TRL Model`。
        """
        try:
            user = whoami()["name"] # 获取用户信息
        # 处理离线情况
        except Exception:
            warnings.warn("无法检索用户信息，假设您正在离线模式下运行。")
            return

        if not os.path.exists(path):
            os.makedirs(path) # 创建目录

        model_card_content = MODEL_CARD_TEMPLATE.format(model_name=model_name, model_id=f"{user}/{path}") # 格式化模型卡片内容
        with open(os.path.join(path, "README.md"), "w", encoding="utf-8") as f:
            f.write(model_card_content) # 写入模型卡片内容

    def _save_pretrained(self, save_directory: str) -> None:
        self.accelerator.unwrap_model(self.model).save_pretrained(save_directory) # 保存预训练模型
        self.tokenizer.save_pretrained(save_directory) # 保存分词器
        self.create_model_card(save_directory) # 创建模型卡片

    def _show_tokens(self, tokens, masks):
        from rich import print
        from rich.text import Text

        text = Text()

        for _i, (token, mask) in enumerate(zip(tokens, masks)):
            if mask == 1:
                text.append(self.tokenizer.decode(token.item()), style="black on deep_sky_blue1") # 如果掩码为1，使用蓝色背景显示
                text.append(" ")
            else:
                text.append(self.tokenizer.decode(token.item()), style="black on cyan3") # 否则使用青色背景显示
                text.append(" ")
        print(text)

    def _prepare_deepspeed(self, model: PreTrainedModelWrapper):
        # 改编自 accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = deepspeed_plugin.deepspeed_config # 获取 DeepSpeed 配置
        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                    # 请注意，`stage3_prefetch_bucket_size` 可能会产生 DeepSpeed 消息，例如: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                    # 这是预期行为，并非错误，请参阅: https://github.com/microsoft/DeepSpeed/discussions/4081
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size * hidden_size, # 归约桶大小
                            "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size, # 第三阶段参数持久化阈值
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size, # 第三阶段预取桶大小
                        }
                    )

        # 如果使用 ZeRO-3，我们分片活动模型和参考模型。
        # 否则，我们假设参考模型适合内存并在每个设备上以禁用 ZeRO（stage 0）的方式初始化
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0 # 如果不是第三阶段优化，则设置为0
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs) # 初始化 DeepSpeed 模型
        model.eval() # 将模型设置为评估模式
        return model
