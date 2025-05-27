print(111)
import sys
import os
env_name = os.environ.get("CONDA_DEFAULT_ENV", "未在 Conda 环境中运行")
print(f"当前 Conda 环境：{env_name}")
print(sys.version)      # 输出完整版本及编译信息，如 "3.9.7 (default, Aug 31 2021, ...)" [1,3,4](@ref)
print(sys.version_info) 
print(os.environ)

from transformers import AutoTokenizer, AutoModel
import peft 
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoTokenizer, AutoModel

#model_path = "/data/zhangyanhong-2401220256/finetune/gpt2-1.4M"  # 本地模型目录路径
model_path = "/data/zhangyanhong-2401220256/finetune/gpt2-1.4M"  # 本地模型目录路径

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 加载模型（支持PyTorch）
model = AutoModel.from_pretrained(model_path)

lora_config = LoraConfig(
    r=4, #As bigger the R bigger the parameters to train.
    lora_alpha=1, # a scaling factor that adjusts the magnitude of the weight matrix. Usually set to 1
    target_modules=["query_key_value"], #You can obtain a list of target modules in the URL above.
    lora_dropout=0.05, #Helps to avoid Overfitting.
    bias="lora_only", # this specifies if the bias parameter should be trained.
    task_type="CAUSAL_LM"
)
print(lora_config)