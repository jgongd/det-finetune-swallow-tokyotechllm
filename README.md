# det-finetuning-swallow-tokyotech-llm
Finetune Swallow - LLM models by Tokyo Institute of Technology - using HPE Machine Learning Development Environment (MLDE) callback with Hugging Face Trainer API to enable MLDE's distributed training, fault tolerance, checkpointing and metrics reporting

# Why Swallow?

- Overview of Swallow
Swallow is an LLM developed by researchers at Tokyo Tech with enhanced Japanese capability, by extending the vocabulary of Llama 2 to include Japanese characters and continual pre-training on a large Japanese web corpus.
The performance of Swallow increased with the amount of training data up to 100B tokens through continual pre-training, and Swallow achieved competitive performance compared to other LLMs trained on English and Japanese datasets from scractch. 

Swallow paper: [Continual Pre-Training for Cross-Lingual LLM Adaptation: Enhancing Japanese Language Capabilities](https://arxiv.org/pdf/2404.17790)


- Why instruction tuning? 
Instruction finetuning is an efficient technique to enhance the capabilities and controllability of LLMs, addressing the issue of mismatch between the training objective and users' objective: LLMs are typically trained on minimizing the contextual next token generation (aka next word prediction); while users prefer LLMs to follow predefined instruction templates and return proper and safe answers in a controllable manner. In this demo, Swallow 70b instruct-v0.1 is used.

Model card on HF: [tokyotech-llm/Swallow-70b-instruct-v0.1](https://huggingface.co/tokyotech-llm/Swallow-70b-instruct-v0.1)

## Why HPE MLDE for finetuning LLMs at scale?

- Distributed training
With Determined, to scale an experiment/trial to multiple GPUs requires a single configuration line change. There is no need to worry about setting up frameworks like Horovod or PyTorch Distributed Data Parallel (DDP), or Pytorch Lightning.

<img src="imgs/dtrain.png" alt="Distributed Training" width="600">

- Hyperparameter search using Adaptive ASHA
To accelerate a search process, HPE MLDE leverages the Adaptive ASHA algorithm in order to find the best set of parameters in the hyperparameter space. 
The idea behind Adaptive ASHA is that we’ll run all the different model variations with different sets of hyperparameters in parallel, then we’ll stop the ones that are not performing well early and continue to train the ones that are performing well until convergence. 

<img src="imgs/hpsearch.png" alt="Distributed Hyperparameter Search" width="900">

- DeepSpeed integration
DeepSpeed API is a lightweight wrapper on PyTorch for training and inference of hyperscale DL models, ex., trillion parameter LLMs. DeepSpeed manages the boilerplate state-of-the-art training techniques, such as distributed training, mixed precision, gradient accumulation, and checkpoints so that users can focus on model development. DeepSpeed make training those models efficiently on 100s or 1000s of GPUs using techniques such as Zero Redundancy Optimizer (ZeRO), 3D parallelism that include data, model parallelism, and pipeline parallelism, and ZeRO-Infinity. 

In this demo, we'll show you how you can leverage DeepSpeed ZeRO stage 3 for finetuning Swallow on MLDE. DeepSpeed ZeRO stage 3 includes all optimizer state partitioning, gradient partitioning, and model parameter partitioning. 

<img src="imgs/deepspeed-zero.png" alt="DeepSpeed-ZeRO" width="500">

Source: [ZeRO & DeepSpeed: New system optimizations enable training models with over 100 billion parameters](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/)

## How MLDE works with HF Trainer

While the BERT family models including ALBERT, BERT, DistilBERT, and RoBERTA are trained or finetuned using a masked language modelling (MLM) loss, later pretrained transformer based models including GPT, GPT-2, Llama, Mistral, Falcon, Phi, etc are trained or finetunde using a causal language modeling (CLM) loss. Find more information about the differences between those objectives in [this Transformer model summary](https://huggingface.co/docs/transformers/model_summary).

- HuggingFace Trainer's original *.py is located at /scripts/run_clm_hf.py

- Integration between HF Trainer and MLDE via MLDE callback

The main callback is located in determined.transformers and the associated DetCallback object is used in model code as in:
hf_callback.py
```bash
det_callback = DetCallback(training_args,
                            filter_metrics=["loss", "accuracy"],
                            tokenizer=feature_extractor)
trainer.add_callback(det_callback)
```
## Major changes in the code for integrating DetCallback

- Import Determined, Determined's DeepSpeed Auto Tuner (DSAT), and Determined Callback

```bash
import determined as det
from determined.pytorch import dsat
from hf_callback import DetCallback
```

- ModelArguments Class: No major changes. We can add --use_lora

```bash
use_lora: bool = field(
        default=False,
        metadata={
            "help": "Whether to use preconfigured LoRA for parameter efficient finetuning via peft library."
```

- DataTrainingArguments Class: No major changes.

- def main() function: det_callback, tb_callback, model_args, data_agrs, and training_args are passed to the main() function.

```bash
def main(det_callback, tb_callback, model_args, data_args, training_args):
```

- DeepSpeed Autotune (DSAT):
[DSAT](https://hpe-mlde.determined.ai/latest/model-dev-guide/api-guides/apis-howto/deepspeed/autotuning.html#deepspeed-autotune-user-guide) helps users optimize setting many DS parameters on specific properties of hardware and model. This is done through and easy-to-use API with very few changes required in user-code. DSAT can be used with HuggingFace Trainer in this situation.

```bash
with dsat.dsat_reporting_context(core_context, op=det_callback.current_op):
            train_result = trainer.train(resume_from_checkpoint=checkpoint)
```
- Executive script under if __name__ == "__main__":

```bash
if __name__ == "__main__":
    info = det.get_cluster_info()
    assert info
    hparams = info.trial.hparams
    model_args, data_args, training_args = parse_input_arguments(hparams)
    if training_args.deepspeed:
        distributed = det.core.DistributedContext.from_deepspeed()
    else:
        distributed = det.core.DistributedContext.from_torch_distributed()

    with det.core.init(distributed=distributed) as core_context:
        user_data = {
            "finetuned_from": model_args.model_name_or_path,
            "tasks": "language-modeling",
            "dataset": data_args.dataset_name,
            "tags": ["language-modeling", "nlp"],
        }

        det_callback = DetCallback(core_context, training_args, user_data=user_data)

        tb_callback = TensorBoardCallback(
            tb_writer=SummaryWriter(core_context.train.get_tensorboard_path())
        )
        main(det_callback, tb_callback, model_args, data_args, training_args)
```
- Parsing arguments
```bash
def dict2args(hparams):
    out = []
    for key in hparams:
        out.append("--" + str(key))
        out.append(str(hparams[key]))
    return out


def parse_input_arguments(
    hparams: Dict[str, Any]
) -> Tuple[ModelArguments, DataTrainingArguments, TrainingArguments]:
    training_arguments = hparams.get("training_arguments", {})
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        args = sys.argv[1:]
        args.extend(dict2args(training_arguments))
        if any("--deepspeed" == arg.strip() for arg in args):
            args = dsat.get_hf_args_with_overwrites(args, hparams)
        model_args, data_args, training_args = parser.parse_args_into_dataclasses(
            args, look_for_args_file=False
        )
    return model_args, data_args, training_args
```
## Configuration file for running MLDE on DeepSpeed

swallow_70b_ds3.yaml
```bash
name: swallow_70b_ds3
debug: false
environment:
  image: determinedai/genai-train:latest
  environment_variables:
    - NCCL_DEBUG=INFO
    - HF_HOME=/hf_cache
    - NCCL_SOCKET_IFNAME=ens,eth,ib
resources:
  slots_per_trial: 4
  resource_pool: A100
searcher:
  name: single
  max_length:
    batches: 100
  metric: eval_loss
hyperparameters:
  deepspeed_config: ds_configs/ds_config_stage_3.json
  training_arguments:
    learning_rate: 1e-5
entrypoint: >-
  python -m determined.launch.deepspeed
  python run_clm.py
  --model_name_or_path tokyotech-llm/Swallow-70b-instruct-v0.1
  --dataset_name camel-ai/math
  --dataset_config_name default
  --do_train
  --do_eval
  --use_lora
  --torch_dtype float16
  --max_steps 100  
  --logging_strategy steps
  --logging_steps 10
  --output_dir /tmp/test-clm
  --eval_steps 10
  --evaluation_strategy steps
  --save_total_limit 1
  --seed 1337
  --save_strategy steps
  --save_steps 20
  --deepspeed ds_configs/ds_config_stage_3.json
  --per_device_train_batch_size 1
  --per_device_eval_batch_size 1
  --trust_remote_code false
  --fp16
max_restarts: 0
workspace: "poc"
project: "swallow"
```
DeepSpeed Configurations

ds_config_stage_3.json
```bash
{
  "fp16": {
    "enabled": "auto",
    "loss_scale": 0,
    "loss_scale_window": 1000,
    "initial_scale_power": 16,
    "hysteresis": 2,
    "min_loss_scale": 1
  },
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": "auto",
      "betas": "auto",
      "eps": "auto",
      "weight_decay": "auto"
    }s
  },
  "scheduler": {
    "type": "WarmupLR",
    "params": {
      "warmup_min_lr": "auto",
      "warmup_max_lr": "auto",
      "warmup_num_steps": "auto"
    }
  },
  "zero_optimization": {
    "stage": 3,
    "overlap_comm": true,
    "contiguous_gradients": true,
    "sub_group_size": 1e9,
    "reduce_bucket_size": 5e8,
    "stage3_prefetch_bucket_size": 5e7,
    "stage3_param_persistence_threshold": 1e5,
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "stage3_gather_16bit_weights_on_model_save": true
  },
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": "auto",
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "flops_profiler": {
    "enabled": true,
    "profile_step": 1,
    "module_depth": -1,
    "top_modules": 1,
    "detailed": true,
    "output_file": null
  }
}
```
- Package installation at start up 

requirements.txt
```yaml
setuptools==59.5.0
accelerate>=0.12.0
tokenizers>=0.13.3
datasets==2.18.0
evaluate==0.4.1
peft==0.10.0 
rouge_score==0.1.2
fire==0.6.0 
sentencepiece
scikit-learn
```
startup-hook.sh
```yaml
pip install -r requirements.txt
```

### How to launch a notebook in MLDE


```bash
# Go back one level from det_files/ to home directory
cd ..
det -m <master_address>:8080/ notebook start --config-file notebook.yaml -c .
```

When Jupyter Lab is launched, open `Finetune Swallow 70B.ipynb` and start interacting with the original and finetuned Swallow 70B models. 


## Appendix

### Troubleshoot issues

AttributeError: 'LlamaTokenizerFast' object has no attribute 'apply_chat_template'
(https://github.com/huggingface/transformers/issues/27078)

RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cuda:1!
Remove `device_map=auto`
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)


[2024-05-03 10:26:43] [bc31b424] [rank=0] [2024-05-03 10:26:43,811] [WARNING] [cpu_adam.py:84:__init__] FP16 params for CPUAdam may not work on AMD CPUs
<none> [2024-05-03 10:26:43] [bc31b424] [rank=0] [exp-12000-trial-61394-0-12000:741  :0:741] Caught signal 4 (Illegal instruction: illegal operand)

Options to define train length:
- num_train_epochs
- max_steps, which will override num_train_epochs
Ref: https://discuss.huggingface.co/t/is-it-possible-to-set-epoch-less-than-1-when-using-trainer/19311

Note: 
- max_length in MLDE override, hence need to set up # of epochs on MLDE side
max_length:
    batches: 100

-------------------------------------------------------------------------------------------------------
Questions: 
1. How an epoch is calculated? How to set up number of epochs in MLDE for HF Trainer?
2. Why finetuning Swallow on a math dataset?
3. How PEFT model is loaded? Why is it taking time?  
4. What's a quick way to test finetuned model on a 
-------------------------------------------------------------------------------------------------------

HTTPStatusError                           Traceback (most recent call last)
File /lore/utils/request_client.py:35, in RequestClient._raise_for_status(self, response)
     34 try:
---> 35     response.raise_for_status()
     36 except httpx.HTTPStatusError as e:

File /usr/local/lib/python3.10/dist-packages/httpx/_models.py:759, in Response.raise_for_status(self)
    758 message = message.format(self, error_type=error_type)
--> 759 raise HTTPStatusError(message, request=request, response=self)

HTTPStatusError: Client error '400 Bad Request' for url 'http://genai.demo.determined.ai:8080/genai/api/v1/dataset'
For more information check: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/400

The above exception was the direct cause of the following exception:

RequestError                              Traceback (most recent call last)
Cell In[38], line 2
      1 # exp_ft = fine_tune(model, dataset, prompt_template_ft, os.environ["FT_NUM_GPUS"], os.environ["FT_RESOURCE_POOL"])
----> 2 exp_ft = fine_tune(model, dataset, prompt_template_ft, FT_NUM_GPUS, FT_RESOURCE_POOL)

Cell In[32], line 25, in fine_tune(model, dataset, prompt_template, num_gpus, resource_pool, name, is_custom_model)
      9     custom_model_kwargs = {
     10         "max_steps": 10,
     11         "num_train_epochs": None,
   (...)
     22         "gradient_checkpointing": True
     23     }
     24     #############################
---> 25 return lore.launch_training(
     26     name=name, # Name of our fine-tuning job
     27     dataset=dataset, # Dataset to fine-tune on
     28     base_model=model, # Model to fine-tune
     29     prompt_template=prompt_template, # The prompt template we defined before
     30     resource_pool=resource_pool, #  We're running fine-tuning on the defined resource pool...
     31     slots_per_trial=num_gpus, # ...using a set of GPUs...
     32     deepspeed=True, # ...in a model-parallel way (DeepSpeed)
     33     **custom_model_kwargs
     34 )

File /lore/client.py:1204, in Lore.launch_training(self, dataset, base_model, name, prompt_template, hf_token, torch_dtype, low_cpu_mem_usage, trust_remote_code, slots_per_trial, max_steps, num_train_epochs, save_total_limit, learning_rate, per_device_train_batch_size, per_device_eval_batch_size, do_train, do_eval, block_size, deepspeed, gradient_checkpointing, ds_cpu_offload, peft_config, seed, output_dir, eval_steps, logging_steps, save_steps, logging_strategy, evaluation_strategy, save_strategy, resource_pool, disable_training_config_recommendation)
   1202 # Register dataset if needed
   1203 if dataset.id is None:
-> 1204     dataset = self.commit_dataset(dataset)
   1206 # Try get training config recommendation
   1207 try_recommend_training_config = self._try_recommend_training_config(
   1208     base_model=base_model,
   1209     peft_config=peft_config,
   1210     disable_training_config_recommendation=disable_training_config_recommendation,
   1211 )

File /lore/client.py:266, in Lore.commit_dataset(self, dataset)
    265 def commit_dataset(self, dataset: bt.Dataset) -> bt.Dataset:
--> 266     return self._post(f"{self._prefix}/dataset", in_data=dataset, out_cls=bt.Dataset)

File /lore/utils/request_client.py:126, in RequestClient._post(self, route, in_data, **kwargs)
    125 def _post(self, route: str, in_data: Optional[BaseModel] = None, **kwargs) -> Any:
--> 126     return self._request("POST", route, in_data=in_data, **kwargs)

File /lore/utils/request_client.py:123, in RequestClient._request(self, method, route, in_data, out_cls, files, params, seq, is_async, skip_await, timeout, token, poll_interval, public)
    121     out_cls = bt.AsyncResponse
    122     seq = False
--> 123 return getter_fn(response, cls=out_cls, seq=seq)

File /lore/utils/request_client.py:45, in RequestClient._get_result(self, response, cls, seq)
     39 def _get_result(
     40     self, response: httpx.Response, cls: Optional[Type[T]] = None, seq: bool = False
     41 ) -> Union[Any, T]:
     42     """
     43     Decode a json response and assert that it was successful.
     44     """
---> 45     self._raise_for_status(response)
     46     try:
     47         return self._decode(response.json(), cls, seq)

File /lore/utils/request_client.py:37, in RequestClient._raise_for_status(self, response)
     35     response.raise_for_status()
     36 except httpx.HTTPStatusError as e:
---> 37     raise RequestError.from_http_error(e) from e

RequestError: Client error '400 Bad Request' for url 'http://genai.demo.determined.ai:8080/genai/api/v1/dataset'
For more information check: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/400
Response text: {"message":"duplicate key value violates unique constraint \"dataset_name_workspace_id_index\"\nDETAIL:  Key (name, workspace_id)=(arxiv_abstracts_2021_short, 628) already exists."}