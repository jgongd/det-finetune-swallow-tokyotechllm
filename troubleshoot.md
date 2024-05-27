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