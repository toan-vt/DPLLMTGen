from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator, get_linear_schedule_with_warmup
from peft import get_peft_config, get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, PeftType, IA3Config, LoraConfig, prepare_model_for_kbit_training, PeftModel, PeftConfig
import torch
from datasets import load_dataset, Dataset, DatasetDict
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import time
import numpy as np
import gc
import random
from src.preprocessing import TASK_DESCRIPTION

EVAL_BATCH_SIZE = 16
MAX_PREFIX_LENGTH = 10

def is_number(s):
    try:
        # if nan return false
        if 'nan' in s:
            return False
        float(s)  # Try to convert the string to a float
        return True
    except ValueError:
        return False

def to_numbers(numbers_str):
    try:
        return True, [float(s.strip()) for s in numbers_str.split()]
    except:
        return False, None

class DPLMTabGen:
    def __init__(self,
                model_name,
                cache_dir=".cache",
                device="auto",
                token=None,
                ft_model_path=None,
                ):
        self.model_name = model_name
        self.optimizer = None
        self.device = device
        self.is_dp_model = False

        if (ft_model_path == None):
            self.load_tokenizer_and_model(model_name, cache_dir, token, device)
        else:
            self.load_ft_mode(cache_dir, model_name, "left", device, token, ft_model_path)

        num_params = sum(p.numel() for p in self.model.parameters())
        num_trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Number of parameters: {num_params}")
        print(f"Number of trainable parameters: {num_trainable_params}")
        print(f"Percentage of trainable parameters: {num_trainable_params / num_params * 100:.2f}%")

    def load_tokenizer_and_model(self, model_name, cache_dir, token, device):
        # model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
        # tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

        if 'gpt2' in model_name:
            target_modules = ["c_attn", "c_proj"]
        if 'Phi-3' in model_name:
            target_modules = ["qkv_proj", "o_proj"]
        elif 'Llama' in model_name:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        elif 'gpt-neo' in model_name:
            target_modules = ["q_proj", "k_proj", "v_proj", "out_proj"]
        peft_config = LoraConfig(
            lora_alpha=8,
            lora_dropout=0.0,
            r=8,
            bias="none",
            task_type="CAUSAL_LM",
            # target_modules=["qkv_proj"]
            target_modules=target_modules
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, 
                                                       cache_dir=f'{cache_dir}/huggingface/{model_name}',
                                                       padding_side="left",
                                                       token=token, trust_remote_code=True)
        if 'Llama' in model_name:
            self.tokenizer.pad_token_id = self.tokenizer.bos_token_id
        elif 'gpt' in model_name:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        elif 'Mistral' in model_name:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        if device == "auto":
            self.model = AutoModelForCausalLM.from_pretrained(model_name, token=token, trust_remote_code=True,
                                                        cache_dir=f'{cache_dir}/huggingface/{model_name}',
                                                        torch_dtype=torch.bfloat16,
                                                        load_in_4bit=True
                                                        # device_map="auto"
                                                    )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, token=token, trust_remote_code=True,
                                                        cache_dir=f'{cache_dir}/huggingface/{model_name}',
                                                        torch_dtype=torch.bfloat16,
                                                        load_in_4bit=True
                                                    )


        # print model architecture, layer names
        for name, param in self.model.named_parameters():
            print(name, param.shape)

        # self.model.gradient_checkpointing_enable()
        self.model = prepare_model_for_kbit_training(self.model,)
        self.model = get_peft_model(self.model, peft_config)
        if device != "auto":
            self.model.to(device)
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def load_ft_mode(self, cache_dir, model_name, padding_side, device, token, ft_model_path):
        peft_config = PeftConfig.from_pretrained(ft_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    cache_dir=cache_dir,
                    padding_side=padding_side,
                    token=token
                )
        if 'Llama' in model_name:
            self.tokenizer.pad_token_id = self.tokenizer.bos_token_id
        elif 'gpt' in model_name:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        elif 'Mistral' in model_name:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        base_model = AutoModelForCausalLM.from_pretrained(ft_model_path, trust_remote_code=True,
                                        cache_dir=f'{cache_dir}/huggingface/{model_name}',
                                        torch_dtype=torch.bfloat16,
                                        token=token, load_in_4bit=True
                                    )
            
        # base_model.gradient_checkpointing_enable()
        base_model = prepare_model_for_kbit_training(base_model)
        self.model = PeftModel.from_pretrained(base_model, ft_model_path, config=peft_config)
        if device != "auto":
            self.model.to(device)
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        # TODO for others that are not LORA
        for name, param in self.model.named_parameters():
            if 'lora' in name:
                param.requires_grad = True

    def format_learning_tune(self,
                             fake_dataset,
                             fake_eval_dataset=None,
                             num_epochs=10,
                             batch_size=64, 
                             lr=3e-3, 
                             save_log=True,
                             log_file="./logs/log.txt",
                             save_every_epoch=False,
                             save_model_dir="./models",
                             verbose=False,
                             format_checking=False,
                             accum_steps=1
                             ):
        train_dataloader = DataLoader(
            fake_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
        )
        if fake_eval_dataset != None:
            eval_dataloader = DataLoader(
                fake_eval_dataset, collate_fn=default_data_collator, batch_size=EVAL_BATCH_SIZE, pin_memory=True
            )
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

        # self.lr_scheduler = get_linear_schedule_with_warmup(
        #     optimizer=self.optimizer,
        #     num_warmup_steps=0,
        #     num_training_steps=(len(train_dataloader) * num_epochs * 1.5),
        # )

        df_log = []
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        current_step = 0
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            print("current lr: ", self.optimizer.param_groups[0]['lr'])
            for step, batch in enumerate(tqdm(train_dataloader, disable=not verbose)):
                current_step += 1
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                total_loss += loss.detach().float()
                loss.backward()
                if ((step + 1) % accum_steps == 0) or (step == len(train_dataloader) - 1):
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                # self.lr_scheduler.step()
            train_epoch_loss = total_loss / len(train_dataloader)
            train_ppl = torch.exp(train_epoch_loss)
            train_epoch_loss = train_epoch_loss.item()
            train_ppl = train_ppl.item()

            if fake_eval_dataset != None:
                with torch.no_grad():
                    self.model.eval()
                    eval_loss = 0
                    for step, batch in enumerate(tqdm(eval_dataloader, disable=not verbose)):
                        batch = {k: v.to(device) for k, v in batch.items()}
                        with torch.no_grad():
                            outputs = self.model(**batch)
                        loss = outputs.loss
                        eval_loss += loss.detach().float()
                    eval_epoch_loss = eval_loss / len(eval_dataloader)
                    eval_ppl = torch.exp(eval_epoch_loss)
                    eval_epoch_loss = eval_epoch_loss.item()
                    eval_ppl = eval_ppl.item()
                    df_log.append({'epoch': epoch, 'step': current_step, 'train_loss': train_epoch_loss, 'train_ppl': train_ppl, 'eval_loss': eval_epoch_loss, 'eval_ppl': eval_ppl})
                    if verbose:
                        now = time.strftime("%Y-%m-%d %H:%M:%S")
                        print(f"{now} - Epoch {epoch} - Train Loss: {train_epoch_loss} Train PPL: {train_ppl} Eval Loss: {eval_epoch_loss} Eval PPL: {eval_ppl}")
            else:
                df_log.append({'epoch': epoch, 'step': current_step, 'train_loss': train_epoch_loss, 'train_ppl': train_ppl})
                if verbose:
                    now = time.strftime("%Y-%m-%d %H:%M:%S")
                    print(f"{now} - Epoch {epoch} - Train Loss: {train_epoch_loss} Train PPL: {train_ppl}")

            if save_every_epoch:
                if not os.path.exists(save_model_dir):
                    os.makedirs(save_model_dir)
                self.model.save_pretrained(f"{save_model_dir}/p1_epoch_{epoch}")
                torch.save(self.optimizer.state_dict(), f"{save_model_dir}/p1_epoch_{epoch}/optimizer.pt")
                # torch.save(self.lr_scheduler.state_dict(), f"{save_model_dir}/p1_epoch_{epoch}/lr_scheduler.pt")

            self.model.save_pretrained(f"{save_model_dir}/p1_final")
            torch.save(self.optimizer.state_dict(), f"{save_model_dir}/p1_final/optimizer.pt")
            # torch.save(self.lr_scheduler.state_dict(), f"{save_model_dir}/p1_final/lr_scheduler.pt")

        if save_log:
            df_log = pd.DataFrame(df_log)
            # creat folder if not exist
            if not os.path.exists(os.path.dirname(log_file)):
                os.makedirs(os.path.dirname(log_file))
            df_log.to_csv(log_file, index=False)

    def dp_tune(self,
                train_dataset,
                eval_dataset=None,
                num_epochs=10,
                batch_size=64,
                accum_steps=1,
                lr=3e-3,
                epsilon=5.0,
                delta=1e-5,
                max_grad_norm=1.0,
                alpha=0.9,
                numerical_loss=False,
                beta=1.0,
                gamma=1.0,
                max_abs_values=None,
                save_every_epoch=False,
                save_every_steps=-1,
                save_model_dir="./dp-models",
                save_log=True,
                log_file="./logs/log.txt",
                verbose=False,
                fastdp=False,
                nondp=False
                ):
        train_dataloader = DataLoader(
            train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
        )
        if eval_dataset != None:
            eval_dataloader = DataLoader(
                eval_dataset, collate_fn=default_data_collator, batch_size=EVAL_BATCH_SIZE, pin_memory=True
            )
        if self.optimizer == None:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
            # self.lr_scheduler = get_linear_schedule_with_warmup(
            #     optimizer=self.optimizer,
            #     num_warmup_steps=0,
            #     num_training_steps=(len(train_dataloader) * num_epochs),
            # )
        if nondp:
            pass
        elif not fastdp:
            from opacus.privacy_engine import PrivacyEngine
            from opacus.accountants.utils import get_noise_multiplier

            privacy_engine = PrivacyEngine()
            sample_rate = accum_steps*batch_size / len(train_dataset)
            noise_multiplier = get_noise_multiplier(target_epsilon=epsilon, target_delta=delta, epochs=num_epochs, sample_rate=sample_rate)
            print("Noise Multiplier using Opacus: ", noise_multiplier)
            # self.model, self.optimizer, train_dataloader = privacy_engine.make_private_with_epsilon(
            #     module=self.model,
            #     optimizer=self.optimizer,
            #     data_loader=train_dataloader,
            #     target_epsilon=epsilon,
            #     max_grad_norm=max_grad_norm,
            #     target_delta=delta,
            #     epochs=num_epochs
            # )
            
            self.model.train()
            self.model, self.optimizer, train_dataloader = privacy_engine.make_private(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=train_dataloader,
                noise_multiplier=noise_multiplier,
                max_grad_norm=max_grad_norm
            )
        else:
            from fastDP import PrivacyEngine
            privacy_engine = PrivacyEngine(
                self.model,
                batch_size=batch_size*accum_steps,
                sample_size=len(train_dataset),
                epochs=num_epochs,
                target_epsilon=epsilon,
                target_delta=delta,
                clipping_fn='automatic',
                clipping_mode='MixOpt',
                origin_params=None,
                clipping_style='all-layer',
            )
            privacy_engine.attach(self.optimizer)
            print("Noise Multiplier using fastDP: ", privacy_engine.noise_multiplier)

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        if numerical_loss:
            if max_abs_values == None:
                raise ValueError("max_abs_values must be provided when numerical_loss is True")
            max_abs_values = torch.tensor(max_abs_values).to(device)

        df_log = []
        current_step = 0
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            for step, batch in enumerate(tqdm(train_dataloader, disable=not verbose)):
                current_step += 1
                is_labels, is_numbers, is_targets, num_columns = batch["is_labels"], batch["is_numbers"], batch["is_targets"], batch["num_columns"]
                del batch["is_labels"], batch["is_numbers"], batch["is_targets"], batch["num_columns"]
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = self.model(**batch)
                
                # calculate new loss func ======================================================
                shifted_logits = torch.cat([torch.zeros(outputs.logits.shape[0], 1, outputs.logits.shape[2]).to("cuda"), outputs.logits[:, :-1]], dim=1)
                flatten_logits = shifted_logits.view(-1, shifted_logits.shape[-1])
                flatten_labels = batch['labels'].view(-1)
                flatten_label_mask = is_labels.view(-1)

                format_logits = flatten_logits[flatten_label_mask == False]
                format_labels = flatten_labels[flatten_label_mask == False]
                value_logits = flatten_logits[flatten_label_mask == True]
                value_labels = flatten_labels[flatten_label_mask == True]

                num_format_tokens = len(format_labels[format_labels != -100])
                num_value_tokens = len(value_labels[value_labels != -100])

                total_format_loss = torch.nn.functional.cross_entropy(format_logits, format_labels)*num_format_tokens
                total_value_loss = torch.nn.functional.cross_entropy(value_logits, value_labels)*num_value_tokens
                loss = 2*(total_format_loss*(1-alpha) + total_value_loss*alpha) / (num_format_tokens + num_value_tokens)
                # loss = (total_format_loss*(1-alpha) + total_value_loss*alpha) / (num_format_tokens + num_value_tokens)
                # ==============================================================================

                # numerical-aware loss =========================================================
                if numerical_loss:
                    output_tokens = torch.argmax(shifted_logits, dim=2)
                    number_loss = 0
                    for idx in range(batch['input_ids'].shape[0]):
                        label_numbers = self.tokenizer.decode(batch['input_ids'][idx][is_numbers[idx] == True], skip_special_tokens=True)
                        output_numbers = self.tokenizer.decode(output_tokens[idx][is_numbers[idx] == True], skip_special_tokens=True)
                        _, label_numbers = to_numbers(label_numbers)
                        success, output_numbers = to_numbers(output_numbers)
                        if success:    
                            for i in range(len(label_numbers)):        
                                number_loss += 0.5*((label_numbers[i] - output_numbers[i])/max_abs_values[num_columns[idx][i]])**2
                        else:
                            number_loss = 2.0*len(label_numbers)
                    number_loss = number_loss / batch['input_ids'].shape[0]
                    loss += beta*number_loss
                # ==============================================================================

                # target-aware loss ============================================================
                if gamma != 0:
                    flatten_target_mask = is_targets.view(-1)
                    target_logits = flatten_logits[flatten_target_mask == True]
                    target_labels = flatten_labels[flatten_target_mask == True]
                    target_loss = torch.nn.functional.cross_entropy(target_logits, target_labels)
                    loss += gamma*target_loss
                # ==============================================================================

                # loss = outputs.loss
                total_loss += loss.detach().float()
                loss.backward()
                if ((step + 1) % accum_steps == 0) or (step == len(train_dataloader) - 1):
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                # self.lr_scheduler.step()
                if save_every_steps != -1 and current_step % save_every_steps == 0:
                    if nondp:
                        epsilon = 99999
                    elif not fastdp:
                        epsilon = privacy_engine.get_epsilon(delta=delta)
                    else:
                        epsilon = privacy_engine.get_privacy_spent()
                    with torch.no_grad():
                        self.model.eval()
                        eval_loss = 0
                        for step, batch in enumerate(tqdm(eval_dataloader, disable=not verbose)):
                            del batch["is_labels"], batch["is_numbers"], batch["is_targets"], batch["num_columns"]
                            batch = {k: v.to(device) for k, v in batch.items()}
                            with torch.no_grad():
                                outputs = self.model(**batch)
                            loss = outputs.loss
                            eval_loss += loss.detach().float()
                        eval_epoch_loss = eval_loss / len(eval_dataloader)
                        eval_ppl = torch.exp(eval_epoch_loss)
                        eval_epoch_loss = eval_epoch_loss.item()
                        eval_ppl = eval_ppl.item()
                        df_log.append({'epoch': 0, 'step': current_step, 'train_loss': 0, 'train_ppl': 0, 
                                    'eval_loss': eval_epoch_loss, 'eval_ppl': eval_ppl, 'epsilon': epsilon})
                    if not os.path.exists(save_model_dir):
                        os.makedirs(save_model_dir)
                    
                    if (not fastdp) and (not nondp):
                        self.model._module.save_pretrained(f"{save_model_dir}/p2_step_{current_step}")
                    else:
                        self.model.save_pretrained(f"{save_model_dir}/p2_step_{current_step}")
            train_epoch_loss = total_loss / len(train_dataloader)
            train_ppl = torch.exp(train_epoch_loss)
            if nondp:
                epsilon = 99999
            elif not fastdp:
                epsilon = privacy_engine.get_epsilon(delta=delta)
            else:
                epsilon = privacy_engine.get_privacy_spent()

            if eval_dataset != None:
                with torch.no_grad():
                    self.model.eval()
                    eval_loss = 0
                    for step, batch in enumerate(tqdm(eval_dataloader, disable=not verbose)):
                        del batch["is_labels"], batch["is_numbers"], batch["is_targets"], batch["num_columns"]
                        batch = {k: v.to(device) for k, v in batch.items()}
                        with torch.no_grad():
                            outputs = self.model(**batch)
                        loss = outputs.loss
                        eval_loss += loss.detach().float()
                    eval_epoch_loss = eval_loss / len(eval_dataloader)
                    eval_ppl = torch.exp(eval_epoch_loss)
                    eval_epoch_loss = eval_epoch_loss.item()
                    eval_ppl = eval_ppl.item()
                    df_log.append({'epoch': epoch, 'step': 0, 'train_loss': train_epoch_loss.item(), 'train_ppl': train_ppl.item(), 
                                'eval_loss': eval_epoch_loss, 'eval_ppl': eval_ppl, 'epsilon': epsilon})

                if verbose:
                    now = time.strftime("%Y-%m-%d %H:%M:%S")
                    print(f"{now} - Epoch {epoch} - Train Loss: {train_epoch_loss} Train PPL: {train_ppl} Eval Loss: {eval_epoch_loss} Eval PPL: {eval_ppl} Epsilon: {epsilon}")
                
            else:
                df_log.append({'epoch': epoch, 'step': 0, 'train_loss': train_epoch_loss.item(), 'train_ppl': train_ppl.item(), 'epsilon': epsilon})
                if verbose:
                    now = time.strftime("%Y-%m-%d %H:%M:%S")
                    print(f"{now} - Epoch {epoch} - Train Loss: {train_epoch_loss} Train PPL: {train_ppl} Epsilon: {epsilon}")

            if save_every_epoch:
                if not os.path.exists(save_model_dir):
                    os.makedirs(save_model_dir)
                if (not fastdp) and (not nondp):
                    self.model._module.save_pretrained(f"{save_model_dir}/p2_epoch_{epoch}_{epsilon}")
                else:
                    self.model.save_pretrained(f"{save_model_dir}/p2_step_{current_step}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self.is_dp_model = True
        if (not fastdp) and (not nondp):
            self.model._module.save_pretrained(f"{save_model_dir}/p2_final_{epsilon}")
        else:
            self.is_dp_model = False
            self.model.save_pretrained(f"{save_model_dir}/p2_final_{epsilon}")
        if save_log:
            # check if folder exist
            if not os.path.exists(os.path.dirname(log_file)):
                os.makedirs(os.path.dirname(log_file))
            df_log = pd.DataFrame(df_log)
            df_log.to_csv(log_file, index=False)
        
    def prompt_in_batch(self, prompts, max_length=128):
        self.model.eval()
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        with torch.no_grad():
            inputs = self.tokenizer(prompts, return_tensors="pt", padding="max_length", max_length=max_length, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = self.model.generate(**inputs, max_length=max_length*2, do_sample=True)
            return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    # def sample_with_self_consitency(self, constraint_dict, num_samples=1024, max_length=128, batch_size=64, target_last=True, temperature=1.0):
    #     if not target_last:
    #         raise ValueError("target_last must be True")
    #     print("Start sampling")
    #     # number of consitent samples
    #     num_consistent_samples = 32
    #     actual_samples_in_batch = int(batch_size/num_consistent_samples)
    #     def get_random_start(constraint_dict):
    #         all_cols = list(constraint_dict.keys())[:-1]
    #         col = np.random.choice(all_cols)
    #         return f'{TASK_DESCRIPTION}{col} is'
    #     column_names = list(constraint_dict.keys())
    #     pbar = tqdm(total=num_samples)
    #     count = 0
    #     df = []
    #     self.model.eval()
    #     device = "cuda:0" if torch.cuda.is_available() else "cpu"
    #     with torch.no_grad():
    #         while count < num_samples:
    #             current_df = []
    #             prompts = [get_random_start(constraint_dict) for _ in range(actual_samples_in_batch)]
    #             inputs = self.tokenizer(prompts, return_tensors="pt", padding="max_length", max_length=max_length).to(device)
    #             if self.is_dp_model:
    #                 outputs = self.model._module.generate(**inputs, max_length=max_length*2, do_sample=True, temperature=temperature)
    #             else:
    #                 outputs = self.model.generate(**inputs, max_length=max_length*2, do_sample=True, temperature=temperature)
    #             outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
    #             outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
    #             outputs = [output.replace(TASK_DESCRIPTION, '') for output in outputs]
    #             for output in outputs:
    #                 wrong_format = False
    #                 col_flags = {}
    #                 for col in constraint_dict:
    #                     col_flags[col] = False
    #                 for substr in output.split(', '):
    #                     for col in column_names:
    #                         if f'{col} is ' in substr:
    #                             val = substr.split(f"{col} is ", 1)[1].strip()
    #                             col_flags[col] = val
    #                             if constraint_dict[col]['type'] == 'categorical':
    #                                 if col_flags[col] not in constraint_dict[col]['unique_values']:
    #                                     print(f'\tERROR: "{col_flags[col]}" not in "{col}"')
    #                                     wrong_format = True
    #                                     break
    #                             elif not is_number(val):
    #                                 print(f'\tERROR: "{val}" is not a number')
    #                                 wrong_format = True
    #                                 break                                
    #                 for col in col_flags:
    #                     if not col_flags[col]:
    #                         wrong_format = True
    #                         break

    #                 if not wrong_format:
    #                     current_df.append(col_flags)
    #                     count += 1
    #                     pbar.update(1)

    #             # start self-consistency checking
    #             if len(current_df) > 0:
    #                 prompts = []
    #                 feature_names = column_names[:-1]
    #                 for sample_features in current_df:
    #                     for _ in range(num_consistent_samples):
    #                         prompt = "{TASK_DESCRIPTION} "
    #                         for feature in feature_names:
    #                             prompt += f'{feature} is {sample_features[feature]} , '
    #                         prompts.append(prompt)
            


    def sample(self, constraint_dict, num_samples=1024, max_length=128, batch_size=64, target_last=False, temperature=1.0):
        """
            constraint_dict = {
                'column_name': {'type': 'categorical', 'unique_values': ['value1', 'value2', ...]},
                'column_name': {'type': 'numerical', 'min': 0, 'max': 100},
            }
        """
        print("Start sampling")
        # def get_random_start(constraint_dict):
        #     cat_cols = [k for k, v in constraint_dict.items() if v['type'] == 'categorical']
        #     col = np.random.choice(cat_cols)
        #     val = np.random.choice(list(constraint_dict[col]['unique_values']))
            
        #     all_cols = list(constraint_dict.keys())
        #     all_cols.remove(col)
        #     second_col = np.random.choice(all_cols)
            
        #     return f'{col} is {val} , {second_col} is'

        def get_random_start(constraint_dict):
            if not target_last:
                col = np.random.choice(list(constraint_dict.keys()))
            else:
                col = np.random.choice(list(constraint_dict.keys())[:-1])
            return f'{TASK_DESCRIPTION}'

        column_names = list(constraint_dict.keys())
        pbar = tqdm(total=num_samples)
        count = 0
        df = []
        self.model.eval()
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        with torch.no_grad():
            while count < num_samples:
                # print("count: ", count)
                prompts = [get_random_start(constraint_dict) for _ in range(batch_size)]
                inputs = self.tokenizer(prompts, return_tensors="pt", padding="max_length", max_length=6).to(device)
                if self.is_dp_model:
                    outputs = self.model._module.generate(**inputs, max_length=max_length+6, do_sample=True, temperature=temperature)
                else:
                    outputs = self.model.generate(**inputs, max_length=max_length+6, do_sample=True, temperature=temperature)
                outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                outputs = [output.replace(TASK_DESCRIPTION, '') for output in outputs]
                for output in outputs:
                    wrong_format = False
                    col_flags = {}
                    for col in constraint_dict:
                        col_flags[col] = False
                    for substr in output.split(', '):
                        for col in column_names:
                            if f'{col} is ' in substr:
                                val = substr.split(f"{col} is ", 1)[1].strip()
                                col_flags[col] = val
                                if constraint_dict[col]['type'] == 'categorical':
                                    if col_flags[col] not in constraint_dict[col]['unique_values']:
                                        print(f'\tERROR: "{col_flags[col]}" not in "{col}"')
                                        wrong_format = True
                                        break
                                elif not is_number(val):
                                    print(f'\tERROR: "{val}" is not a number')
                                    wrong_format = True
                                    break
                                # elif (float(val) < constraint_dict[col]['min']) or(float(val) > constraint_dict[col]['max']):
                                #     wrong_format = True
                                #     break
                                
                    for col in col_flags:
                        if not col_flags[col]:
                            wrong_format = True
                            break

                    if not wrong_format:
                        df.append(col_flags)
                        count += 1
                        pbar.update(1)
                        
        print("Finish sampling")
        return pd.DataFrame(df)
    
    def sample_dev(self, constraint_dict, num_samples=1024, max_length=128, batch_size=64, target_last=False, temperature=1.0):
        """
            constraint_dict = {
                'column_name': {'type': 'categorical', 'unique_values': ['value1', 'value2', ...]},
                'column_name': {'type': 'numerical', 'min': 0, 'max': 100},
            }
        """
        print("Start sampling")
        # def get_random_start(constraint_dict):
        #     cat_cols = [k for k, v in constraint_dict.items() if v['type'] == 'categorical']
        #     col = np.random.choice(cat_cols)
        #     val = np.random.choice(list(constraint_dict[col]['unique_values']))
            
        #     all_cols = list(constraint_dict.keys())
        #     all_cols.remove(col)
        #     second_col = np.random.choice(all_cols)
            
        #     return f'{col} is {val} , {second_col} is'

        def get_random_start(constraint_dict):
            if not target_last:
                col = np.random.choice(list(constraint_dict.keys()))
            else:
                col = np.random.choice(list(constraint_dict.keys())[:-1])
            return f'{TASK_DESCRIPTION}'
        column_names = list(constraint_dict.keys())
        pbar = tqdm(total=num_samples)
        count = 0
        df = []
        self.model.eval()
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        with torch.no_grad():
            while count < num_samples:
                # print("count: ", count)
                prompts = [get_random_start(constraint_dict) for _ in range(batch_size)]
                inputs = self.tokenizer(prompts, return_tensors="pt", padding="max_length", max_length=MAX_PREFIX_LENGTH).to(device)
                if self.is_dp_model:
                    outputs = self.model._module.generate(**inputs, max_length=max_length+MAX_PREFIX_LENGTH, do_sample=True, temperature=temperature, output_scores=True, output_logits=True, return_dict_in_generate=True)
                else:
                    outputs = self.model.generate(**inputs, max_length=max_length+MAX_PREFIX_LENGTH, do_sample=True, temperature=temperature, output_scores=True, output_logits=True, return_dict_in_generate=True)

                # print(outputs.sequences[0])
                # print(self.tokenizer.encode(", Reached.on.Time is"))
                # print(self.tokenizer.decode([3363]))
                # print(outputs.keys())
                # for key in outputs.keys():
                #     print(key, type(outputs[key]))
                # print("max_length: ", max_length)
                # print("outputs.sequences length: ", len(outputs.sequences))
                # print("outputs.scores length: ", len(outputs.scores))
                # print("outputs.logits length: ", len(outputs.logits))
                # print(type(outputs.scores))

                sequences = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
                sequences = [seq.replace(TASK_DESCRIPTION, '') for seq in sequences]
                for i, output in enumerate(sequences):
                    wrong_format = False
                    col_flags = {}
                    for col in constraint_dict:
                        col_flags[col] = False
                    for substr in output.split(', '):
                        for col in column_names:
                            if f'{col} is ' in substr:
                                val = substr.split(f"{col} is ", 1)[1].strip()
                                col_flags[col] = val
                                if constraint_dict[col]['type'] == 'categorical':
                                    if col_flags[col] not in constraint_dict[col]['unique_values']:
                                        print(f'\tERROR: "{col_flags[col]}" not in "{col}"')
                                        wrong_format = True
                                        break
                                elif not is_number(val):
                                    print(f'\tERROR: "{val}" is not a number')
                                    wrong_format = True
                                    break
                                # elif (float(val) < constraint_dict[col]['min']) or(float(val) > constraint_dict[col]['max']):
                                #     wrong_format = True
                                #     break
                                
                    for col in col_flags:
                        if not col_flags[col]:
                            wrong_format = True
                            break

                    if not wrong_format:                        
                        # get the index of the token that is the last one but not the special token
                        last_token_index_in_seq = 0
                        # print(f"seq: {output}")
                        # print("token ids:", outputs.sequences[i])
                        for j in range(len(outputs.sequences[i])-1, -1, -1):
                            # print(f"token {j}: {outputs.sequences[i][j]}")
                            if outputs.sequences[i][j] != self.tokenizer.pad_token_id:
                                # print(f"selected token {j}: {outputs.sequences[i][j]}")
                                last_token_index_in_seq = j
                                break
                        last_token_index_in_score = last_token_index_in_seq - MAX_PREFIX_LENGTH
                        score = outputs.scores[last_token_index_in_score][i]
                        # get probability by softmax on score
                        prob = torch.nn.functional.softmax(score, dim=0)
                        # print(f"score: {score}")
                        # print(f"prob: {prob}")

                        # get top k tokens
                        k = 2
                        top_k = torch.topk(prob, k)
                        print(f"top {k} tokens: {top_k}")
                        for j in range(k):
                            print(f"token: {top_k.indices[j]}, prob: {top_k.values[j]}, token: {self.tokenizer.decode([top_k.indices[j]])}")
                        print("\n")

                        # get the token with the highest probability
                        top_token_index = torch.argmax(prob)
                        # check if it was sellected
                        if top_token_index != outputs.sequences[i][last_token_index_in_seq]:
                            # print(f"top token: {top_token_index}, prob: {prob[top_token_index]}, token: {self.tokenizer.decode([top_token_index])}")
                            # print(f"selected token: {outputs.sequences[i][last_token_index_in_seq]}, prob: {prob[outputs.sequences[i][last_token_index_in_seq]]}, token: {self.tokenizer.decode([outputs.sequences[i][last_token_index_in_seq]])}")
                            # print(f"seq: {output}")
                            # print(f"score: {score}")
                            # print(f"prob: {prob}")
                            # print("\n")
                            continue
                        if (prob[top_token_index] < 0.6): # only get high prob samples
                            continue

                        df.append(col_flags)
                        count += 1
                        pbar.update(1)


        print("Finish sampling")
        return pd.DataFrame(df)

    def format_checking(self, constraint_dict, num_samples=256, max_length=128, batch_size=64, with_prob=False, verbose=False, target_last=False, wrong_format_verbose=False):
        """
            constraint_dict = {
                'column_name': {'type': 'categorical', 'unique_values': ['value1', 'value2', ...]}, 'prob': 
                'column_name': {'type': 'numerical', 'min': 0, 'max': 100},
            }
        """
        # def get_random_start(constraint_dict):
        #     cat_cols = [k for k, v in constraint_dict.items() if v['type'] == 'categorical']
        #     col = np.random.choice(cat_cols)
        #     val = np.random.choice(list(constraint_dict[col]['unique_values']))
            
        #     all_cols = list(constraint_dict.keys())
        #     all_cols.remove(col)
        #     second_col = np.random.choice(all_cols)
            
        #     return f'{col} is {val} , {second_col} is'

        def get_random_start(constraint_dict):
            if not target_last:
                all_cols = list(constraint_dict.keys())
            else:
                all_cols = list(constraint_dict.keys())[:-1]
            col = np.random.choice(all_cols)
            
            return f'{TASK_DESCRIPTION}'


        column_names = list(constraint_dict.keys())
        count = 0
        df = []

        passed_count = 0
        self.model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pbar = tqdm(total=num_samples, disable=not verbose)
        with torch.no_grad():
            while count < num_samples:
                prompts = [get_random_start(constraint_dict) for _ in range(batch_size)]
                inputs = self.tokenizer(prompts, return_tensors="pt", padding="max_length", max_length=MAX_PREFIX_LENGTH).to(device)
                if self.is_dp_model:
                    outputs = self.model._module.generate(**inputs, max_length=max_length + MAX_PREFIX_LENGTH, do_sample=True)
                else:
                    outputs = self.model.generate(**inputs, max_length=max_length + MAX_PREFIX_LENGTH, do_sample=True)
                outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                outputs = [output.replace(TASK_DESCRIPTION, '') for output in outputs]
                for output in outputs:
                    if wrong_format_verbose:
                        print(f'output: "{output}"')
                    wrong_format = False
                    col_flags = {}
                    for col in constraint_dict:
                        col_flags[col] = False
                    for substr in output.split(', '):
                        for col in column_names:
                            if f'{col} is ' in substr:
                                val = substr.split(f"{col} is ", 1)[1].strip()
                                col_flags[col] = val
                                if constraint_dict[col]['type'] == 'categorical':
                                    if col_flags[col] not in constraint_dict[col]['unique_values']:
                                        if wrong_format_verbose:
                                            print(f'\tERROR: "{col_flags[col]}" not in "{col}"')
                                        wrong_format = True
                                        break
                                elif not is_number(val):
                                    if wrong_format_verbose:
                                        print(f'\tERROR: "{val}" is not a number of "{col}"')
                                    wrong_format = True
                                    break
                                # elif (float(val) < constraint_dict[col]['min']) or(float(val) > constraint_dict[col]['max']):
                                #     if wrong_format_verbose:
                                #         print(f'\tERROR: "{val}" is out of range of "{col}"')
                                #     wrong_format = True
                                #     break

                    for col in col_flags:
                        if not col_flags[col]:
                            wrong_format = True
                            break

                    if not wrong_format:
                        passed_count += 1
                    count += 1
                    pbar.update(1)
        pbar.close()

        return passed_count/count
    