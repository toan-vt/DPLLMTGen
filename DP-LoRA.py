from peft import get_peft_model, LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, prepare_model_for_kbit_training, default_data_collator
import torch
from torch.utils.data import DataLoader
from opacus.privacy_engine import PrivacyEngine
from opacus.accountants.utils import get_noise_multiplier
from tqdm import tqdm

token = ""
model_name = "meta-llama/Llama-2-7b-chat-hf"

peft_config = LoraConfig(
            lora_alpha=8,
            lora_dropout=0.0,
            r=8,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
        )
tokenizer = AutoTokenizer.from_pretrained(model_name, 
                                            cache_dir=f'.cache/huggingface/{model_name}',
                                            padding_side="left")
tokenizer.pad_token_id = tokenizer.bos_token_id

model = AutoModelForCausalLM.from_pretrained(model_name, token=token, trust_remote_code=True,
                                            cache_dir=f'.cache/huggingface/{model_name}',
                                            torch_dtype=torch.bfloat16,
                                            load_in_4bit=True
                                        )
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)
model.config.pad_token_id = tokenizer.pad_token_id
device = "cuda"
model = model.to(device)

batch_size = 64 
accum_steps = 2 # accummulating gradients
epsilon = 1.0
delta = 1e-5
num_epochs = 10
lr = 5e-4
max_grad_norm = 1.0

train_dataset = None # your training dataset (in huggingface dataset format)
train_dataloader = DataLoader(
    train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
)
eval_dataset = None # your evaluation dataset (in huggingface dataset format)
eval_dataloader = DataLoader(
    eval_dataset, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
)

privacy_engine = PrivacyEngine()
sample_rate = accum_steps*batch_size / len(train_dataset) # sample rate with gradient accumulation
noise_multiplier = get_noise_multiplier(target_epsilon=epsilon, target_delta=delta, epochs=num_epochs, sample_rate=sample_rate)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
model, optimizer, train_dataloader = privacy_engine.make_private(
    model=model,
    optimizer=optimizer,
    data_loader=train_dataloader,
    noise_multiplier=noise_multiplier,
    max_grad_norm=max_grad_norm
)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    current_step = 0
    for step, batch in enumerate(tqdm(train_dataloader)):
        current_step += 1
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss = loss
        total_loss += loss.item()
        loss.backward()
        if current_step % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            
    model.eval()
    test_total_loss = 0
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
            loss = outputs.loss
            test_total_loss += loss.item()
    print(f"Epoch {epoch} | Train Loss: {total_loss/len(train_dataloader)}, Test Loss: {test_total_loss / len(eval_dataloader)}")
