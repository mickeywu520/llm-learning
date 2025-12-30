from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
import psutil 
from transformers import TrainingArguments

# 1. 加载模型
max_seq_length = 2048 # Can choose any sequence length!, default is 4096
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/gemma-3-270m-it",  # 如果报错请改模型名
    max_seq_length = max_seq_length, # Choose any for long context!
    load_in_4bit = False,  # 4 bit quantization to reduce memory
    load_in_8bit = False, # [NEW!] A bit more accurate, uses 2x memory
    load_in_16bit = True, # [NEW!] Enables 16bit LoRA
    full_finetuning = False, # [NEW!] We have full finetuning now!
    # token = "hf_...", # use one if using gated models
)

'''
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/gemma-3-270m-it",  # 如果报错请改模型名
    max_seq_length = 1024,
    dtype = torch.float16,
    load_in_4bit = True,   # 省显存
)
'''

# 2. 添加 LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r = 128, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 128, # default value is 128*2
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

# 3. 加载数据
dataset = load_dataset(
    "json",
    data_files = "chatbot_datdaset.json",
    split = "train"
)
print("第一条数据：", dataset[0])

# 4. 数据格式化
'''
def format_prompt(example):
    prompt = (
        f"### 指令:\n{example['instruction']}\n\n"
        f"### 输入:\n{example['input']}\n\n"
        f"### 回答:\n{example['output']}"
    )
    return {"text": prompt}

dataset = dataset.map(format_prompt)
'''

def formatting_func(examples):
    # ========= 关键修复点 =========
    # 如果是单条样本，转成 batch 形式
    if isinstance(examples["instruction"], str):
        instructions = [examples["instruction"]]
        inputs = [examples["input"]]
        outputs = [examples["output"]]
    else:
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]
    # =============================

    texts = []
    for instruction, input_text, response in zip(instructions, inputs, outputs):
        user_content = instruction
        if input_text and input_text.strip():
            user_content += "\n" + input_text

        messages = [
            {"role": "user", "content": user_content},
            {"role": "model", "content": response},
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        texts.append(text)

    return texts  # 必须是 list[str]

# 5. 训练参数
'''
training_args = TrainingArguments(
    output_dir = "./gemma-270m-lora",
    per_device_train_batch_size = 4,
    gradient_accumulation_steps = 4,
    num_train_epochs = 3,
    learning_rate = 2e-4,
    fp16 = True,
    logging_steps = 10,
    save_steps = 200,
    save_total_limit = 2,
    optim = "adamw_8bit",
    report_to = "none",
)
'''

# 6. Trainer
'''
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    eval_dataset = None, # Can set up evaluation!
    args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 1, # Use GA to mimic batch size!
        warmup_steps = 5,
        # num_train_epochs = 1, # Set this for 1 full training run.
        max_steps = 100,
        learning_rate = 5e-5, # Reduce to 2e-5 for long training runs
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir="outputs",
        report_to = "none", # Use TrackIO/WandB etc
    ),
)
'''

# 6. Trainer（关键：不要用 dataset_text_field="text"）
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,  # 直接传原始 dataset
    formatting_func = formatting_func,  # 关键！
    # 注意：这里不设置 dataset_text_field，让 SFTTrainer 自动处理
    args = SFTConfig(
        # 删除 dataset_text_field 行！
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 1,
        warmup_steps = 5,
        max_steps = 100,
        learning_rate = 5e-5,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir="outputs",
        report_to = "none",
    ),
)


'''
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = 1024,
    args = training_args,
)
'''

# 7. 开始训练
trainer.train()

# 8. 保存模型
model.save_pretrained("gemma-270m-lora")
tokenizer.save_pretrained("gemma-270m-lora")

# 9. Save to GGUF locally
print(f' --------- save to GGUF locally ---------\n')
if True: # Change to True to save to GGUF
    model.save_pretrained_gguf(
        "gemma-270m-lora",
        tokenizer,
        quantization_method = "Q8_0", # For now only Q8_0, BF16, F16 supported
    )

