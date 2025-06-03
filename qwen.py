from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import torch
from Qwen25_VL.qwen_vl_utils.src.qwen_vl_utils import process_vision_info
from PIL import Image
import numpy as np
import os
import pandas as pd
from sklearn.metrics import f1_score
from tqdm import tqdm
import random


os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model_dir=snapshot_download("Qwen/Qwen2.5-VL-7B-Instruct")   # 下载模型 下载一次
model_path = "./Qwen/Qwen2.5-VL-7B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, device_map='auto', torch_dtype=torch.float16,trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path)
min_pixels = 144 * 28 * 28
max_pixels = 1296 * 28 * 28
processor = AutoProcessor.from_pretrained(model_path, min_pixels=min_pixels, max_pixels=max_pixels, use_fast=True)

model.enable_input_require_grads()   # 允许梯度更新
model.config.use_cache = False

prompt_class = (
    "你是一名楼道消防安全专家，请判断这张图片是否是楼道区域，非楼道区域包括街道、室内以及其他城市空间。如果不是，请输出'非楼道'；如果是，根据图片内容判断其消防隐患等级："
    "- '高风险'：楼道中出现电动车、电瓶、飞线充电等可能起火的元素，存在重大安全隐患。。\n"
    "- '中风险'：楼道内堆放大量物品，严重影响通行，或堆放纸箱、木质或布质的家具、泡沫等可燃物品，需要限期整改。\n"
    "- '低风险'：楼道中有少量物品，基本不影响通行，物品靠边有序摆放。\n"
    "- '无风险'：楼道无堆放物品，符合安全标准。\n"
    "请分析这张图，并返回一个标签：非楼道、无风险、低风险、中风险或高风险。"
)

def process_func(example):
    MAX_LENGTH = 8192
    file_path = example["img_path"]
    output_content = example["label"]
    prompt = prompt_class
    #prompt = '请判断这张图片的消防隐患等级。只能输出以下五种标签中的一个：非楼道、无风险、低风险、中风险、高风险。'#,"resized_height": 224, "resized_width": 224
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"{file_path}","min_pixels": min_pixels, "max_pixels": max_pixels},
                {"type": "text", "text": f"{prompt}"},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text],images=image_inputs,videos=video_inputs,padding=True,return_tensors="pt",)

    inputs = {key: value.tolist() for key, value in inputs.items()}

    # 构造目标输出
    response = tokenizer(f"{output_content}", add_special_tokens=True)
    input_ids = inputs["input_ids"][0] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = inputs["attention_mask"][0] + response["attention_mask"] + [1]
    labels = [-100] * len(inputs["input_ids"][0]) + response["input_ids"] + [tokenizer.pad_token_id]

    # 截断
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {"input_ids": np.array(input_ids),"attention_mask": torch.tensor(np.array(attention_mask)),
        "labels": labels, "pixel_values": torch.tensor(inputs["pixel_values"]),
        "image_grid_thw": torch.tensor(inputs["image_grid_thw"]).squeeze(0)}


from torch.utils.data import Dataset as Dataset
class LazyFireDataset(Dataset):
    def __init__(self, path_file):
        self.samples = []
        with open(path_file, "r") as f:
            for line in tqdm(f, desc="Loading dataset"):
                img_name, label = line.strip().split('\t')
                img_path = os.path.join("./data/train", img_name)
                try:
                    img = Image.open(img_path).load()
                    self.samples.append({"img_path": img_path, "label": label})
                except Exception as e:
                    print(f"[Warning] Failed to load image {img_path}: {e}")
                    continue
        '''
        #B榜加入A榜数据训练
        with open('./data/submitA.txt', "r") as f:
            for line in tqdm(f, desc="Loading dataset"):
                img_name, label = line.strip().split('\t')
                img_path = os.path.join("./data/A", img_name)
                try:
                    img = Image.open(img_path).load()
                    self.samples.append({"img_path": img_path, "label": label})
                except Exception as e:
                    print(f"[Warning] Failed to load image {img_path}: {e}")
                    continue
        '''
        random.shuffle(self.samples)
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return process_func(self.samples[idx])


train_dataset = LazyFireDataset("./data/train.txt")
print(f"Train dataset size: {len(train_dataset)}")

# 3. 配置 LoRA
from peft import LoraConfig, get_peft_model
config = LoraConfig(
    task_type="CAUSAL_LM",
    #target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,
    r=16, # 64,
    lora_alpha=16, #16,
    lora_dropout=0.05,
    bias="none",
)

# 将 LoRA 应用于模型
peft_model = get_peft_model(model, config)
peft_model.enable_input_require_grads()
peft_model.config.use_cache = False  # Ensure no caching during gradient checkpointing

#  4. 训练模型
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq, default_data_collator
args = TrainingArguments(
    output_dir="./output/Qwen2.5-VL-LoRA",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=32,
    logging_steps=10,
    num_train_epochs=3,
    save_steps=64,
    learning_rate=1e-4,
    gradient_checkpointing=True,
    remove_unused_columns=False,
    disable_tqdm=False,
    label_names=["labels"],
    dataloader_num_workers=4,
)

class MyDataCollator(DataCollatorForSeq2Seq):
    def __call__(self, features):
        # 注意这里覆盖的是 labels 处理
        for feature in features:
            if isinstance(feature["labels"], np.ndarray):
                feature["labels"] = feature["labels"].tolist()

        batch = super().__call__(features)

        # 修复警告：提前转成 np.array 后再转 tensor
        if "labels" in batch and isinstance(batch["labels"], list):
            batch["labels"] = torch.tensor(np.array(batch["labels"]), dtype=torch.int64)
        return batch
class MyTrainer(Trainer):
    def _move_model_to_device(self, model, device):
        # 不要移动模型（避免 meta tensor 报错）
        return model
trainer = MyTrainer(
    model=peft_model,
    args=args,
    train_dataset=train_dataset,  # 需要提供数据
    data_collator=MyDataCollator(tokenizer=tokenizer, padding=True),
)
trainer.train()
trainer.save_model()

# 测试
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = os.path.join("./checkpoints", f"run_{timestamp}")
os.makedirs(save_dir, exist_ok=True)
save_log = os.path.join(save_dir, 'train_logs.txt')
def log_training_config(lora_config, training_args, save_log_path):
    with open(save_log_path, "a") as f:
        f.write("\n====== LoRA Configuration ======\n")
        for k, v in vars(lora_config).items():
            f.write(f"{k}: {v}\n")

        f.write("\n====== Training Arguments ======\n")
        for k, v in vars(training_args).items():
            f.write(f"{k}: {v}\n")

        f.write("================================\n\n")
        f.write(prompt_class)
        f.close()

log_training_config(config, args, save_log)

peft_model.eval()
prompt = prompt_class
#prompt = "请判断这张图片的消防隐患等级。只能输出以下五种标签中的一个：非楼道、无风险、低风险、中风险、高风险。"
mapping_dict = {'高风险': 0, '中风险': 1,'低风险': 2,'无风险': 3,'非楼道': 4}
label_to_text = {0: '高风险', 1: '中风险', 2: '低风险', 3: '无风险', 4: '非楼道'}
label_weights = [2.0, 1.5, 1.0, 1.0, 1.0]

class LazyInferDataset(Dataset):
    def __init__(self, csv_path, image_dir):
        self.data = pd.read_csv(csv_path, sep="\t", header=None).values.tolist()
        self.image_dir = image_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data[idx][0]
        img_path = os.path.join(self.image_dir, img_name)
        try:
            image = Image.open(img_path).convert("RGB")
        except:
            return self.__getitem__((idx + 1) % len(self.data))
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"{img_path}","min_pixels": min_pixels, "max_pixels": max_pixels},
                    {"type": "text", "text": f"{prompt}"},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text],images=image_inputs,videos=video_inputs,padding=True,return_tensors="pt",)
        return inputs
def predict(model):
    dataset = LazyInferDataset("./data/B_new.txt", "./data/B")
    all_preds = []
    for idx in tqdm(range(len(dataset)), desc="Predicting"):
        inputs = dataset[idx]  # inputs 是单条样本，已含 batch_size=1
        inputs = inputs.to(model.device)

        with torch.no_grad():  # 调用模型生成输出。max_new_tokens=128表示生成的输出最多包含128个新token。
            generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [  # 从生成的generated_ids中移除输入部分，只保留生成的输出部分。
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        if "高风险" in output_text[0]:
            prediction = "高风险"
        elif "中风险" in output_text[0]:
            prediction = "中风险"
        elif "低风险" in output_text[0]:
            prediction = "低风险"
        elif "无风险" in output_text[0]:
            prediction = "无风险"
        elif "非楼道" in output_text[0]:
            prediction = "非楼道"
        else:   # 并不准确，也没有验证是否有错误输出，一般来看，没有错误输出的情况
            prediction = "无风险"

        #all_preds.append(output_text[0])
        all_preds.append(prediction)

    test_df = pd.read_csv("./data/B_new.txt", sep="\t", header=None)
    test_df["label"] = np.stack(all_preds)
    test_df.to_csv(os.path.join(save_dir, "submit.txt"), index=False, header=False, sep="\t")

predict(peft_model)

'''
class LazyInferDataset1(Dataset1):
    def __init__(self, csv_path, image_dir):
        self.data = pd.read_csv(csv_path, sep="\t", header=None).values.tolist()
        self.image_dir = image_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, label = self.data[idx]
        img_path = os.path.join(self.image_dir, img_name)
        try:
            image = Image.open(img_path).convert("RGB")
        except:
            return self.__getitem__((idx + 1) % len(self.data))
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"{img_path}","min_pixels": min_pixels, "max_pixels": max_pixels},
                    {"type": "text", "text": f"{prompt}"},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text],images=image_inputs,videos=video_inputs,padding=True,return_tensors="pt",)
        return inputs, mapping_dict[label]

def predict1(model):
    dataset = LazyInferDataset1("./data/shuffled_label_A.txt", "/opt/data/private/qwen/data/A")
    all_preds, all_gts = [], []
    for idx in tqdm(range(len(dataset)), desc="Predicting"):
        inputs, gt = dataset[idx]  # inputs 是单条样本，已含 batch_size=1
        inputs = inputs.to(model.device)

        with torch.no_grad():  # 调用模型生成输出。max_new_tokens=128表示生成的输出最多包含128个新token。
            generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [  # 从生成的generated_ids中移除输入部分，只保留生成的输出部分。
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        #all_preds.append(output_text[0])
        #print(output_text[0])
        # 简单关键词匹配提取分类标签
        if "高风险" in output_text[0]:
            prediction = "高风险"
        elif "中风险" in output_text[0]:
            prediction = "中风险"
        elif "低风险" in output_text[0]:
            prediction = "低风险"
        elif "无风险" in output_text[0]:
            prediction = "无风险"
        elif "非楼道" in output_text[0]:
            prediction = "非楼道"
        else:
            prediction = "无风险"

        #all_preds.append(output_text[0])
        all_preds.append(prediction)
        all_gts.append(gt)

    save_submit(all_preds, all_gts)

def save_submit(preds_text,all_gts):
    test_df = pd.read_csv("./data/shuffled_A.txt", sep="\t", header=None)
    test_df["label"] = np.stack(preds_text)
    original_df = pd.read_csv("./data/A.txt", sep="\t", header=None)
    reordered_labels = test_df.set_index(0).loc[original_df[0].tolist()].reset_index(drop=True)
    result_df = pd.concat([original_df[0], reordered_labels], axis=1)
    result_df.to_csv(os.path.join(save_dir,"submit.txt"), index=False, header=False, sep="\t")

    all_preds = [mapping_dict[pred] for pred in preds_text]
    f1s = f1_score(all_gts, all_preds, average=None, labels=range(len(label_weights)))
    weighted_f1 = sum(f * w for f, w in zip(f1s, label_weights))  # / sum(label_weights)
    print(f"A:  Weighted F1={weighted_f1:.4f}, Class F1s={f1s}")

    arr = test_df['label'].value_counts()  # .index
    with open(save_log, 'a') as fp:
        fp.write("\n")
        fp.write(f"A:  Weighted F1={weighted_f1:.4f}, Class F1s={f1s}\n")
        for label, count in arr.items():
            fp.write(f"{label}\t{count}\n")
        fp.close()
    print(arr)
    acc(all_gts, all_preds)

from sklearn.metrics import accuracy_score
def acc(all_gts, all_preds):
    accuracy = accuracy_score(all_gts, all_preds)
    print(f"\nA:  Accuracy={accuracy:.4f}")
    class_accuracies = []
    for label in range(len(label_weights)):
        class_mask = [gt == label for gt in all_gts]
        class_preds = [pred for pred, mask in zip(all_preds, class_mask) if mask]
        class_gts = [gt for gt, mask in zip(all_gts, class_mask) if mask]
        class_accuracy = accuracy_score(class_gts, class_preds)
        class_accuracies.append(class_accuracy)
    print(f"Class Accuracies: {class_accuracies}")
    with open(save_log, 'a') as fp:
        fp.write(f"\nA:  Accuracy={accuracy:.4f}, Class Accuracies={class_accuracies}\n")
        fp.close()
'''


