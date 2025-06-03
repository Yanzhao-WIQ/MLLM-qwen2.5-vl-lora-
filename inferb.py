
# 1. 加载模型
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import torch
from Qwen25_VL.qwen_vl_utils.src.qwen_vl_utils import process_vision_info
from PIL import Image
import numpy as np
import os
import pandas as pd
from sklearn.metrics import f1_score
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "./Qwen/Qwen2.5-VL-7B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, device_map='auto', torch_dtype=torch.float16,trust_remote_code=True)#.to(device)'auto'
min_pixels = 144 * 28 * 28
max_pixels = 1296 * 28 * 28
processor = AutoProcessor.from_pretrained(model_path, min_pixels=min_pixels, max_pixels=max_pixels, use_fast=True)
model.enable_input_require_grads()   # 允许梯度更新
model.config.use_cache = False

prompt1 = (
    "你是一名楼道消防安全专家，请判断这张图片是否是楼道区域，如果不是，请输出'非楼道'；如果是，根据图片内容判断其消防隐患等级："
    "- '高风险'：楼道中出现电动车、电瓶、飞线充电等可能起火的元素。\n"
    "- '中风险'：楼道内堆放大量物品，严重影响通行，或堆放纸箱、木质或布质的家具、泡沫等可燃物品。\n"
    "- '低风险'：楼道中有少量物品，基本不影响通行，物品靠边有序摆放。\n"
    "- '无风险'：楼道无堆放物品。\n"
    "请分析这张图，并返回一个标签：非楼道、无风险、低风险、中风险或高风险。"
)

prompt11 = (
    "你是一名楼道消防安全专家，请判断这张图片是否是楼道区域，非楼道区域包括街道、室内以及其他城市空间。如果不是，请输出'非楼道'；如果是，根据图片内容判断其消防隐患等级："
    "- '高风险'：楼道中出现电动车、电瓶、飞线充电等可能起火的元素，存在重大安全隐患。\n"
    "- '中风险'：楼道内堆放大量物品，严重影响通行，或堆放纸箱、木质或布质的家具、泡沫等可燃物品，需要限期整改。\n"
    "- '低风险'：楼道中有少量物品，基本不影响通行，物品靠边有序摆放。\n"
    "- '无风险'：楼道无堆放物品，符合安全标准。\n"
    "请分析这张图，并返回一个标签：非楼道、无风险、低风险、中风险或高风险。"
)
prompt = (
    "你是一名楼道消防安全专家，请判断这张图片是否是楼道区域，非楼道区域包括街道、室内以及其他城市空间。如果不是，请输出'非楼道'；如果是，根据图片内容判断其消防隐患等级："
    "- '高风险'：楼道中出现电动车、电瓶、飞线充电等可能起火的元素，存在重大安全隐患。\n"
    "- '中风险'：楼道内堆放大量物品，严重影响通行，需要限期整改。\n"
    "- '低风险'：楼道中有少量物品，基本不影响通行，物品靠边有序摆放。\n"
    "- '无风险'：楼道无堆放物品，符合安全标准。\n"
    "请分析这张图，请特别注意'中风险'与'低风险'的区分。返回一个标签：非楼道、无风险、低风险、中风险或高风险。"
)
# 3. 配置 LoRA
from peft import LoraConfig, PeftModel
from torch.utils.data import Dataset as Dataset1
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

latest_checkpoint = "./output05124/Qwen2.5-VL-LoRA/checkpoint-222"
print(f"Latest checkpoint path: {latest_checkpoint}")
val_peft_model = PeftModel.from_pretrained(model, latest_checkpoint, config=config)
val_peft_model.eval()

save_dir = "./checkpointsB/runB_20250514_195547-12-8"
save_log = os.path.join(save_dir, 'train_logs.txt')

mapping_dict = {'高风险': 0, '中风险': 1,'低风险': 2,'无风险': 3,'非楼道': 4}
label_to_text = {0: '高风险', 1: '中风险', 2: '低风险', 3: '无风险', 4: '非楼道'}
label_weights = [2.0, 1.5, 1.0, 1.0, 1.0]


class LazyInferDataset(Dataset1):
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
    dataset = LazyInferDataset("./data/B_new.txt", "./qwen/data/B")
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

    test_df = pd.read_csv("./data/B_new.txt", sep="\t", header=None)
    test_df["label"] = np.stack(all_preds)
    test_df.to_csv(os.path.join(save_dir, "submit.txt"), index=False, header=False, sep="\t")

    arr = test_df['label'].value_counts()  # .index
    with open(save_log, 'a') as fp:
        fp.write(f"{latest_checkpoint}\n")
        for label, count in arr.items():
            fp.write(f"{label}\t{count}\n")
        fp.close()
    print(arr)

opt = 0
predict(val_peft_model)

