from transformers import AutoTokenizer, AutoModel

# 定义模型名称和保存目录
model_name = 'bert-base-chinese'
save_directory = 'local_bert_model'

# 下载并保存分词器
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(save_directory)

# 下载并保存模型
model = AutoModel.from_pretrained(model_name)
model.save_pretrained(save_directory)

print(f"模型和分词器已成功下载并保存到 {save_directory} 目录。")
