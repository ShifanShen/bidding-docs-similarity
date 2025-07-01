from sentence_transformers import SentenceTransformer

# 定义模型名称和保存目录
model_name = 'shibing624/text2vec-base-chinese'
save_directory = 'local_text2vec_model'

# 下载并保存模型
model = SentenceTransformer(model_name)
model.save(save_directory)

print(f"模型已成功下载并保存到 {save_directory} 目录。")
