# 加载流程

加载模型的流程：`from_pretrained` → 读 `config.json` → 调用模型类的 `__init__` 构建结构 → 加载 `pytorch_model.bin` 权重到结构中 → 调用 `forward` 完成前向计算。

模型的代码不会通过`from_pretrained`方法下载到本地，读取代码是读取的本地已安装的`transformers`库源码中的代码。

本地模型文件：

```plaintext
./gpt2_local/
├── config.json          # 模型配置
├── pytorch_model.bin    # 模型权重
├── tokenizer_config.json # 分词器配置
└── tokenizer.json       # 分词器词表
```



# tokenizer

tokenizer用于将文本转化成词典序号和掩码，不同的tokenizer对应的词典和转换后的掩码有所不同。

```python
from transformers import AutoTokenizer

# 根据任务类型选择对应的tokenizer，模型和tokenizer都用同一个model_name，如果不同需要能兼容，例如tokenizer = AutoTokenizer.from_pretrained("gpt2") && model = AutoModel.from_pretrained("gpt1")
model_name = ...
tokenizer = AutoTokenizer.from_pretrained(model_name)

text = ...	# 文本

input_ids = tokenizer.encode(text)	# 得到词典序号的列表
inputs = tokenizer(text)	# 得到input_ids, attention_mask, token_type_ids等组成的列表
```



# 常用模型加载类

1. `AutoModel`

   - **描述**：通用的模型加载类，仅加载模型的**主体部分**，不附带任何特定任务的输出头，适用于需要自定义任务头的情况。

   - 适用场景：

     - **特征提取**：从文本中提取嵌入或特征，用于下游任务。
   - **自定义任务**：为特定任务设计专属的输出层，如自定义的分类器或回归器。

2. `AutoModelForCausalLM`

   - **描述**：用于因果语言建模（Causal Language Modeling）任务，包含适用于生成任务的输出头。

   - 适用场景：

     - **文本生成**：如对话系统、内容创作、自动补全等。
     - **因果语言建模**：根据上下文生成后续文本的任务。
     - **快速部署**：无需额外添加任务头，适合快速搭建对话系统。

3. `AutoModelForMaskedLM`

   - **描述**：用于掩码语言建模（Masked Language Modeling）任务，包含适用于填空任务的输出头。

   - 适用场景：

     - **填空任务**：如句子补全、文本理解等。
   - **预训练模型微调**：进一步训练模型以增强其理解能力。

4. `AutoModelForSeq2SeqLM`

   - **描述**：用于序列到序列（Sequence-to-Sequence）任务，包含编码器-解码器架构的输出头。

   - 适用场景：

     - **机器翻译**：将一种语言翻译成另一种语言。
   - **文本摘要**：生成文本的简短摘要。

5. `AutoModelForQuestionAnswering`

   - **描述**：用于问答任务，包含用于预测答案起始和结束位置的输出头。

   - 适用场景：

     - **抽取式问答**：从文本中提取并生成问题的答案。

6. `AutoModelForTokenClassification`

   - **描述**：用于标注任务，如命名实体识别，包含专门的输出头。

   - 适用场景：

     - **命名实体识别（NER）**：识别文本中的实体，如人名、地名等。
   - **词性标注**：为每个词分配词性标签。

7. `AutoModelForSequenceClassification`

   - **描述**：用于序列分类任务，包含分类头部。

   - 适用场景：

     - **文本分类**：如情感分析、主题分类等。

- **语音识别后处理**：对转录的文本进行分类。



# 代码示例

`AutoModelForCausalLM`：

```python
##################默认从HF下载，即使本地有缓存############
from transformers import AutoTokenizer, AutoModelForCausalLM

# 指定模型名称
model_name = "gpt2"

# 加载 Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 加载预训练模型
model = AutoModelForCausalLM.from_pretrained(model_name)

# 输入文本
input_text = "Once upon a time"

# 编码输入
inputs = tokenizer(input_text, return_tensors="pt")

# 生成文本
outputs = model.generate(**inputs, max_length=50, do_sample=True, top_p=0.95, temperature=0.7)

# 解码生成的文本
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
```



```python
#########通过上述代码下载过一次预训练模型到本地后，可直接使用本地模型#########
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils.hub import cached_file

cache_path = cached_file("gpt2", "config.json", local_files_only=True)
model_path = "\\".join(cache_path.split("\\")[:-1])

# 加载 Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    local_files_only=True  # 禁用远程下载/验证，核心参数
)

# 加载预训练模型
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    local_files_only=True,  # 禁用远程请求
    low_cpu_mem_usage=True  # 可选：减少CPU内存占用
)

# 输入文本
input_text = "Once upon a time"

# 编码输入
inputs = tokenizer(input_text, return_tensors="pt")

# 生成文本
outputs = model.generate(**inputs, max_length=50, do_sample=True, top_p=0.95, temperature=0.7)

# 解码生成的文本
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
```



`AutoModelForMaskedLM`

```python
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers.utils.hub import cached_file

# 指定模型名字
cache_path = cached_file("bert-base-uncased", "config.json", local_files_only=True)
model_path = "\\".join(cache_path.split("\\")[:-1])

# 加载Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    local_files_only=True
)

# 加载预训练模型
model = AutoModelForMaskedLM.from_pretrained(
    model_path,
    local_files_only=True
)

# 编码输入
input_text = "Zhanhui Zhong is a [MASK]."
inputs = tokenizer(input_text, return_tensors="pt")

# 获取输出
with torch.no_grad():
    outputs = model(**inputs)
    predictions = outputs.logits

# 获取最高得分的预测词
masked_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
predicted_token_id = predictions[0, masked_index].argmax(dim=-1).item()
predicted_token = tokenizer.decode(predicted_token_id)

print(predicted_token)
```

