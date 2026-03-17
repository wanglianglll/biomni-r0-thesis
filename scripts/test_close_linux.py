import transformers, inspect
from transformers import TrainingArguments

print("transformers version:", transformers.__version__)
print("TrainingArguments init signature:")
print(inspect.signature(TrainingArguments.__init__))
# 或打印参数名列表：
print(TrainingArguments.__init__.__code__.co_varnames)
# 也可以打印文件位置确认来源
import inspect
print(inspect.getfile(TrainingArguments))