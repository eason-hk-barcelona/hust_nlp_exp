import torch
from ltp import LTP
import os

# 获取当前脚本所在目录的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 使用本地下载好的模型路径
model_path = os.path.join(current_dir, "base1")
print(f"加载本地模型: {model_path}")

ltp = LTP(model_path) 

if torch.cuda.is_available():
    print("检测到CUDA，使用GPU加速")
    ltp.to("cuda")

input_file = 'data/test_final.txt'
output_file = 'cws_result.txt'

with open(input_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()
print(f"已读取{len(lines)}行文本")

with open(output_file, 'w', encoding='utf-8') as f_out:
    print("开始进行分词...")
    for i, line in enumerate(lines):
        if (i+1) % 10 == 0:
            print(f"已处理 {i+1}/{len(lines)} 行...")
            
        line = line.strip()
        if not line:
            f_out.write('\n')
            continue
        
        # 使用LTP进行分词
        output = ltp.pipeline([line], tasks=["cws"])
        words = output.cws[0]  # 获取分词结果列表
        
        # 将分词结果用空格连接并写入文件
        f_out.write(' '.join(words) + '\n')

print('分词完成，输出到', output_file)