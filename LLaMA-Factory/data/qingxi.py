import json
import os
import re

def clean_image_tag(content):
    """
    清理和统一<image>标签格式
    将<image>\n、\n<image>等格式统一为<image>并放在最前面
    """
    # 移除所有换行符
    content = content.replace('\n', '')
    
    # 查找所有<image>标签的位置
    image_pattern = r'<image>'
    matches = list(re.finditer(image_pattern, content))
    
    if matches:
        # 如果有<image>标签，将其移到最前面
        # 先移除所有<image>标签
        content_without_image = re.sub(image_pattern, '', content)
        # 在开头添加<image>
        content = '<image>' + content_without_image.strip()
    
    return content

def convert_json_format(input_file, output_file):
    """
    将原始JSON格式转换为新的格式，并统一处理<image>标签
    
    原始格式:
    [
        {
            "id": 1,
            "image": "wikidata_imgs/1900 Storm Memorial_wiki.jpg",
            "conversations": [
                {
                    "from": "human",
                    "value": "<image>\nCan you describe the visual artwork depicted in the image?"
                },
                {
                    "from": "gpt", 
                    "value": "The 1900 Storm Memorial is a ten-foot tall bronze sculpture..."
                }
            ]
        }
    ]
    
    目标格式:
    {
        "messages": [
            {
                "content": "<image>Who is he?",
                "role": "user"
            },
            {
                "content": "He's Thomas Muller from Bayern Munich.",
                "role": "assistant"
            }
        ],
        "images": [
            "/home/bingxing2/ailab/group/ai4bio/renyuchen/jkl/lora_null/mllm_demo_data/2.jpg"
        ]
    }
    """
    
    # 图像路径前缀
    image_prefix = "/hkfs/work/workspace/scratch/lmu_chd4938-MINED_26/MINED_26/data_ckpt/dataset/lora_null/"
    
    # 读取原始JSON文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    converted_data = []
    
    for item in data:
        # 提取图像路径并添加前缀
        image_path = item.get('image', '')
        if image_path:
            # 如果路径不是绝对路径，则添加前缀
            if not image_path.startswith('/'):
                image_path = image_prefix + image_path
            else:
                # 如果已经是绝对路径，直接使用
                pass
        
        # 转换对话格式
        messages = []
        conversations = item.get('conversations', [])
        
        for conv in conversations:
            role = conv.get('from', '')
            content = conv.get('value', '')
            
            # 将角色名称映射到新格式
            if role == 'human':
                role = 'human'
                # 对human角色的content进行<image>标签清理
                content = clean_image_tag(content)
            elif role == 'gpt':
                role = 'gpt'
            
            messages.append({
                "form": content,
                "value": role
            })
        
        # 创建新的数据格式
        converted_item = {
            "conversations": conversations,
            "images": [image_path] if image_path else []
        }
        
        converted_data.append(converted_item)
    
    # 保存转换后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)
    
    print(f"转换完成！")
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print(f"转换了 {len(converted_data)} 条记录")
    print(f"图像路径前缀: {image_prefix}")

def main():
    # 设置输入和输出文件路径
    input_file = "/hkfs/work/workspace/scratch/lmu_chd4938-MINED_26/LLaVA/10_random_reply_training_data.json"  # 请替换为你的输入文件名
    output_file = "/hkfs/work/workspace/scratch/lmu_chd4938-MINED_26/Qwen2.5-VL-null-space/LLaMA-Factory/data/new_10_random_reply_training_data.json"  # 输出文件名
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误：输入文件 '{input_file}' 不存在！")
        print("请确保输入文件在当前目录下，或修改 input_file 变量为正确的文件路径。")
        return
    
    # 执行转换
    convert_json_format(input_file, output_file)

if __name__ == "__main__":
    main()