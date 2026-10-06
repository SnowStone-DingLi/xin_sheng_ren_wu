"""
任务7：大模型应用 - 使用 Qwen2.5-1.5B-Instruct 替代 ChatGLM-6B 做 zero-shot 文本分类。
演示通过 prompt 让大模型完成文本分类任务（in-context learning）。
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = r'D:\my_models\Qwen2.5-1.5B-Instruct'

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, trust_remote_code=True,
    torch_dtype=torch.float32, device_map='cpu'
)

class_examples = {
    '人物': '岳云鹏，本名岳龙刚，1985年4月15日出生于河南省濮阳市南乐县，中国内地相声、影视男演员。2005年，首次登台演出。',
    '书籍': '《三体》是刘慈欣创作的长篇科幻小说系列，由《三体》《三体2：黑暗森林》《三体3：死神永生》组成。',
    '电视剧': '《狂飙》是由中央电视台、爱奇艺出品，徐纪周执导，张译、张颂文领衔主演的反黑刑侦剧。',
    '电影': '《流浪地球》是由郭帆执导，吴京特别出演的科幻冒险电影。',
    '城市': '乐山，古称嘉州，四川省辖地级市，位于四川中部，全市总面积12720.03平方公里。',
    '国家': '瑞士联邦，简称"瑞士"，首都伯尔尼，位于欧洲中部，总面积41284平方千米。'
}


def init_prompts():
    class_list = list(class_examples.keys())
    pre_history = []
    system_msg = f'你是一个文本分类器，你需要将给你的句子分类到：{class_list}类别中。只输出类别名称。'
    pre_history.append({'role': 'system', 'content': system_msg})
    for _type, example in class_examples.items():
        pre_history.append({'role': 'user', 'content': f'"{example}" 是 {class_list} 里的什么类别？'})
        pre_history.append({'role': 'assistant', 'content': _type})
    return {'class_list': class_list, 'pre_history': pre_history}


def inference(sentences, custom_settings):
    for sentence in sentences:
        messages = list(custom_settings['pre_history'])
        messages.append({'role': 'user', 'content': f'"{sentence}" 是 {custom_settings["class_list"]} 里的什么类别？'})

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors='pt').to('cpu')
        outputs = model.generate(
            **inputs, max_new_tokens=10, do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
        )
        answer = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        print(f'>>> sentence: {sentence}')
        print(f'>>> inference answer: {answer.strip()}')


if __name__ == '__main__':
    custom_settings = init_prompts()
    sentences = [
        '加拿大（英语/法语：Canada），首都渥太华，位于北美洲北部。',
        '《琅琊榜》是由山东影视传媒集团出品，胡歌、刘涛等主演的古装剧。',
        '《满江红》是由张艺谋执导，沈腾、易烊千玺等主演的悬疑喜剧电影。',
        '布宜诺斯艾利斯是阿根廷共和国的首都和最大城市。',
        '张译（原名张毅），1978年2月17日出生于黑龙江省哈尔滨市，中国内地男演员。',
    ]
    inference(sentences, custom_settings)
