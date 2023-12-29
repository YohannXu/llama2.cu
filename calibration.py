import operator
import torch
import torch.nn as nn
from pathlib import Path
import json
from torch.fx import GraphModule
import argparse

from models.llama2 import Llama
from models.tokenizer import Tokenizer
from utils.llama.rotary_emb import precompute_rotary_emb
from utils.llama.mask import get_mask
from utils.llama.sample import sample_top_p
from models.llama2 import RMSNorm, RotaryEmb, Score
from utils.quantize.modules import QLinear, QRMSNorm
from utils.quantize.tracer import CustomTracer
from utils.quantize.convert import convert
from utils.quantize.state import enable_calibration
from utils.quantize.export import export_onnx

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt-dir', type=str, default='weights/llama-7b-chat')
args = parser.parse_args()
ckpt_dir = args.ckpt_dir

torch.cuda.set_device(0)
torch.manual_seed(1)
torch.set_default_tensor_type(torch.cuda.HalfTensor)

params = {
    'dim': 1024,
    'n_layers': 4,
    'n_heads': 4,
    'vocab_size': -1,
    'multiple_of': 256,
    'norm_eps': 1e-5,
}

with open(Path(ckpt_dir) / 'params.json') as f:
    params.update(json.load(f))

tokenizer = Tokenizer((Path(ckpt_dir) / 'tokenizer.model').as_posix())
params['vocab_size'] = tokenizer.n_words

max_seq_len = 112
temperature = 0.6
top_p = 0.9

freqs_cos, freqs_sin = precompute_rotary_emb(params['dim'] // params['n_heads'], max_seq_len * 2)

embedding = nn.Embedding(params['vocab_size'], params['dim'])

llama = Llama(**params)

ckpt = sorted(Path(ckpt_dir).glob('*.pth'))[0]
state_dict = torch.load(ckpt, map_location='cpu')
embedding_state_dict = {'weight': state_dict.pop('tok_embeddings.weight')}
embedding.load_state_dict(embedding_state_dict, strict=False)
llama.load_state_dict(state_dict, strict=False)
llama.eval()

del embedding_state_dict
del state_dict

mappings = {
    nn.Linear: QLinear,
    RMSNorm: QRMSNorm,
}
quant_functions = [
    torch.cat,
    torch.matmul,
    operator.mul,
    operator.add
]
quant_modules = [
    QLinear,
    QRMSNorm,
    RotaryEmb,
    Score,
    nn.SiLU
]
tracer = CustomTracer([nn.Linear, nn.SiLU, RMSNorm, RotaryEmb, Score])
graph = tracer.trace(llama)
llama = GraphModule(llama, graph, llama.__class__.__name__)

input_scale = embedding.weight.abs().max() / 127
freq_scale = torch.tensor([2 / 255.], dtype=torch.half)

llama = convert(llama, mappings, quant_functions, quant_modules, input_scale, freq_scale)

enable_calibration(llama)

prompts = [
    "Hello, who are you?",

    # linguistics
    "What is a noun?",
    "What is a verb?",
    "What is an adjective?",
    "What is an adverb?",
    "What is a preposition?",
    "What is a conjunction?",
    "What is a pronoun?",
    "What is a simile?",
    "What is a metaphor?",
    "What is a synonym?",

    # math
    "What is addition?",
    "What is subtraction?",
    "What is multiplication?",
    "What is division?",
    "What is an integer?",
    "What is a decimal?",
    "What is a fraction?",
    "What is a percentage?",
    "What is a square?",
    "What is a cube?",

    # physical
    "What is gravity?",
    "What is velocity?",
    "What is acceleration?",
    "What is the law of conservation of energy?",
    "What is the difference between speed and velocity?",
    "What is Newton's first law of motion?",
    "What is the formula for calculating force?",
    "What is the concept of momentum?",
    "What is the principle of conservation of momentum?",
    "What is the difference between potential energy and kinetic energy?",

    # chemical
    "What is an element?",
    "What is a compound?",
    "What is an ion?",
    "What is an acid?",
    "What is a base?",
    "What is pH?",
    "What is a redox reaction?",
    "What is a chemical bond?",
    "What is a molecule?",
    "What is a chemical reaction?",

    # biology
    "What is a cell?",
    "What is DNA?",
    "What is a gene?",
    "What is cell division?",
    "What is evolution?",
    "What is photosynthesis?",
    "What is respiration?",
    "What is a cell membrane?",
    "What is genetic variation?",
    "What is an ecosystem?",

    # geography
    "What is the Earth?",
    "What are longitude and latitude?",
    "What is a continent?",
    "What is an ocean?",
    "What is a river?",
    "What is a mountain range?",
    "What is a lake?",
    "What is an island?",
    "What is climate?",
    "What are natural disasters?",

    # common sense
    "What is the largest organ in the human body?",
    "Which is the largest continent on Earth?",
    "Where is the capital city located?",
    "What is the solar system?",
    "How many teeth does an adult human have?",
    "What is the longest river in the world?",
    "What is cocoa?",
    "What makes up the human genome?",
    "What is oxygen?",
    "Is the moon a solid or a gas?"

]

for prompt in prompts:
    print("question:", prompt)
    prompt = f"[INST] {prompt} [/INST]"

    token = tokenizer.encode(prompt, bos=True, eos=False)
    token = torch.tensor(token, dtype=torch.long).unsqueeze(0)

    results = []
    k_caches = [torch.randn(1, 0, params['n_heads'], params['dim'] // params['n_heads']) for _ in range(params['n_layers'])]
    v_caches = [torch.randn(1, 0, params['n_heads'], params['dim'] // params['n_heads']) for _ in range(params['n_layers'])]

    start_pos = 0
    total_len = 0
    while True:
        seq_len = token.shape[1]
        freq_cos = freqs_cos[:, start_pos: start_pos + seq_len]
        freq_sin = freqs_sin[:, start_pos: start_pos + seq_len]
        mask = get_mask(seq_len)

        token_embed = embedding(token)
        logits, k_caches, v_caches = llama(token_embed, k_caches, v_caches, freq_cos, freq_sin, mask)

        probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
        next_token = sample_top_p(probs, top_p)
        token = next_token
        next_token = next_token.reshape(-1)

        total_len += seq_len
        if total_len >= max_seq_len:
            break
        if next_token == tokenizer.eos_id:
            break
        results.append(next_token.item())
        start_pos = start_pos + seq_len

    del k_caches, v_caches

    result = tokenizer.decode(results)
    print("answer:", result)
    print("")

state_dict = llama.state_dict()
state_dict['tok_embeddings.weight'] = embedding.weight.detach()
torch.save(state_dict, 'llama-7b-chat-quant.pth')
export_onnx(llama, 'llama-7b-chat-quant.onnx')
