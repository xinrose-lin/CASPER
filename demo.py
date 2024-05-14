
from utils.modelUtils import *
from utils.utils import *
import seaborn as sns
import torch
import numpy as np
import matplotlib.pyplot as plt
from casper import nethook


from utils.utils import load_conversation_template
template_name = 'llama-2'
conv_template = load_conversation_template(template_name)


### load model - llama 2 7b

# model_name ="/root/autodl-tmp/base/model"  # or "Llama2-7B" or "EleutherAI/gpt-neox-20b"
model_name = "/root/autodl-tmp/llama2/base/model"  


mt = ModelAndTokenizer(
    model_name,
    low_cpu_mem_usage=True,
    torch_dtype=(torch.float16 if "20b" in model_name else None),
    device = 'cuda:0'
)
print('model', mt.model)

### 
test_prompt = generate_input(conv_template,"What is your name?")

### predict next token
predict_token(
    mt,
    [test_prompt],
    return_p=True,
)

print('test prompt', test_prompt)

output = generate_outputs(test_prompt,mt,)
print('output', output)
print('end of script')
### CASPER ###


### functions: casper analysis for layer 

# def trace_with_patch_layer(
#     model,  # The model
#     inp,  # A set of inputs
#     states_to_patch,  # A list of (token index, layername) triples to restore
#     answers_t,  # Answer probabilities to collect  
# ):
#     prng = np.random.RandomState(1)  # For reproducibility, use pseudorandom noise
#     layers = [states_to_patch[0], states_to_patch[1]]

#     # Create dictionary to store intermediate results
#     inter_results = {}

#     def untuple(x):
#         return x[0] if isinstance(x, tuple) else x

#     # Define the model-patching rule.
#     def patch_rep(x, layer):
#         if layer not in layers:
#             return x

#         if layer == layers[0]:
#             inter_results["hidden_states"] = x[0].cpu()
#             inter_results["attention_mask"] = x[1][0].cpu()
#             inter_results["position_ids"] = x[1][1].cpu()
#             return x
#         elif layer == layers[1]:
#             short_cut_1 = inter_results["hidden_states"].cuda()
#             short_cut_2_1 = inter_results["attention_mask"].cuda()
#             short_cut_2_2 = inter_results["position_ids"].cuda()
#             short_cut_2 = (short_cut_2_1, short_cut_2_2)
#             short_cut = (short_cut_1, short_cut_2)
#             return short_cut
            
#     with torch.no_grad(), nethook.TraceDict(
#         model,
#         layers,
#         edit_output=patch_rep,
#     ) as td:
#         outputs_exp = model(**inp)

#     probs = torch.softmax(outputs_exp.logits[1:, -1, :], dim=1).mean(dim=0)[answers_t]

#     return probs