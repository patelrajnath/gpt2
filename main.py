from transformers import GPT2LMHeadModel
import matplotlib.pyplot as plt
from transformers import pipeline, set_seed

model_hf = GPT2LMHeadModel.from_pretrained('gpt2')
sd = model_hf.state_dict()


for k, v in sd.items():
    print(k, v.size())

print(sd['transformer.wpe.weight'].view(-1)[:20])
plt.imshow(sd['transformer.wpe.weight'], cmap='gray')
plt.show()

set_seed(42)
generator = pipeline('text-generation', 'gpt2')
print(generator("Hello, I'm a language model", max_length=30, num_return_sequences=5))


