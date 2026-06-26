import torch
from model import BigramLanguageModel

allowed = sorted(set(
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    " \n\t"
    ".,;:!?()-–—'\"/°%+"
    "αβγδεζθκλμπστψω"
    "µ²³±≤≥"
))
stoi = { ch:i for i,ch in enumerate(allowed) }
itos = { i:ch for i,ch in enumerate(allowed) }
encode = lambda s: [stoi[c] for c in s if c in stoi]
decode = lambda l: ''.join([itos[i] for i in l])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = BigramLanguageModel()
model.load_state_dict(torch.load('pathology_gpt.pt', map_location=device))
model.eval()
model.to(device)
print("Model loaded!")

def generate(prompt, max_new_tokens=500):
    context = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    return decode(model.generate(context, max_new_tokens=max_new_tokens)[0].tolist())

while True:
    prompt = input("\nEnter prompt (or 'quit'): ")
    if prompt.lower() == 'quit':
        break
    print("\n" + generate(prompt))