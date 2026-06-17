#tokenizer 

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

texts = [ "I love NLP", "Transformers are amazing"]

token = tokenizer(texts,padding = True,max_length=2,return_tensors= "pt")

# print(output)
# print(output["input_ids"].shape)

# embeddings

import torch
import torch.nn as nn

vocab_size = tokenizer.vocab_size
embedding = nn.Embedding(num_embeddings = vocab_size,embedding_dim = 3)


token_id = token['input_ids']
output = embedding(token_id)

# print(output.shape)
# print(output)


from torch.nn.utils.rnn import pad_sequence



s1 = torch.tensor([5,7,2])
s2 = torch.tensor([8,1])
s3 = torch.tensor([3,9,4,6])

padded = pad_sequence([s1,s2,s3],batch_first = True,padding_value = 0)

print(padded)