#transformer question
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data import Field, BucketIterator, TabularDataset
from transformers import MarianMTModel, MarianTokenizer

# Download and load the MarianMT model and tokenizer
model_name = 'Helsinki-NLP/opus-mt-en-any'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Set device (cuda if available, else cpu)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define Fields for source and target languages
SRC = Field(tokenize="spacy", tokenizer_language="en", init_token="<sos>", eos_token="<eos>")
TRG = Field(tokenize="spacy", tokenizer_language="xx_XX", init_token="<sos>", eos_token="<eos>")

# Define TabularDataset for your data
data_fields = [('source', SRC), ('target', TRG)]
train_data, valid_data, test_data = TabularDataset.splits(
    path='path/to/dataset',
    train='train.csv',
    validation='valid.csv',
    test='test.csv',
    format='csv',
    fields=data_fields
)

# Build vocabulary
SRC.build_vocab(train_data, min_freq=2)
TRG.build_vocab(train_data, min_freq=2)

# Define BucketIterator
train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=64,
    device=device
)

# Define Transformer model
class Transformer(nn.Module):
    def _init_(self, src_vocab_size, trg_vocab_size, model, device, max_len=100):
        super()._init_()

        self.src_pad_idx = SRC.vocab.stoi[SRC.pad_token]
        self.trg_pad_idx = TRG.vocab.stoi[TRG.pad_token]

        self.encoder = model.get_encoder()
        self.decoder = model.get_decoder()

        self.fc_out = nn.Linear(model.config.d_model, trg_vocab_size)
        self.dropout = nn.Dropout(0.1)

        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        self.device = device
        self.max_len = max_len

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        trg_len = trg.shape[1]

        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()

        trg_mask = trg_pad_mask & trg_sub_mask

        return trg_mask

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        enc_src = self.encoder(src, attention_mask=src_mask)
        dec_src = self.decoder(trg, attention_mask=trg_mask, encoder_hidden_states=enc_src['last_hidden_state'])

        output = self.fc_out(dec_src['last_hidden_state'])

        return output


# Instantiate the model
src_vocab_size = len(SRC.vocab)
trg_vocab_size = len(TRG.vocab)
model = Transformer(src_vocab_size, trg_vocab_size, model, device).to(device)

# Define optimizer and criterion
optimizer = optim.Adam(model.parameters(), lr=0.0005)
criterion = nn.CrossEntropyLoss(ignore_index=TRG.vocab.stoi[TRG.pad_token])

# Training loop
def train(model, iterator, optimizer, criterion, clip):
    model.train()

    for batch in iterator:
        src = batch.source
        trg = batch.target

        optimizer.zero_grad()

        output = model(src, trg[:, :-1])
        output_dim = output.shape[-1]

        output = output.contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)

        loss = criterion(output, trg)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

# Training the model for a number of epochs
N_EPOCHS = 10
CLIP = 1

for epoch in range(N_EPOCHS):
    train(model, train_iterator, optimizer, criterion, CLIP)

# Translation function
def translate_sentence(sentence, src_field, trg_field, model, device, max_len=100):
    model.eval()

    if isinstance(sentence, str):
        tokens = [token.text.lower() for token in spacy_en(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    tokens = [src_field.init_token] + tokens + [src_field.eos_token]
    src_indexes = [src_field.vocab.stoi[token] for token in tokens]

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
    src_mask = model.make_src_mask(src_tensor)

    with torch.no_grad():
        enc_src = model.encoder(src_tensor, attention_mask=src_mask)

    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

    for i in range(max_len):
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

        trg_mask = model.make_trg_mask(trg_tensor)

        with torch.no_grad():
            output = model.decoder(trg_tensor, attention_mask=trg_mask, encoder_hidden_states=enc_src['last_hidden_state'])

        pred_token = output.argmax(2)[:, -1].item()
        trg_indexes.append(pred_token)

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break

    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
    return trg_tokens[1:]

# Example translation
example_sentence = "Translate this sentence to an Indian language."
translated_sentence = translate_sentence(example_sentence, SRC, TRG, model, device)

print(f"Input: {example_sentence}")
print(f"Translation: {' '.join(translated_sentence)}")