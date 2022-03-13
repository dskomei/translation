import io
from collections import Counter
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab
from torchtext.utils import download_from_url, extract_archive


def read_texts(filepaths):
    texts = []
    for filepath in filepaths:
        with io.open(filepath, encoding="utf8") as file:
            for text in file:
                texts.append(text)
    return texts
    

def build_vocab(filepath, tokenizer):
    
    counter = Counter()
    with io.open(filepath, encoding="utf8") as f:
        for string_ in f:
            counter.update(tokenizer(string_))
    return Vocab(counter, specials=['<unk>', '<pad>', '<start>', '<end>'])


def data_process(filepaths, vocab_src, vocab_tgt, tokenizer_src, tokenizer_tgt):
    
    raw_de_iter = iter(io.open(filepaths[0], encoding="utf8"))
    raw_en_iter = iter(io.open(filepaths[1], encoding="utf8"))
    
    data = []
    for (raw_de, raw_en) in zip(raw_de_iter, raw_en_iter):
        de_tensor_ = torch.tensor(
            [vocab_src[token] for token in tokenizer_src(raw_de.rstrip("\n"))], dtype=torch.long
        )
        en_tensor_ = torch.tensor(
            [vocab_tgt[token] for token in tokenizer_tgt(raw_en.rstrip("\n"))], dtype=torch.long
        )
        data.append((de_tensor_, en_tensor_))
        
    return data


url_base = 'https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/'
train_urls = ('train.de.gz', 'train.en.gz')
val_urls = ('val.de.gz', 'val.en.gz')

train_filepaths = [extract_archive(download_from_url(url_base + url))[0] for url in train_urls]
valid_filepaths = [extract_archive(download_from_url(url_base + url))[0] for url in val_urls]

tokenizer_src = get_tokenizer('spacy', language='de_core_news_sm')
tokenizer_tgt = get_tokenizer('spacy', language='en_core_web_sm')

vocab_src = build_vocab(train_filepaths[0], tokenizer_src)
vocab_tgt = build_vocab(train_filepaths[1], tokenizer_tgt)

train_data = data_process(train_filepaths, vocab_src, vocab_tgt, tokenizer_src, tokenizer_tgt)
val_data = data_process(valid_filepaths, vocab_src, vocab_tgt, tokenizer_src, tokenizer_tgt)

batch_size = 128
PAD_IDX = vocab_src['<pad>']
START_IDX = vocab_src['<start>']
END_IDX = vocab_src['<end>']


def generate_batch(data_batch):
    
    batch_src, batch_tgt = [], []
    for src, tgt in data_batch:
        batch_src.append(torch.cat([torch.tensor([START_IDX]), src, torch.tensor([END_IDX])], dim=0))
        batch_tgt.append(torch.cat([torch.tensor([START_IDX]), tgt, torch.tensor([END_IDX])], dim=0))
        
    batch_src = pad_sequence(batch_src, padding_value=PAD_IDX)
    batch_tgt = pad_sequence(batch_tgt, padding_value=PAD_IDX)
    
    return batch_src, batch_tgt


train_iter = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=generate_batch)
valid_iter = DataLoader(val_data, batch_size=batch_size, shuffle=True, collate_fn=generate_batch)
