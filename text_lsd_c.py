# ============================================================
# TextLSD-C: LSD-C for Text Clustering (Reuters-10K + 20News)
# Paper: https://arxiv.org/abs/2006.10039
# ============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForMaskedLM, BertModel
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import normalized_mutual_info_score as nmi_score
import numpy as np
import os
import argparse
import random
from tqdm import tqdm
import pickle
from utils_algo import PairEnum, BCE_softlabels, sigmoid_rampup, cluster_acc

# ============================================================
# 1. Dataset: Reuters-10K + 20 Newsgroups
# ============================================================
class TextDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx])
        item['idx'] = idx
        return item

# Load Reuters-10K (from original repo)
def load_reuters10k():
    # Download from: https://github.com/srebuffi/lsd-clusters
    # We use preprocessed version
    with open('reuters10k.pkl', 'rb') as f:
        data = pickle.load(f)
    return data['texts'], data['labels']

# Load 20 Newsgroups
def load_20newsgroups():
    data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    return data.data, data.target

# ============================================================
# 2. Text RICAP (cắt ghép câu)
# ============================================================
def text_ricap(batch, beta=0.3):
    texts = [batch[i]['input_ids'] for i in range(len(batch))]
    attn_masks = [batch[i]['attention_mask'] for i in range(len(batch))]
    B = len(texts)
    I_x = texts[0].size(0)  # seq len

    w = int(np.round(I_x * np.random.beta(beta, beta)))
    h = int(np.round(I_x * np.random.beta(beta, beta)))
    w_ = [w, I_x - w, w, I_x - w]
    h_ = [h, h, I_x - h, I_x - h]

    cropped = {}
    W = {}
    idxs = {}
    for k in range(4):
        x_k = np.random.randint(0, I_x - w_[k] + 1)
        y_k = np.random.randint(0, I_x - h_[k] + 1)
        if w_[k] * h_[k] > (I_x * I_x / 4):
            idxs[k] = list(range(B))
        else:
            idxs[k] = np.random.permutation(B).tolist()
        cropped[k] = torch.stack([texts[i][x_k:x_k+w_[k]] for i in idxs[k]])
        W[k] = w_[k] * h_[k] / (I_x * I_x)

    # Reconstruct
    top = torch.cat([cropped[0], cropped[1]], dim=1)
    bot = torch.cat([cropped[2], cropped[3]], dim=1)
    mixed = torch.cat([top, bot], dim=1)
    mixed = mixed[:, :I_x]  # crop to max_len
    mixed_mask = torch.ones_like(mixed)

    return mixed, mixed_mask, W, idxs

# ============================================================
# 3. Model: BERT + Linear Head
# ============================================================
class BertCluster(nn.Module):
    def __init__(self, num_classes=10, bert_model='bert-base-uncased'):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model)
        self.linear = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0]  # [CLS]
        logits = self.linear(cls_emb)
        return logits, cls_emb

# ============================================================
# 4. Pretrain: Masked Language Modeling
# ============================================================
def pretrain_bert_mlm(train_texts, epochs=3, batch_size=32):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)

    dataset = TextDataset(train_texts, tokenizer=tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tokenizer.pad(x, return_tensors='pt'))

    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(loader, desc=f"MLM Pretrain [Epoch {epoch+1}]"):
            inputs = {k: v for k, v in batch.items() if k in ['input_ids', 'attention_mask', 'labels']}
            outputs = model(**inputs)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        print(f"MLM Loss: {total_loss/len(loader):.4f}")

    # Save backbone
    torch.save(model.bert.state_dict(), 'bert_mlm_pretrain.pt')
    return tokenizer

# ============================================================
# 5. LSD-C Training Loop
# ============================================================
def train_lsd_c(args, model, device, train_loader, optimizer, epoch, tokenizer):
    model.train()
    bce = BCE_softlabels()
    w_cons = args.rampup_coefficient * sigmoid_rampup(epoch, args.rampup_length)
    total_loss = 0

    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(device)
        attn_mask = batch['attention_mask'].to(device)

        # Text RICAP
        mixed_ids, mixed_mask, W, idxs = text_ricap(batch)
        mixed_ids, mixed_mask = mixed_ids.to(device), mixed_mask.to(device)

        optimizer.zero_grad()
        logits, emb = model(input_ids, attn_mask)
        logits_bar, _ = model(mixed_ids, mixed_mask)

        prob = F.softmax(logits, dim=1)
        prob_bar = F.softmax(logits_bar, dim=1)

        # kNN similarity in [CLS] space
        emb_norm = emb / emb.norm(dim=1, keepdim=True)
        cosine = torch.mm(emb_norm, emb_norm.t())
        target_ulb = (cosine > args.hyperparam).float()

        # Adapt RICAP labels
        target_ulb_ricap = torch.zeros_like(target_ulb)
        for k in range(4):
            if isinstance(idxs[k], str):
                target_ulb_ricap += W[k] * target_ulb
            else:
                target_ulb_ricap += W[k] * target_ulb[:, idxs[k]].diagonal().view(-1, 1)

        # Loss
        prob_row, _ = PairEnum(prob)
        _, prob_bar_col = PairEnum(prob_bar)
        loss_clus = bce(prob_row, prob_bar_col, target_ulb_ricap.view(-1))
        loss_cons = F.mse_loss(prob_bar, prob_bar.detach())
        loss = loss_clus + w_cons * loss_cons

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)

# ============================================================
# 6. Evaluation
# ============================================================
def evaluate(model, device, loader):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attn_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].numpy() if 'labels' in batch else None
            _, emb = model(input_ids, attn_mask)
            pred = emb.cpu().numpy()
            preds.append(pred)
            if labels is not None:
                trues.append(labels)
    preds = np.concatenate(preds)
    if trues:
        trues = np.concatenate(trues)
        acc = cluster_acc(trues, preds.argmax(1))
        nmi = nmi_score(trues, preds.argmax(1))
        return acc, nmi
    return preds

# ============================================================
# 7. Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='reuters10k', choices=['reuters10k', '20news'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--rampup_length', type=int, default=50)
    parser.add_argument('--rampup_coefficient', type=float, default=5.0)
    parser.add_argument('--hyperparam', type=float, default=0.9)  # cosine threshold
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--pretrain', action='store_true')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Load data
    if args.dataset == 'reuters10k':
        texts, labels = load_reuters10k()
        num_classes = 10
    else:
        texts, labels = load_20newsgroups()
        num_classes = 20

    # Pretrain MLM
    if args.pretrain:
        print("Pretraining BERT with MLM...")
        pretrain_bert_mlm(texts, epochs=3)

    # Load pretrained BERT
    model = BertCluster(num_classes=num_classes).to(device)
    if os.path.exists('bert_mlm_pretrain.pt'):
        state_dict = torch.load('bert_mlm_pretrain.pt', map_location=device)
        model.bert.load_state_dict(state_dict)

    # Freeze BERT except last layer
    for name, param in model.bert.named_parameters():
        if 'layer.11' not in name and 'layer.10' not in name:
            param.requires_grad = False

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                          lr=0.01, momentum=0.9, weight_decay=5e-4)

    # Dataset
    dataset = TextDataset(texts, labels, tokenizer)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        collate_fn=lambda x: tokenizer.pad(x, return_tensors='pt'))

    # Train
    for epoch in range(args.epochs):
        loss = train_lsd_c(args, model, device, loader, optimizer, epoch, tokenizer)
        print(f"Epoch {epoch}: Loss {loss:.4f}")

    # Evaluate
    acc, nmi = evaluate(model, device, loader)
    print(f"Final ACC: {acc:.4f}, NMI: {nmi:.4f}")

if __name__ == '__main__':
    main()
