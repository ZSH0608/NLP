# 作业一：文本分类实现

基于不同文本表示方法（Bag-of-Words / Word Embedding / BERT）在 NYT 新闻数据上完成文本分类，并比较 Accuracy 与 Macro-F1。


## 项目结构

```text
lab1/
├── README.md                 # 运行说明
├── requirements.txt
├── 实验报告.md
├── data/
│   ├── raw/                  # nyt.csv, ag.csv
│   └── processed/            # train/val/test.csv（80/10/10 划分结果）
├── embeddings/               
├── models/                   # w2v_*.model, bert/final/
├── results/                  
├── scripts/
│   ├── prepare_data.py
│   ├── run_task1.py
│   ├── run_task2.py
│   └── run_task3.py
└── src/
    ├── config.py
    ├── data/
    ├── features/
    ├── models/
    ├── train/
    └── utils/
```

## 环境准备

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

### GloVe（Task 2.1）

代码会通过 gensim 自动加载 `glove-wiki-gigaword-100`。

### BERT（Task 3）

首次运行从 Hugging Face 拉取 `google-bert/bert-base-uncased`。

## 数据说明

| 文件 | 用途 |
|------|------|
| `data/raw/nyt.csv` | 主分类数据，列：`text`, `label` |
| `data/raw/ag.csv` | 训练 Word2Vec，列：`text` |

NYT 划分：`random_state=42`，Train / Val / Test = 80% / 10% / 10%（9215 / 1151 / 1153）。

## 运行步骤

```bash
python scripts/prepare_data.py
python scripts/run_task1.py
python scripts/run_task2.py
python scripts/run_task3.py   
```

### 关键超参

- Logistic Regression：`solver=saga`, `max_iter=3000`, `class_weight=balanced`, `random_state=42`
- Word2Vec：`vector_size=100`, `window=5`, `min_count=2`, `sg=1`, `epochs=5`
- BERT：`max_length=64`, `epochs=3`, `batch_size=16`, `lr=2e-5`（MPS 训练约 19 分钟）

## 评价指标

- Accuracy
- Macro-F1
