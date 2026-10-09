"""全局配置：路径、划分比例、随机种子、模型超参默认值。"""

from pathlib import Path

# 项目根目录（nlp-HW-1/）
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# 优先 data/raw/；若不存在则回退到课程下发目录 作业一/
_NYT_CANDIDATES = (
    DATA_RAW_DIR / "nyt.csv",
    PROJECT_ROOT / "作业一" / "nyt.csv",
)
_AG_CANDIDATES = (
    DATA_RAW_DIR / "ag.csv",
    PROJECT_ROOT / "作业一" / "ag.csv",
)


def _first_existing(paths) -> Path:
    for p in paths:
        if p.exists():
            return p
    return paths[0]


NYT_CSV = _first_existing(_NYT_CANDIDATES)
AG_CSV = _first_existing(_AG_CANDIDATES)

# 数据划分（作业要求）
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
RANDOM_SEED = 42

# Task 2
EMBEDDING_DIM = 100
GLOVE_PATH = EMBEDDINGS_DIR / "glove.6B.100d.txt"

# Task 2 Word2Vec 默认超参
W2V_WINDOW = 5
W2V_MIN_COUNT = 2
W2V_SG = 1
W2V_EPOCHS = 5

# Task 3（作业固定要求）
BERT_MODEL_NAME = "google-bert/bert-base-uncased"
BERT_MAX_LENGTH = 64
BERT_EPOCHS = 3
BERT_BATCH_SIZE = 16
BERT_LR = 2e-5
