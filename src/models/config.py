import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

AUGMENTED_TRAIN_DATA_PATH = os.path.join(BASE_DIR, "data", "augmented_train_data.csv")
TRAIN_DATA_PATH = AUGMENTED_TRAIN_DATA_PATH if os.path.exists(AUGMENTED_TRAIN_DATA_PATH) else os.path.join(BASE_DIR, "data", "train_data.csv")
TEST_DATA_PATH = os.path.join(BASE_DIR, "data", "test_data.csv")

MODEL_NAME_DISTILBERT = "distilbert-base-uncased"
MODEL_OUTPUT_DIR_DISTILBERT = os.path.join(BASE_DIR, "phishing_model")
MODEL_NAME = MODEL_NAME_DISTILBERT
MODEL_OUTPUT_DIR = MODEL_OUTPUT_DIR_DISTILBERT

LOG_DIR = os.path.join(BASE_DIR, "logs")
PLOT_DIR = os.path.join(BASE_DIR, "logs", "plots")
EXPERIMENTS_DIR = os.path.join(LOG_DIR, "experiments")
EVAL_DIR = os.path.join(LOG_DIR, "evaluation")
PROGRESS_FILE = os.path.join(LOG_DIR, "training_progress.json")

NUM_EPOCHS = 1
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
MAX_SEQ_LENGTH = 128
WARMUP_RATIO = 0.06
LR_SCHEDULER_TYPE = "cosine"
LOGGING_STEPS = 50
SAVE_STEPS = 200
MAX_TRAIN_SAMPLES = None
FIXED_EVAL_SAMPLES = 5000
RANDOM_SEED = 42
GRADIENT_ACCUMULATION_STEPS = 1
NUM_WORKERS = 0
FP16 = os.environ.get("FP16", "0") == "1"

MIN_RECALL_PHISHING = float(os.environ.get("MIN_RECALL_PHISHING", "0.85"))
MIN_RECALL_MALWARE = float(os.environ.get("MIN_RECALL_MALWARE", "0.80"))
MIN_RECALL_DEFACEMENT = float(os.environ.get("MIN_RECALL_DEFACEMENT", "0.75"))
QUALITY_GATES = {
    1: MIN_RECALL_PHISHING,
    2: MIN_RECALL_MALWARE,
    3: MIN_RECALL_DEFACEMENT,
}

TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE = (2, 4)

GENERATE_PLOTS = True
SAVE_TRAINER_STATE = True

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)
