from pathlib import Path
_current_dir = Path(__file__).parent

ROOT_DIR = _current_dir.parent
DATA_DIR = Path(ROOT_DIR, 'data')
MODELS_DIR = Path(ROOT_DIR, 'models')
LOGS_DIR = Path(ROOT_DIR, 'lightning_logs')