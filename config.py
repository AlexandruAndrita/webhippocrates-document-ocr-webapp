import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("MODEL")

DPI = int(os.getenv("DPI"))
MAX_PAGES = int(os.getenv("MAX_PAGES"))

REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT"))
OPENAI_TIMEOUT = int(os.getenv("OPENAI_TIMEOUT"))
DOCUMENT_TIMEOUT = int(os.getenv("DOCUMENT_TIMEOUT"))

MAX_RETRIES = int(os.getenv("MAX_RETRIES"))
RETRY_DELAY = int(os.getenv("RETRY_DELAY"))

BATCH_SIZE = int(os.getenv("BATCH_SIZE"))
MAX_WORKERS = int(os.getenv("MAX_WORKERS"))

# Flask Configuration
# FLASK_PORT = int(os.getenv("PORT", "5000"))
# FLASK_HOST = os.getenv("HOST", "0.0.0.0")
# FLASK_DEBUG = os.getenv("FLASK_DEBUG", "False").lower() == "true"
