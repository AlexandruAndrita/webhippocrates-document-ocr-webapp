import base64
import requests
from datetime import datetime
from io import BytesIO
from pathlib import Path
from PIL import Image
from config import REQUEST_TIMEOUT


def img_to_data_url(img: Image.Image) -> str:
    """Convert PIL Image to base64 data URL."""
    buf = BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def parse_date(date_time_string):
    if not date_time_string:
        return datetime.min
        
    date_formats = (
        "%d.%m.%Y",     # 08.10.2022
        "%d %b %Y",     # 08 Oct 2022
        "%d %B %Y",     # 08 October 2022
        "%b %d, %Y",    # Oct 08, 2022
        "%B %d, %Y",    # October 08, 2022
        "%Y-%m-%d",     # 2022-10-08
        "%d/%m/%Y",     # 08/10/2022
        "%m/%d/%Y",     # 10/08/2022
        "%d-%m-%Y",     # 08-10-2022
        "%Y-%m-%d"      # 2022-10-08
    )
    
    for date_format in date_formats:
        try:
            return datetime.strptime(date_time_string.strip(), date_format)
        except ValueError:
            continue
    
    return datetime.min


def fetch_document_links(api_url: str) -> list[str]:
    response = requests.get(api_url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    data = response.json()
    documents = data.get("documents", [])
    links = [doc["document_url"] for doc in documents if "document_url" in doc]
    return links


def download_file_from_url(url: str) -> BytesIO:
    """Download file from URL and return as BytesIO object."""
    response = requests.get(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return BytesIO(response.content)
