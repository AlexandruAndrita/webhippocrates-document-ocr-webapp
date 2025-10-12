import pypdfium2 as pdfium
from io import BytesIO
from pathlib import Path
from PIL import Image
from config import DPI, MAX_PAGES
from utils import img_to_data_url, download_file_from_url


def pdf_to_data_urls(pdf_source, dpi=None, limit=None):
    if dpi is None:
        dpi = DPI
    if limit is None:
        limit = MAX_PAGES
        
    scale = float(dpi) / 72.0

    # Load the document from path or memory
    if isinstance(pdf_source, (str, Path)):
        pdf = pdfium.PdfDocument(str(pdf_source))
    elif isinstance(pdf_source, BytesIO):
        pdf = pdfium.PdfDocument(pdf_source.getvalue())
    elif isinstance(pdf_source, (bytes, bytearray)):
        pdf = pdfium.PdfDocument(pdf_source)
    else:
        raise TypeError("pdf_source must be a path (str/PathLike), BytesIO, or bytes")

    total_pages = len(pdf)
    pages_to_render = total_pages if limit is None else min(limit, total_pages)

    data_urls = []
    for i in range(pages_to_render):
        page = pdf[i]
        bitmap = page.render(scale=scale)
        pil_img = bitmap.to_pil().convert("RGB")
        data_urls.append(img_to_data_url(pil_img))

    return data_urls


def process_pdf_document(path: str) -> list[str]:
    document_name = Path(path).name
    
    if path.startswith("http"):
        print(f"[{document_name}] Downloading PDF from URL...")
        temp_pdf = download_file_from_url(path)
        print(f"[{document_name}] PDF downloaded, converting to images...")
        data_urls = pdf_to_data_urls(temp_pdf, dpi=DPI, limit=MAX_PAGES)
    else:
        print(f"[{document_name}] Converting local PDF to images...")
        data_urls = pdf_to_data_urls(str(path), dpi=DPI, limit=MAX_PAGES)
        
    if not data_urls:
        raise RuntimeError("PDF to image conversion failed or document has no pages.")
    
    print(f"[{document_name}] Converted to {len(data_urls)} images")
    return data_urls


def process_image_document(path: str) -> str:
    document_name = Path(path).name
    
    if path.startswith("http"):
        print(f"[{document_name}] Downloading image from URL...")
        img_data = download_file_from_url(path)
        img = Image.open(img_data).convert("RGB")
    else:
        print(f"[{document_name}] Loading local image...")
        img = Image.open(path).convert("RGB")
        
    print(f"[{document_name}] Image loaded")
    return img_to_data_url(img)


def get_document_type(path: str) -> str:
    path_lower = path.lower()
    
    if path_lower.endswith(".pdf"):
        return "pdf"
    elif path_lower.endswith((".png", ".jpg", ".jpeg")):
        return "image"
    else:
        return "unknown"