import base64
import json
from io import BytesIO
from pathlib import Path
import time
import os
import requests
import pypdfium2 as pdfium
# from pdf2image import convert_from_path, convert_from_bytes
from flask import Flask, request, jsonify
from PIL import Image
from openai import OpenAI
from datetime import datetime

OPENAI_API_KEY = os.getenv("OPEN_AI_KEY") 
MODEL = os.getenv("MODEL") 
DPI = int(os.getenv("DPI"))
MAX_PAGES = int(os.getenv("MAX_PAGES"))

client = OpenAI(api_key=OPENAI_API_KEY)
app = Flask(__name__)

def img_to_data_url(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"

# pdf2image + poppler
# def pdf_to_data_urls(pdf_source, dpi: int = 250, limit: int | None = None) -> list[str]:
#     if isinstance(pdf_source, (str, os.PathLike)):
#         # specific file path
#         pages = convert_from_path(pdf_source, dpi=dpi)
#     else:
#         # object in bytes
#         pages = convert_from_bytes(pdf_source.getvalue(), dpi=dpi)
#     if limit is not None:
#         pages = pages[:limit]
#     return [img_to_data_url(p.convert("RGB")) for p in pages]

def pdf_to_data_urls(pdf_source, dpi, limit):
    scale = float(dpi) / 72.0

    # Load the document from path or memory
    if isinstance(pdf_source, (str, os.PathLike)):
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

def call_openai_with_images(image_urls: list[str]) -> dict:
    system_msg = "Ești un extractor de date din documente medicale. Returnează DOAR JSON valid."

    instruction = (
        "Extrage următoarele câmpuri din imaginea de document medical furnizată. "
        "Dacă un câmp lipsește, folosește null. Daca anumite cuvinte cheie nu se regasesc, iar in prompt ti se indica sa folosesti anumite valori e.g. `True` sau `False`, foloseste-le. Nu inventa valori.\n\n"
        "- titlu_document"
        "Tipurile de titluri pot fi 'Scrisoare Medicala', 'Bilet de iesire din spital', 'Bilet de iesire', 'Bilet de externare'. Inafara de acestea, exista si alte titluri care pot aparea in document si trebuie extrase.\n\n"

        "- nume_prenume_pacient"
        "Va aprea dupa urmatoarele cuvinte cheie: 'Nume si prenume', poate aparea dupa 'Pacientul'. Intotdeauna este un nume scris cu majuscule.\n\n"

        "- variabila_booleana_diagnostic_curent"
        "Va returna `True` daca 'titlu_document' contine urmatoarele cuvinte cheie: 'Scrisoare medicala', 'Bilet de iesire din spital', 'Bilet de iesire', 'Bilet de externare'. Altfel, va returna `False`.\n\n"

        "- variabila_booleana_analize_medicale"
        "Va returna `True` daca documentul contine urmatoarele cuvinte cheie: 'hemoglobina', 'hematocrit', 'hemoleucograma'. Altfel, va returna `False`.\n\n"

        "- variabila_booleana_examen_hispotatologic"
        "Va returna `True` daca documentul contine urmatoarele cuvinte cheie: 'histopatologica', 'histopatologic', 'microscopie', 'macroscopie', 'imunohistochimie', 'biopsie', 'biopsic', 'biopsice', 'OncoType', 'examen imunohistochimic', 'IHC', 'EHP'. Altfel, va returna `False`.\n\n"

        "- variabila_booleana_interpretari_ale_imagisticii"
        "Va returna `True` daca documentul contine urmatoarele cuvinte cheie: 'ecografie', 'explorare ecografica', 'substanta de contrast', 'SC', 'CT', 'rezonanta magnetica', 'computer tomografie', 'computer tomograf', 'PET-CT', 'scintigrafie', 'scintigrafic', 'coronarografie', 'mamografie'. Altfel, va returna `False`.\n\n"

        "- cod_numeric_personal_cod_unic_asigurare_pacient"
        "Codul va aparea dupa urmatoarele cuvinte cheie: 'CNP', 'Cod Numeric Personal' sau 'cod unic de asigurare'.\n\n"

        "- data_introducere_document"
        "Poate aparea in urmatorele formate: 'dd.mm.yyyy', 'dd/mm/yyyy', 'dd-mm-yyyy'. Poate aparea dupa urmatoarele cuvinte cheie: 'Data inregistrarii', 'Data emiterii', 'Introdus la data', 'data:' sau alte tipuri de expresii. Daca data include ora si minutul, exclude-le si returneaza doar ziua, luna, anul sub format specific anterior.\n\n"

        "- data_rezultat\n\n"

        "- diagnostic_pacient"
        "Diagnosticul va aparea dupa cuvintele cheie: 'Diagnostic', 'Diagnosticul', 'Diagnostificat cu'\n\n"

        "- sumar_document"
        "Genereaza un rezumat detaliat al documentului prezentand etapele de investigatie, analizele facute de pacient, starea pacientului, tratamentele care trebuie urmate si diagnosticul. Daca unul din termenii anteriori nu se regaseste in document, nu il mentiona. Rezumatul trebuie sa aiba maxim 500 de caractere.\n\n"
    )

    user_content = [{"type": "text", "text": instruction}]
    for url in image_urls:
        user_content.append({"type": "image_url", "image_url": {"url": url}})

    resp = client.chat.completions.create(
        model=MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_content},
        ],
        temperature=0,
    )
    raw = resp.choices[0].message.content or "{}"
    try:
        return json.loads(raw)
    except Exception:
        return {"raw_response": raw}

def parse_date(date_time_string):
    if not date_time_string:
        return datetime.min
    for date_time_format in (
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
    ):
        try:
            return datetime.strptime(date_time_string.strip(), date_time_format)
        except ValueError:
            continue
    
    return datetime.min
    
def fetch_document_links(api_url: str) -> list[str]:
    response = requests.get(api_url)
    response.raise_for_status()
    data = response.json()
    documents = data.get("documents", [])
    return [doc["document_url"] for doc in documents if "document_url" in doc]

def create_dict_result(PATHS_URL):
    openai_results = dict()
    PATHS = fetch_document_links(PATHS_URL)

    for PATH in PATHS:
        # need to avoid the rate limit
        time.sleep(1)
        file = Path(PATH)
        result = dict()
        print(f"Starting OpenAI call for {PATH}")
        start = time.time()

        if PATH.endswith(".pdf"):
            if PATH.startswith("http"):
                # Download PDF to memory
                pdf_bytes = requests.get(PATH).content
                temp_pdf = BytesIO(pdf_bytes)
                data_urls = pdf_to_data_urls(temp_pdf, dpi=DPI, limit=MAX_PAGES)
            else:
                data_urls = pdf_to_data_urls(str(file), dpi=DPI, limit=MAX_PAGES)
            if not data_urls:
                raise RuntimeError("PDF to image conversion failed or document has no pages.")
            result = call_openai_with_images(data_urls)

        elif PATH.endswith((".png", ".jpg", ".jpeg")):
            if PATH.startswith("http"):
                img_bytes = requests.get(PATH).content
                img = Image.open(BytesIO(img_bytes)).convert("RGB")
            else:
                img = Image.open(file).convert("RGB")
            data_url = img_to_data_url(img)
            result = call_openai_with_images([data_url])

        openai_results[Path(PATH).name] = result

        end = time.time()
        print(f"Finished OpenAI call for {PATH}. Time taken {end - start:.2f} seconds")

    openai_results_sorted = dict(sorted(
        openai_results.items(),
        key=lambda item: parse_date(
            item[1].get("data_introducere_document") or item[1].get("data_rezultat", "")
        ),
        reverse=True
    ))

    return openai_results_sorted

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(silent=True) or {}
    paths_url = data.get("paths_url")
    if not paths_url:
        return jsonify({"ok": False, "error": "Missing 'paths_url'"}), 400
    try:
        result = create_dict_result(paths_url)
        return jsonify({"ok": True, "result": result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

if __name__ == "__main__":
    app.run()