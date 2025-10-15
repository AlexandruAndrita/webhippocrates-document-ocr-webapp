import json
from openai import OpenAI
from config import OPENAI_API_KEY, MODEL, OPENAI_TIMEOUT

client = OpenAI(api_key=OPENAI_API_KEY, timeout=OPENAI_TIMEOUT)

def call_openai_with_images(image_urls: list[str]) -> dict:
    """
    Call OpenAI API with medical document images for data extraction.
    
    Args:
        image_urls: List of base64 data URLs for images
        
    Returns:
        Dictionary with extracted medical document data
    """
    if not client:
        return {"error": "OpenAI client not initialized - check API key configuration"}
    system_msg = "Ești un extractor de date din documente medicale. Returnează DOAR JSON valid."

    instruction = (
        "Extrage următoarele câmpuri din imaginea de document medical furnizată. "
        "Dacă un câmp lipsește, folosește null. Daca anumite cuvinte cheie nu se regasesc, iar in prompt ti se indica sa folosesti anumite valori e.g. `True` sau `False`, foloseste-le. Nu inventa valori.\n\n"
        
        "- titlu_document\n"
        "Tipurile de titluri pot fi 'Scrisoare Medicala', 'Bilet de iesire din spital', 'Bilet de iesire', 'Bilet de externare'. Inafara de acestea, exista si alte titluri care pot aparea in document si trebuie extrase.\n\n"

        "- nume_prenume_pacient\n"
        "Va aprea dupa urmatoarele cuvinte cheie: 'Nume si prenume', poate aparea dupa 'Pacientul'. Intotdeauna este un nume scris cu majuscule.\n\n"

        "- variabila_booleana_diagnostic_curent\n"
        "Va returna `True` daca 'titlu_document' contine urmatoarele cuvinte cheie: 'Scrisoare medicala', 'Bilet de iesire din spital', 'Bilet de iesire', 'Bilet de externare'. Altfel, va returna `False`.\n\n"

        "- variabila_booleana_analize_medicale\n"
        "Va returna `True` daca documentul contine urmatoarele cuvinte cheie: 'hemoglobina', 'hematocrit', 'hemoleucograma'. Altfel, va returna `False`.\n\n"

        "- variabila_booleana_examen_hispotatologic\n"
        "Va returna `True` daca documentul contine urmatoarele cuvinte cheie: 'histopatologica', 'histopatologic', 'microscopie', 'macroscopie', 'imunohistochimie', 'biopsie', 'biopsic', 'biopsice', 'OncoType', 'examen imunohistochimic', 'IHC', 'EHP'. Altfel, va returna `False`.\n\n"

        "- variabila_booleana_interpretari_ale_imagisticii\n"
        "Va returna `True` daca documentul contine urmatoarele cuvinte cheie: 'ecografie', 'explorare ecografica', 'substanta de contrast', 'SC', 'CT', 'rezonanta magnetica', 'computer tomografie', 'computer tomograf', 'PET-CT', 'scintigrafie', 'scintigrafic', 'coronarografie', 'mamografie'. Altfel, va returna `False`.\n\n"

        "- cod_numeric_personal_cod_unic_asigurare_pacient\n"
        "Codul va aparea dupa urmatoarele cuvinte cheie: 'CNP', 'Cod Numeric Personal' sau 'cod unic de asigurare'.\n\n"

        "- data_introducere_document\n"
        "Poate aparea in urmatorele formate: 'dd.mm.yyyy', 'dd/mm/yyyy', 'dd-mm-yyyy'. Poate aparea dupa urmatoarele cuvinte cheie: 'Data inregistrarii', 'Data emiterii', 'Introdus la data', 'data:' sau alte tipuri de expresii. Daca data include ora si minutul, exclude-le si returneaza doar ziua, luna, anul sub format specific anterior.\n\n"

        "- data_rezultat\n\n"

        "- diagnostic_pacient\n"
        "Diagnosticul va aparea dupa cuvintele cheie: 'Diagnostic', 'Diagnosticul', 'Diagnostificat cu'\n\n"

        "- sumar_document\n"
        "Genereaza un rezumat detaliat al documentului prezentand etapele de investigatie, analizele facute de pacient, starea pacientului, tratamentele care trebuie urmate si diagnosticul. Daca unul din termenii anteriori nu se regaseste in document, nu il mentiona. Rezumatul trebuie sa aiba maxim 500 de caractere.\n\n"
    )

    user_content = [{"type": "text", "text": instruction}]  
    for url in image_urls:
        user_content.append({"type": "image_url", "image_url": {"url": url}})

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_content},
            ]
        )
        
        raw = resp.choices[0].message.content or "{}"
        return json.loads(raw)
        
    except json.JSONDecodeError as e:
        return {"raw_response": raw, "json_error": str(e)}
    except Exception as e:
        return {"api_error": str(e)}

