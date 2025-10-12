import time
import concurrent.futures
from pathlib import Path
from typing import Tuple, Dict, Any
from datetime import datetime

import requests

from config import (
    MAX_RETRIES, RETRY_DELAY, DOCUMENT_TIMEOUT, MAX_WORKERS, BATCH_SIZE,
    REQUEST_TIMEOUT, OPENAI_TIMEOUT
)
from document_processor import (
    process_pdf_document, process_image_document, get_document_type
)
from openai_service import call_openai_with_images
from utils import fetch_document_links, parse_date


def _process_document_core(path: str) -> dict:
    document_name = Path(path).name
    doc_type = get_document_type(path)
    
    if doc_type == "pdf":
        print(f"[{document_name}] Processing PDF document")
        data_urls = process_pdf_document(path)
        print(f"[{document_name}] Calling OpenAI with {len(data_urls)} images...")
        return call_openai_with_images(data_urls)
        
    elif doc_type == "image":
        print(f"[{document_name}] Processing image document")
        data_url = process_image_document(path)
        print(f"[{document_name}] Calling OpenAI with image...")
        return call_openai_with_images([data_url])
        
    else:
        raise RuntimeError(f"Unsupported file type: {path}")


def process_single_document(path: str) -> Tuple[str, Dict[str, Any]]:
    document_name = Path(path).name
    print(f"[{document_name}] Starting processing...")
    print(f"[{document_name}] Timeouts - Request: {REQUEST_TIMEOUT}s, OpenAI: {OPENAI_TIMEOUT}s, Document: {DOCUMENT_TIMEOUT}s")
    print(f"[{document_name}] Retry configuration - Max retries: {MAX_RETRIES}, Retry delay: {RETRY_DELAY}s")
    
    start_time = time.time()
    last_error = None
    
    for attempt in range(MAX_RETRIES + 1):
        try:
            attempt_start = time.time()
            
            if attempt > 0:
                print(f"[{document_name}] Retry attempt {attempt}/{MAX_RETRIES}")
                time.sleep(RETRY_DELAY)
            else:
                print(f"[{document_name}] Initial attempt")
            
            result = _process_document_core(path)
            
            attempt_time = time.time() - attempt_start
            total_time = time.time() - start_time
            
            if attempt > 0:
                print(f"[{document_name}] Successfully processed on retry {attempt} in {attempt_time:.2f}s (total: {total_time:.2f}s)")
            else:
                print(f"[{document_name}] Successfully processed on first attempt in {attempt_time:.2f}s")
            
            return document_name, result
            
        except requests.exceptions.Timeout as e:
            last_error = f"Network timeout: {str(e)}"
            print(f"[{document_name}] Attempt {attempt + 1} failed - {last_error}")
        except requests.exceptions.RequestException as e:
            last_error = f"Network error: {str(e)}"
            print(f"[{document_name}] Attempt {attempt + 1} failed - {last_error}")
        except Exception as e:
            last_error = f"Processing error: {str(e)}"
            print(f"[{document_name}] Attempt {attempt + 1} failed - {last_error}")
        

        if attempt < MAX_RETRIES:
            print(f"[{document_name}] Will retry in {RETRY_DELAY} seconds...")
        else:
            total_time = time.time() - start_time
            print(f"[{document_name}] All {MAX_RETRIES + 1} attempts failed after {total_time:.2f}s")
    
    return document_name, {
        "error": last_error,
        "attempts": MAX_RETRIES + 1,
        "final_failure": True
    }


def process_document_batch(paths_batch: list[str], batch_number: int) -> Dict[str, Any]:
    batch_results = dict()
    print(f"Processing batch {batch_number} with {len(paths_batch)} documents")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_path = {executor.submit(process_single_document, path): path for path in paths_batch}

        # per document --> DOCUMENT_TIMEOUT * (MAX_RETRIES + 1) + (MAX_RETRIES * RETRY_DELAY)
        max_time_per_doc = DOCUMENT_TIMEOUT * (MAX_RETRIES + 1) + (MAX_RETRIES * RETRY_DELAY)
        batch_timeout = max_time_per_doc * len(paths_batch)
        print(f"Batch timeout set to {batch_timeout} seconds ({max_time_per_doc}s per document including retries)")
        
        try:
            for future in concurrent.futures.as_completed(future_to_path, timeout=batch_timeout):
                try:
                    filename, result = future.result(timeout=10)
                    batch_results[filename] = result
                    
                    if "error" in result:
                        if "final_failure" in result:
                            print(f"Failed permanently after {result.get('attempts', 'unknown')} attempts: {filename}")
                        else:
                            print(f"Failed: {filename}")
                    else:
                        print(f"Successfully processed: {filename}")
                        
                except Exception as e:
                    path = future_to_path[future]
                    print(f"Future execution failed for {path}: {str(e)}")
                    batch_results[Path(path).name] = {"error": f"Future execution error: {str(e)}"}
                    
        except concurrent.futures.TimeoutError:
            print(f"Batch {batch_number} timed out after {batch_timeout} seconds")
            for future, path in future_to_path.items():
                if not future.done():
                    future.cancel()
                    filename = Path(path).name
                    if filename not in batch_results:
                        batch_results[filename] = {"error": f"Processing timeout after {batch_timeout} seconds"}
                elif future.done() and Path(path).name not in batch_results:
                    try:
                        filename, result = future.result()
                        batch_results[filename] = result
                    except Exception as e:
                        batch_results[Path(path).name] = {"error": str(e)}
    
    print(f"Completed batch {batch_number}: processed {len(batch_results)} documents")
    return batch_results


def create_dict_result(paths_url: str) -> Dict[str, Any]:
    openai_results = dict()
    paths = fetch_document_links(paths_url)
    
    if not paths:
        print("No documents found to process")
        return openai_results
    
    total_documents = len(paths)
    batch_size = BATCH_SIZE
    total_batches = (total_documents + batch_size - 1) // batch_size
    
    print(f"Total documents to process: {total_documents}")
    print(f"Processing in {total_batches} batches of {batch_size} documents each")
    print(f"Configuration: Request timeout: {REQUEST_TIMEOUT}s, OpenAI timeout: {OPENAI_TIMEOUT}s, Document timeout: {DOCUMENT_TIMEOUT}s")
    print(f"Retry configuration: Max retries: {MAX_RETRIES}, Retry delay: {RETRY_DELAY}s")
    
    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, total_documents)
        
        batch_paths = paths[start_idx:end_idx]
        print(f"\n--- Starting Batch {batch_num + 1}/{total_batches} ---")
        print(f"Documents in this batch: {[Path(p).name for p in batch_paths]}")
        
        try:
            batch_results = process_document_batch(batch_paths, batch_num + 1)
            openai_results.update(batch_results)
            
            if batch_num < total_batches - 1:
                print(f"Waiting 2 seconds before next batch...")
                time.sleep(2)
                
        except Exception as e:
            print(f"Error processing batch {batch_num + 1}: {str(e)}")
            for path in batch_paths:
                openai_results[Path(path).name] = {"error": f"Batch processing failed: {str(e)}"}

    openai_results_sorted = dict(sorted(
        openai_results.items(),
        key=lambda item: parse_date(
            item[1].get("data_introducere_document") or item[1].get("data_rezultat", "")
        ) if "error" not in item[1] else datetime.min,
        reverse=True
    ))
    
    successful_count = sum(1 for r in openai_results_sorted.values() if "error" not in r)
    failed_count = len(openai_results_sorted) - successful_count
    retry_failures = sum(1 for r in openai_results_sorted.values() if r.get("final_failure", False))
    
    print(f"\nProcessing complete! Total results: {len(openai_results_sorted)}")
    print(f"Successful: {successful_count}, Failed: {failed_count}")
    if retry_failures > 0:
        print(f"Failed after all retries: {retry_failures}")
    
    return openai_results_sorted