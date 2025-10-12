import time
from flask import Flask, request, jsonify

from config import (FLASK_HOST, FLASK_PORT, FLASK_DEBUG, MAX_RETRIES, RETRY_DELAY, BATCH_SIZE)
from batch_processor import create_dict_result
from utils import fetch_document_links

app = Flask(__name__)

@app.route("/analyze", methods=["POST"])
def analyze():
    """Analyze documents using batch processing with retry mechanism"""
    data = request.get_json(silent=True) or {}
    paths_url = data.get("paths_url")
    if not paths_url:
        return jsonify({"ok": False, "error": "Missing 'paths_url'"}), 400
    
    try:
        print(f"Starting analysis for URL: {paths_url}")
        start_time = time.time()
        
        try:
            links = fetch_document_links(paths_url)
            total_documents = len(links)
            estimated_time = total_documents * 15
            
            print(f"Found {total_documents} documents to process")
            print(f"Estimated processing time: {estimated_time} seconds ({estimated_time/60:.1f} minutes)")
            
            if total_documents > 50:
                return jsonify({
                    "ok": False, 
                    "error": f"Too many documents ({total_documents}). Maximum is 50."
                }), 400
                
        except Exception as e:
            print(f"Error fetching document links: {str(e)}")
            return jsonify({"ok": False, "error": f"Failed to fetch document links: {str(e)}"}), 400
        
        # process documents
        result = create_dict_result(paths_url)
        
        end_time = time.time()
        processing_time = end_time - start_time
        print(f"Total processing time: {processing_time:.2f} seconds")
        
        # successful vs failed documents
        successful_count = sum(1 for res in result.values() if "error" not in res)
        failed_count = len(result) - successful_count
        retry_failures = sum(1 for res in result.values() if res.get("final_failure", False))
        
        return jsonify({
            "ok": True, 
            "result": result,
            "processing_time_seconds": processing_time,
            "document_count": len(result),
            "successful_documents": successful_count,
            "failed_documents": failed_count,
            "retry_failures": retry_failures,
            "batch_processing": True,
            "batch_size": BATCH_SIZE,
            "retry_config": {
                "max_retries": MAX_RETRIES,
                "retry_delay": RETRY_DELAY
            }
        })
        
    except Exception as e:
        print(f"Error in /analyze endpoint main method: {str(e)}")
        return jsonify({"ok": False, "error": str(e)}), 500

if __name__ == "__main__":
    # app.run(debug=FLASK_DEBUG, host=FLASK_HOST, port=FLASK_PORT)
    app.run()