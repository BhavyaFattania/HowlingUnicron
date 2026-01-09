from llama_cloud_services import LlamaParse
import os
def parser_for(file_path, output_folder="./input"):
    parser = LlamaParse(
    api_key="llx-r9aHdNJURHoh8Kw5TeXeWQJrZWEmhAihKUmb5K3At4JxI7qn",  # See how to get your API key at https://developers.llamaindex.ai/python/cloud/general/api_key
    parse_mode="parse_page_with_agent",  # The parsing mode
    result_type ="markdown", # The model to use
    high_res_ocr=True,  # Whether to use high resolution OCR (slower but more precise)
    adaptive_long_table=True,  # Adaptive long table. LlamaParse will try to detect long table and adapt the output
    outlined_table_extraction=True,  # Whether to try to extract outlined tables
    )

    # 2. Execution
    print(f"--- Starting Pipeline for: {file_path} ---")
        
    # load_data returns a list of document objects
    documents = parser.load_data(file_path)

    # 3. Export to GraphRAG Input Directory
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    base_name = os.path.basename(file_path).replace(".pdf", ".md")
    final_path = os.path.join(output_folder, base_name)

    with open(final_path, "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(f"\n\n")
            f.write(doc.text + "\n\n")

    print(f"--- Pipeline Complete! File saved to: {final_path} ---")
    
path = ""
parser_for(path)