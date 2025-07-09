import google.generativeai as genai
import pdfplumber
import os
import json
from dotenv import load_dotenv
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import time

def extract_and_mark_page_text(pages):
    """
    Extracts text from a list of pdfplumber page objects and embeds page markers.
    The page number used is the actual page number from the PDF document.
    """
    if not pages:
        return ""
    
    full_text = ""
    for page in pages:
        # Use the actual page number from the PDF
        page_number = page.page_number
        page_text = page.extract_text()
        if page_text:
            # Add a clear marker at the beginning of each page's content
            full_text += f"\n\n--- PAGE {page_number} ---\n\n" + page_text
    return full_text

def get_structured_data_from_gemini(api_key, text_chunk):
    """
    Sends a large text chunk to the Gemini API to identify headers and extract 
    their corresponding text into a structured JSON format.
    """
    try:
        genai.configure(api_key=api_key)
        # Using a model with a larger context window is ideal for this task
        model = genai.GenerativeModel('gemini-2.5-flash')

        # --- SYSTEM PROMPT: ONLY EXTRACT MULTIPLE ALTERNATIVE QUESTIONS ---
        system_prompt = f"""
You are an expert data extractor. Your ONLY task is to extract all multiple alternative questions (such as multiple choice, alternative, or similar questions) from the provided text. Ignore all other content.

For each alternative question you find, extract:
- The question text (as the value for the key 'question')
- The list of options/choices (as the value for the key 'options')

Return a single JSON array, where each item is a JSON object with the following keys:
- 'question': the question text
- 'options': a list of strings, each being an option/choice

If no alternative questions are found, return an empty list [].
DO NOT include any other content, explanations, or markdown. Only output the JSON array.

Example output:
[
  {{
    "question": "Which of the following is an example of biotechnology?",
    "options": ["A. Baking bread", "B. Painting", "C. Running", "D. Reading"]
  }},
  {{
    "question": "Biotechnology involves the use of?",
    "options": ["A. Living organisms", "B. Rocks", "C. Metals", "D. Plastics"]
  }}
]
"""
        
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        response = model.generate_content(system_prompt, safety_settings=safety_settings)
        # Basic validation of the response
        if not response.text:
            return None, "API returned an empty response."

        # Try to extract the first valid JSON array from the response (in case of extra text)
        import re
        response_text = response.text.strip()
        # Remove markdown code block markers if present
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        # Try to extract the first JSON array
        match = re.search(r'(\[.*?\])', response_text, re.DOTALL)
        if match:
            response_text = match.group(1)
        return response_text, None
    except Exception as e:
        return None, f"An error occurred with the Gemini API: {e}"

def save_data_to_json(data_list, output_path):
    """Saves the list of dictionaries to a JSON file."""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data_list, f, indent=4)
        return True, None
    except Exception as e:
        return False, f"Error saving to JSON file: {e}"

def process_pdf_in_chunks(pdf_path, pages_per_chunk=10, max_chars_per_call=90000):
    """
    Process a PDF file in chunks of specified pages, handling text extraction and API calls.
    
    Args:
        pdf_path (str): Path to the PDF file
        pages_per_chunk (int): Number of pages to process in each chunk
        max_chars_per_call (int): Maximum characters to send in a single API call
    
    Returns:
        list: List of structured data extracted from the PDF
    """
    all_structured_data = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            print(f"\nTotal pages in PDF: {total_pages}")
            
            # Process PDF in chunks of pages
            for start_page in range(0, total_pages, pages_per_chunk):
                end_page = min(start_page + pages_per_chunk, total_pages)
                print(f"\n--- Processing pages {start_page + 1} to {end_page} ---")
                
                # Get the specified range of pages
                chunk_pages = pdf.pages[start_page:end_page]
                chunk_text = extract_and_mark_page_text(chunk_pages)
                
                if not chunk_text.strip():
                    print(f"No text extracted from pages {start_page + 1} to {end_page}. Skipping.")
                    continue
                
                print(f"  > Extracted {len(chunk_text)} characters from pages {start_page + 1} to {end_page}")
                
                # Split into smaller batches if needed based on character limit
                text_batches = []
                if len(chunk_text) > max_chars_per_call:
                    print(f"  > Chunk text is too long. Splitting into smaller batches of ~{max_chars_per_call} chars.")
                    for i in range(0, len(chunk_text), max_chars_per_call):
                        text_batches.append(chunk_text[i:i + max_chars_per_call])
                else:
                    text_batches.append(chunk_text)
                
                print(f"  > Processing chunk in {len(text_batches)} batch(es).")
                
                # Process each batch
                for i, text_batch in enumerate(text_batches):
                    print(f"    - Processing Batch {i+1}/{len(text_batches)}...")
                    
                    structured_data_str, error = get_structured_data_from_gemini(os.getenv("gemma_gemini_api"), text_batch)
                    
                    if error:
                        print(f"    - Error processing batch: {error}")
                        time.sleep(5)  # Add delay before next attempt
                        continue
                    
                    if structured_data_str:
                        try:
                            parsed_data = json.loads(structured_data_str)
                            if isinstance(parsed_data, list):
                                all_structured_data.extend(parsed_data)
                                print(f"    - Successfully parsed and added {len(parsed_data)} records from batch.")
                            else:
                                print("    - Warning: API did not return a list. Response skipped.")
                        except json.JSONDecodeError:
                            print(f"    - CRITICAL: Could not decode JSON from API response.")
                    
                    # Add a small delay between batches to avoid rate limiting
                    time.sleep(2)
                
    except Exception as e:
        print(f"A critical error occurred while processing the PDF: {e}")
        return []
    
    return all_structured_data

def main():
    """Main function to process PDF and extract structured content."""
    
    load_dotenv()
    YOUR_API_KEY = os.getenv("gemma_gemini_api")
    INPUT_PDF_PATH = "/workspaces/training_repository/pdf_files/2017 Bio EUEE @BrightAcademy9_12.pdf"
    OUTPUT_JSON_PATH = "Grade_matic_Biology_page_chunks.json"
    
    if not YOUR_API_KEY:
        print("Error: Gemini API key not found in .env file.")
        return

    if not os.path.exists(INPUT_PDF_PATH):
        print(f"Error: PDF file not found at '{INPUT_PDF_PATH}'")
        return

    print("--- Starting PDF Processing ---")
    
    # Process the PDF in chunks of 10 pages each
    all_structured_data = process_pdf_in_chunks(
        INPUT_PDF_PATH,
        pages_per_chunk=10,  # Process 10 pages at a time
        max_chars_per_call=90000  # Maximum characters per API call
    )

    print(f"\n--- All chunks processed. Found a total of {len(all_structured_data)} sections. ---")
    
    # Add sequential IDs to the records
    final_data_with_ids = []
    for i, record in enumerate(all_structured_data):
        record['_id'] = f'rec_{i + 1}'
        final_data_with_ids.append(record)

    print(f"Saving final structured data to '{OUTPUT_JSON_PATH}'...")
    
    success, error = save_data_to_json(final_data_with_ids, OUTPUT_JSON_PATH)
    
    if error:
        print(error)
    else:
        print(f"\n✅ All done! Your structured content has been saved to {OUTPUT_JSON_PATH}")
        print("\n--- FINAL OUTPUT PREVIEW (first 2 items) ---")
        print(json.dumps(final_data_with_ids[:2], indent=2))
        print("-" * 40)

if __name__ == "__main__":
    main()
