import json

# input_path = "/workspaces/training_repository/parse_pdf/Grade_9_Biology_structured_content_3.json"
# output_path = "/workspaces/training_repository/parse_pdf/Grade_9_Biology_structured_content_3.json"

# Load the JSON data
with open("/workspaces/training_repository/parse_pdf/Grade_10_Biology_keyword_definitions.json", "r", encoding="utf-8") as m:
    data = json.load(m)

# # Decrement page_number for each entry
c = 0
for ent in data:
    # if "page_number" in ent and isinstance(ent["page_number"], int):
        # ent["page_number"] -= 4
    ent["_id"]=f"rec_{c}"
    c+=1

# # Write the updated data back to file
with open("/workspaces/training_repository/parse_pdf/Grade_10_Biology_keyword_definitions.json", "w", encoding="utf-8") as p:
    json.dump(data, p, ensure_ascii=False, indent=4)

# print(f"Updated {len(data)} entries and saved to '{output_path}'.")from pinecone import Pinecone
