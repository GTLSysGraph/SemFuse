import json
import re

# Read file
with open("../data/output_json/Amazon_output.json", "r") as f:
    data = json.load(f)

def clean_output_field(output_str):
    output_str = re.sub(r"^```json\s*", "", output_str.strip())
    output_str = re.sub(r"\s*```$", "", output_str.strip())
    try:
        return json.loads(output_str)
    except json.JSONDecodeError:
        print("JSON parsing failed, keep original string")
        return output_str  # fallback

# Clean all data
for item in data:
    if "output" in item:
        item["output"] = clean_output_field(item["output"])

# Save to new file
with open("../data/output_json/Amazon_output_cleaned.json", "w") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)