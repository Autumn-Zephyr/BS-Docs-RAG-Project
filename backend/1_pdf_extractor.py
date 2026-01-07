import pdfplumber

def extract_text_pdfplumber(pdf_path):
    full_text = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text.append(text)

    return "\n".join(full_text)

text = extract_text_pdfplumber("Raw Data/BS-DS_ Jan 2026 Grading document (STUDENT).pdf")
output_file = "Cleaned Data/extracted_text.txt"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(text)

print(text[:1000])