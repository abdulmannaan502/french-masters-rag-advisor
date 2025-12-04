from pathlib import Path
from typing import List
from pypdf import PdfReader

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text from a single PDF file."""
    reader = PdfReader(str(pdf_path))
    texts: List[str] = []

    for page_num, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            texts.append(page_text)
        else:
            print(f"Warning: no text extracted from page {page_num} of {pdf_path.name}")

    return "\n".join(texts)


def process_all_pdfs():
    """Extract text from all PDFs in data/raw and save to data/processed."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = list(RAW_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {RAW_DIR.resolve()}")
        return

    for pdf_file in pdf_files:
        print(f"Processing {pdf_file.name} ...")
        text = extract_text_from_pdf(pdf_file)

        out_path = PROCESSED_DIR / f"{pdf_file.stem}.txt"
        out_path.write_text(text, encoding="utf-8")

        print(f"Saved extracted text to {out_path}")


if __name__ == "__main__":
    process_all_pdfs()
