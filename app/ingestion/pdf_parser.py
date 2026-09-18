import os
from pathlib import Path
import hashlib
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling_core.types.doc.document import PictureItem

def parse_pdf(pdf_path: str) -> list[dict]:
    """
    Parses a research paper PDF and returns structured data containing text, tables, and figures.
    """
    # Ensure figures directory exists
    os.makedirs("figures", exist_ok=True)
    
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = True
    pipeline_options.images_scale = 2.0
    pipeline_options.generate_page_images = False
    pipeline_options.generate_picture_images = True
    
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options
            )
        }
    )
    result = converter.convert(pdf_path)
    doc = result.document
    doc_dict = doc.export_to_dict()
    
    elements = []
    current_section = "Unknown"
    source_filename = os.path.basename(pdf_path)
    
    # Ensure figures directory exists
    os.makedirs("figures", exist_ok=True)
    
    # Loop through all elements in the document
    # iterate_items() yields (item, level) tuples in reading order
    for item, level in doc.iterate_items():
        raw_label = getattr(item, "label", "")
        # Safely extract label string, handling Enum if present
        label = raw_label.value.lower() if hasattr(raw_label, "value") else str(raw_label).lower()
        label = label.replace("-", "_")
        
        # Determine page number
        page_no = 1
        prov = getattr(item, "prov", [])
        if prov and len(prov) > 0:
            page_no = getattr(prov[0], "page_no", 1)

        # 1. Text elements (paragraphs, headings, abstract, introduction, conclusion, references)
        if label in ["text", "section_header", "title", "list_item", "footnote", "caption", "page_header", "page_footer", "reference", "paragraph"]:
            text_content = getattr(item, "text", "")
            
            # Track the last seen heading for section_title
            if label in ["section_header", "title"] and text_content:
                current_section = text_content
                
            elements.append({
                "type": "text",
                "content": text_content,
                "image_path": None,
                "page_number": page_no,
                "section_title": current_section,
                "source_filename": source_filename
            })
            
        # 2. Table elements
        elif label == "table":
            markdown_content = ""
            if hasattr(item, "export_to_markdown"):
                markdown_content = item.export_to_markdown()
                
            elements.append({
                "type": "table",
                "content": markdown_content,
                "image_path": None,
                "page_number": page_no,
                "section_title": current_section,
                "source_filename": source_filename
            })
            
        # 3. Figure elements (pictures/figures)
        elif label in ["picture", "image", "figure"] or isinstance(item, PictureItem):
            saved_path = ""
            try:
                # generate unique filename from content hash
                fig_filename = f"fig_{hashlib.md5(str(item).encode()).hexdigest()[:8]}.png"
                fig_path = os.path.join("figures", fig_filename)
                
                # save image using docling's built-in image export
                with open(fig_path, "wb") as f:
                    image = item.get_image(doc)
                    if image is not None:
                        image.save(f, format="PNG")
                        saved_path = fig_path
            except Exception as e:
                print(f"Could not save figure: {e}")
                
            # get caption if exists
            caption = ""
            try:
                if hasattr(item, 'captions') and item.captions:
                    caption = " ".join([getattr(c, "text", "") for c in item.captions])
                elif hasattr(item, 'caption') and item.caption:
                    caption = str(item.caption)
                elif hasattr(item, "text") and item.text:
                    caption = getattr(item, "text", "")
            except:
                pass
                
            elements.append({
                "type": "figure",
                "content": caption,
                "image_path": saved_path,
                "page_number": page_no,
                "section_title": current_section,
                "source_filename": source_filename
            })

    return elements

if __name__ == "__main__":
    results = parse_pdf("test.pdf")
    for r in results[:5]:
        content_preview = r["content"][:80] if r["content"] else ""
        print(r["type"], r["page_number"], content_preview)

