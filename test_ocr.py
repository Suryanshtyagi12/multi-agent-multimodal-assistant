from PIL import Image
import pytesseract
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


img = Image.open(
    r"D:\multi-agent-multimodel-assistant\temp_uploads\Screenshot 2025-12-19 030341.png"
)
print(pytesseract.image_to_string(img)[:500])