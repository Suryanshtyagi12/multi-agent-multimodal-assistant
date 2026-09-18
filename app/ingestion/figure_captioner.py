import os
import time
import base64
import logging
from google import genai
from google.genai import types
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

GEMINI_VISION_MODEL = "gemini-2.5-flash"

GEMINI_KEYS = [
    os.getenv("GEMINI_API_KEY"),
    os.getenv("GEMINI_API_KEY_2"),
    os.getenv("GEMINI_API_KEY_3")
]
GEMINI_KEYS = [k for k in GEMINI_KEYS if k]

FIGURE_PROMPT = """You are analyzing a figure from a research paper.
Describe what this figure shows in detail. Include:
- Type of visualization (graph, diagram, architecture, flowchart)
- What axes represent if applicable
- The key finding or pattern shown
- Any notable values or comparisons visible
Be specific and technical. Minimum 3 sentences."""

def _image_to_bytes(image_path: str) -> bytes:
    with open(image_path, "rb") as f:
        return f.read()

def _caption_with_gemini(image_path: str, api_key: str) -> str:
    client = genai.Client(api_key=api_key)
    image_bytes = _image_to_bytes(image_path)
    
    response = client.models.generate_content(
        model=GEMINI_VISION_MODEL,
        contents=[
            types.Part.from_bytes(
                data=image_bytes,
                mime_type="image/png"
            ),
            FIGURE_PROMPT
        ]
    )
    return response.text.strip()

def caption_figure(image_path: str) -> str:
    if not os.path.exists(image_path):
        logger.warning(f"Figure not found: {image_path}")
        return ""
    
    if not GEMINI_KEYS:
        logger.error("No Gemini API keys found")
        return ""
    
    for i, api_key in enumerate(GEMINI_KEYS, start=1):
        try:
            logger.info(f"Captioning with Gemini key {i}: {image_path}")
            description = _caption_with_gemini(image_path, api_key)
            logger.info(f"Gemini key {i} succeeded")
            time.sleep(0.5)
            return description
        except Exception as e:
            logger.warning(f"Gemini key {i} failed: {e}")
            time.sleep(1)
            continue
    
    logger.error(f"All Gemini keys failed for {image_path}")
    return ""

def caption_all_figures(chunks: list[dict]) -> list[dict]:
    figure_chunks = [c for c in chunks if c["type"] == "figure"
                     and c.get("image_path")]
    
    if not figure_chunks:
        logger.info("No figures to caption")
        return chunks
    
    print(f"Captioning {len(figure_chunks)} figures with "
          f"{len(GEMINI_KEYS)} Gemini keys...")
    
    fig_num = 0
    for chunk in chunks:
        if chunk["type"] == "figure" and chunk.get("image_path"):
            fig_num += 1
            print(f"  Figure {fig_num}/{len(figure_chunks)}: "
                  f"{chunk['image_path']}")
            description = caption_figure(chunk["image_path"])
            if description:
                chunk["content"] = description
            time.sleep(1)
    
    return chunks
