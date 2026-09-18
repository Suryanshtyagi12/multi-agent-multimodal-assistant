# Error Fix Log

## Error Encountered
- Error name: `[Errno 11001] getaddrinfo failed` (DNS resolution error for api-inference.huggingface.co)
- File: `app/ingestion/embedder.py`
- Line number: Inside `embed_text()` and `embed_batch()` during `client.post()`

## Root Cause
- Hugging Face officially retired the `api-inference.huggingface.co` subdomain for the Serverless Inference API in 2025. The A record was removed, which is why DNS resolution fails.
- The new default inference endpoint is `router.huggingface.co/hf-inference`.
- Using `client.post()` with a manual task string in older/some code snippets forced the legacy fallback URL. 
- Additionally, calling the new `router.huggingface.co` endpoints requires the `HF_TOKEN` to be a **Fine-Grained token** with explicit "Inference" permissions checked, otherwise it returns a `403 Forbidden`.

## Fix Applied
- Replaced the manual `client.post(json=...)` calls with the native `client.feature_extraction()` method in `app/ingestion/embedder.py`. The modern `InferenceClient` natively routes to `router.huggingface.co/hf-inference` correctly under the hood.
- Added a robust `_normalize_embedding` function to handle any shape returned by `feature_extraction` (which can be a 1D, 2D, or 3D numpy array or list) and flatten it into a 1024-dim `list[float]`.
- Implemented exponential backoff for the retry logic `delay = BASE_DELAY * (2 ** attempt)`.
- Re-wrote `embed_text` and `embed_batch` to use this new architecture.

## Packages Updated
- None required directly. (`huggingface_hub` >= 0.25 natively handles the new routing).

## Verification
- To verify the fix works, you MUST generate a new Fine-Grained access token on huggingface.co/settings/tokens and check the box that says "Make calls to the Serverless Inference API". Update `.env` with this new token.
- Run `python app/ingestion/embedder.py`
- Expected output:
  `Embedding dimension: 1024`
  `First 5 values: [0.0123, -0.0456, ...]`

## Error Encountered
- Error name: `pip's dependency resolver does not currently take into account all the packages that are installed.`
- File: Terminal (during `pip install sentence-transformers google-generativeai`)
- Line number: N/A

## Root Cause
- Installing `sentence-transformers` automatically upgraded `huggingface-hub` to `1.19.0` and `typer` to `0.25.1`. 
- `docling 2.5.0` (which is hardcoded in `requirements.txt`) has strict upper bounds: it requires `huggingface_hub<1,>=0.23` and `typer<0.13.0,>=0.12.5`. This caused a version conflict breaking the Docling parser.

## Fix Applied
- Ran a force-downgrade using pip: `.\venv\Scripts\pip.exe install "huggingface_hub<1" "typer<0.13.0"`.
- This ensures `docling` dependencies remain satisfied while still being recent enough to support `sentence-transformers` and the new HF routing logic.

## Packages Updated
- `huggingface-hub` | `1.19.0` | downgraded to `<1` (e.g. `0.26.0`)
- `typer` | `0.25.1` | downgraded to `<0.13.0` (e.g. `0.12.5`)

## Verification
- Run `.\venv\Scripts\pip.exe check`.
- Expected output: `No broken requirements found.`

## Error Encountered
- Error name: `Unrecognized processing class in BAAI/bge-m3`
- File: Terminal (during `python app/ingestion/embedding_manager.py`)
- Line number: N/A

## Root Cause
- The `sentence-transformers` 5.6.0 installed `transformers` 5.12.1. However, because we forced `huggingface-hub` to downgrade to `0.26` for `docling` compatibility, `transformers` 5.12.1 crashed or failed to properly load the processing class because it expects `huggingface-hub >= 1.5.0`.

## Fix Applied
- Downgraded the sentence transformers stack to older stable versions that support `huggingface-hub < 1`: `.\venv\Scripts\pip.exe install "sentence-transformers<3.0.0" "transformers<4.40.0"`.

## Packages Updated
- `sentence-transformers` | `5.6.0` | downgraded to `2.7.0`
- `transformers` | `5.12.1` | downgraded to `4.39.3`

## Verification
- Run `.\venv\Scripts\python.exe app/ingestion/embedding_manager.py`.
- The model should download successfully and output the 1024-dim test vector without crashing.

## Error Encountered
- Error name: `ModuleNotFoundError: No module named 'app'`
- File: Terminal (during `python app/ingestion/embedder.py`)
- Line number: `from app.ingestion.embedding_manager import embed_text`

## Root Cause
- When running a python script directly from a subfolder (`python app/.../script.py`), Python does not automatically add the root project directory to its `sys.path`. It only adds the directory the script resides in. Therefore, the absolute import `app.ingestion...` fails because it doesn't know what `app` is.

## Fix Applied
- Added `sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))` to the top of `app/ingestion/embedder.py`. 
- This ensures that if the script is executed directly for testing, the project root is forcibly added to the module search path, allowing `app.ingestion` imports to succeed.

## Packages Updated
- None

## Verification
- Run `python app/ingestion/embedder.py`.
- Expected output: `Embedding dimension: 1024` and the first 5 values.
