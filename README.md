# Smart Document Restoration

Extract readable, semantically corrected text from document images (printed scans, screenshots, or handwritten notes).

The API accepts an image, runs OCR to pull out raw text, cleans it, then passes it through a grammar-correction transformer so the final output is clearer than raw OCR alone.

## Workflow

```
Image upload (Express)
    → Python OCR worker (PaddleOCR / TrOCR / Tesseract)
    → Text cleanup (strip.py)
    → Grammar correction (restore_text.py)
    → JSON response with corrected text
```

**Node** handles the HTTP API and file upload. **Python** handles image processing, OCR, and text restoration. The server calls a Python script via `child_process.exec` and returns whatever it prints to stdout.

## Project structure

```
Smart_Doc_restoration/
├── README.md
├── .gitignore
└── server/
    ├── index.js              # Express app (port 8080)
    ├── package.json
    ├── routes/
    │   └── upload.js         # POST /api/upload — multer + exec Python worker
    └── ai_worker/
        ├── paddle_ocr.py     # Default OCR: PaddleOCR (handwriting-friendly)
        ├── trocr_ocr.py      # TrOCR (transformer OCR)
        ├── ocr_only.py       # Tesseract pipeline entrypoint
        ├── ocr.py            # pytesseract wrapper
        ├── preprocess.py     # OpenCV preprocessing for Tesseract
        ├── strip.py          # Regex text cleanup
        └── restore_text.py   # Grammar correction (HuggingFace)
```

Switch OCR backends in `server/routes/upload.js` by changing the `exec` command:

| Backend   | Command |
|-----------|---------|
| PaddleOCR | `python ai_worker/paddle_ocr.py "${imagePath}"` |
| TrOCR     | `python ai_worker/trocr_ocr.py "${imagePath}"` |
| Tesseract | `python ai_worker/ocr_only.py "${imagePath}"` |

## Packages

**Node** (`server/`)

- `express` — HTTP server
- `multer` — multipart file uploads
- `cors` — CORS support (installed; wire up in `index.js` if needed)

**Python**

- `opencv-python` — image read/preprocess
- `pytesseract` — Tesseract OCR
- `transformers`, `torch`, `pillow` — TrOCR + grammar corrector
- `paddleocr`, `paddlepaddle` — PaddleOCR (default)

**System**

- [Tesseract](https://github.com/tesseract-ocr/tesseract) — required only for the Tesseract path

## Setup

1. Clone the repo and install Node dependencies:

```bash
git clone <repo-url>
cd Smart_Doc_restoration/server
npm install
```

2. Install Python dependencies (use one environment for all workers):

```bash
pip install opencv-python pytesseract transformers torch pillow paddleocr paddlepaddle
```

3. Start the server from `server/`:

```bash
node index.js
```

You should see: `Server is running on port 8080`

> First run downloads HuggingFace models (grammar corrector, and TrOCR/PaddleOCR weights). This can take a few minutes.

## Usage

**Endpoint:** `POST http://localhost:8080/api/upload`  
**Body:** `form-data` with field name `image` (file)

### curl

```bash
curl -X POST http://localhost:8080/api/upload \
  -F "image=@/path/to/your/document.png"
```

### Postman

1. Method: **POST**
2. URL: `http://localhost:8080/api/upload`
3. Body → **form-data**
4. Key: `image` (type: **File**)
5. Value: select your image
6. Send

### Example response

```json
{
  "message": "OCR extraction succesful",
  "extracted_text": "Smart Architecture: If image is screenshot..."
}
```

### Errors

- `400` — no file uploaded
- `500` — Python worker failed (check terminal stderr; often missing deps or bad image path)

## Notes

- The app is **local only** — clone, install deps, start the server, then send requests as above.
- Handwritten phone photos work best with **PaddleOCR** or **TrOCR** (`TROCR_MODEL=microsoft/trocr-base-handwritten`). Tesseract suits printed text.
- Uploaded files are stored in `server/uploads/` (gitignored).
