import os
import sys

import cv2
from PIL import Image

from preprocess import preprocess_image
from restore_text import restore_text
from strip import clean_text

try:
    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
except Exception as e:  # pragma: no cover
    torch = None
    TrOCRProcessor = None
    VisionEncoderDecoderModel = None
    _IMPORT_ERROR = e


_processor = None
_model = None


def _get_device():
    if torch is None:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_model():
    global _processor, _model
    if _processor is not None and _model is not None:
        return _processor, _model

    if TrOCRProcessor is None or VisionEncoderDecoderModel is None:
        raise RuntimeError(
            "Missing TrOCR dependencies. Install: pip install transformers torch pillow\n"
            f"Original import error: {_IMPORT_ERROR}"
        )

    model_name = os.environ.get("TROCR_MODEL", "microsoft/trocr-base-printed")
    _processor = TrOCRProcessor.from_pretrained(model_name)
    _model = VisionEncoderDecoderModel.from_pretrained(model_name)
    _model.to(_get_device())
    _model.eval()
    return _processor, _model


def _smart_crop_for_ocr(image_bgr):
    """
    Crop the photo to the main ink/text region with a bit of padding.
    Helps a lot for phone photos with large backgrounds.
    """
    if image_bgr is None:
        raise ValueError("Could not read image (cv2.imread returned None).")

    h, w = image_bgr.shape[:2]
    if h == 0 or w == 0:
        return image_bgr

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Invert so ink becomes "white" for easier bounding box extraction.
    # Otsu handles variable lighting reasonably well.
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Remove small specks, connect nearby strokes.
    k = max(3, (min(h, w) // 200) | 1)  # odd kernel size, scales with image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)

    ys, xs = (th > 0).nonzero()
    if len(xs) == 0 or len(ys) == 0:
        return image_bgr

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())

    pad = int(0.04 * max(h, w))
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(w - 1, x1 + pad)
    y1 = min(h - 1, y1 + pad)

    # Avoid absurd crops (e.g., a tiny dot).
    if (x1 - x0) < 0.2 * w or (y1 - y0) < 0.2 * h:
        return image_bgr

    return image_bgr[y0 : y1 + 1, x0 : x1 + 1]


def extract_text_trocr(image_bgr_or_gray):
    """
    Takes a cv2 image (BGR or GRAY) and returns extracted text using TrOCR.
    """
    processor, model = _load_model()

    if len(image_bgr_or_gray.shape) == 2:
        rgb = cv2.cvtColor(image_bgr_or_gray, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(image_bgr_or_gray, cv2.COLOR_BGR2RGB)

    pil_img = Image.fromarray(rgb)

    device = _get_device()
    pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values.to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            pixel_values,
            max_new_tokens=128,
            num_beams=4,
            early_stopping=True,
        )

    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text


def run_full_pipeline(image_path: str, preprocess: bool = True) -> str:
    if preprocess:
        # TrOCR usually performs better with a natural-looking crop,
        # not an aggressively binarized threshold image.
        original = cv2.imread(image_path)
        image = _smart_crop_for_ocr(original)
    else:
        image = cv2.imread(image_path)
    text = extract_text_trocr(image)
    cleaned_text = clean_text(text)
    restored = restore_text(cleaned_text)
    return restored


def _main(argv):
    if len(argv) < 2:
        print("Usage: python trocr_ocr.py <image_path> [--no-preprocess]", file=sys.stderr)
        return 2

    image_path = argv[1]
    do_preprocess = "--no-preprocess" not in argv[2:]

    try:
        result = run_full_pipeline(image_path, preprocess=do_preprocess)
    except Exception as e:
        print(str(e), file=sys.stderr)
        return 1

    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))

