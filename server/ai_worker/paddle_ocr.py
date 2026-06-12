import sys

import cv2

from restore_text import restore_text
from strip import clean_text

try:
    from paddleocr import PaddleOCR
except Exception as e:  # pragma: no cover
    PaddleOCR = None
    _IMPORT_ERROR = e


_ocr = None


def _load_ocr():
    global _ocr
    if _ocr is not None:
        return _ocr

    if PaddleOCR is None:
        raise RuntimeError(
            "Missing PaddleOCR dependencies. Install: pip install paddleocr\n"
            f"Original import error: {_IMPORT_ERROR}"
        )

    # For handwriting, detection+recognition generally beats pure recognizers on full pages.
    # `use_angle_cls=True` helps with slight rotations.
    _ocr = PaddleOCR(use_angle_cls=True, lang="en")
    return _ocr


def _smart_crop_for_ocr(image_bgr):
    """
    Crop the image to the main ink/text region with padding.
    Helps remove notebook margins/background for phone photos.
    """
    if image_bgr is None:
        raise ValueError("Could not read image (cv2.imread returned None).")

    h, w = image_bgr.shape[:2]
    if h == 0 or w == 0:
        return image_bgr

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    k = max(3, (min(h, w) // 200) | 1)
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

    if (x1 - x0) < 0.2 * w or (y1 - y0) < 0.2 * h:
        return image_bgr

    return image_bgr[y0 : y1 + 1, x0 : x1 + 1]


def extract_text_paddle(image_bgr):
    ocr = _load_ocr()

    # PaddleOCR expects RGB if you pass numpy arrays.
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    result = ocr.ocr(rgb, cls=True)

    lines = []
    for page in result:
        # page: list of [box, (text, score)]
        for item in page:
            if not item or len(item) < 2:
                continue
            text, score = item[1]
            if text:
                lines.append(text)

    return "\n".join(lines).strip()


def run_full_pipeline(image_path: str, smart_crop: bool = True) -> str:
    image = cv2.imread(image_path)
    if smart_crop:
        image = _smart_crop_for_ocr(image)

    text = extract_text_paddle(image)
    cleaned = clean_text(text)
    restored = restore_text(cleaned)
    return restored


def _main(argv):
    if len(argv) < 2:
        print("Usage: python paddle_ocr.py <image_path> [--no-crop]", file=sys.stderr)
        return 2

    image_path = argv[1]
    smart_crop = "--no-crop" not in argv[2:]

    try:
        out = run_full_pipeline(image_path, smart_crop=smart_crop)
    except Exception as e:
        print(str(e), file=sys.stderr)
        return 1

    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))

