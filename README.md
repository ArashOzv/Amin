# Amin - Django Manual Image Cleaner

This repo now uses a **manual UI workflow** instead of automatic OCR/mask extraction.

You upload an image, choose draw/erase tool, paint the area you want removed, then press **Clean Selected Area**.
The backend sends your image + drawn mask to your FLUX Cloudflare worker, and only the masked region is replaced.

## What you get

- Browser UI to manually paint removal mask.
- Draw / erase brush tool.
- Per-request guidance + optional seed controls.
- Backend composites cleaned output only in masked region.
- Repeatable iterative workflow (draw -> clean -> draw again).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python manage.py migrate
python manage.py runserver
```

Open `http://127.0.0.1:8000/`.

## Worker endpoint

By default, backend uses:

- `https://dry-butterfly-fc08.sasimankan12man.workers.dev/`

Override with env var:

```bash
export FLUX_WORKER_URL="https://your-worker-url/"
```

## Usage flow

1. Upload image.
2. Paint white mask over text/object to remove.
3. Click **Clean Selected Area**.
4. If needed, paint another area and clean again.
5. Download final PNG.


## Upload size limits

Django is configured for large vertical pages with:

- `DATA_UPLOAD_MAX_MEMORY_SIZE = 80 MB`
- `FILE_UPLOAD_MAX_MEMORY_SIZE = 80 MB`

If you still hit `RequestDataTooBig`, raise these values in `web_cleaner/settings.py`.
