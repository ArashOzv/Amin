import base64
import io
from unittest.mock import Mock, patch

from django.test import Client, TestCase, override_settings
from PIL import Image


def _img_data_url(color=(0, 0, 0, 255), size=(8, 8), mask=False):
    img = Image.new("RGBA", size, color)
    if mask:
        img = Image.new("RGBA", size, (0, 0, 0, 0))
        px = img.load()
        px[2, 2] = (255, 255, 255, 255)
    else:
        # Explicit transparent/blank mask to avoid platform-dependent defaults.
        img = Image.new("RGBA", size, (0, 0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


class CleanSelectionTests(TestCase):
    def setUp(self):
        self.client = Client(enforce_csrf_checks=False)

    def test_empty_mask_returns_400(self):
        image = _img_data_url()
        empty_mask = _img_data_url(mask=False)
        res = self.client.post("/api/clean/", {"image": image, "mask": empty_mask, "guidance": "2.0"})
        self.assertEqual(res.status_code, 400)
        self.assertIn("Mask is empty", res.json()["error"])


    @override_settings(DATA_UPLOAD_MAX_MEMORY_SIZE=128)
    def test_request_too_large_returns_413(self):
        huge = "x" * 2048
        res = self.client.post("/api/clean/", {"image": huge, "mask": huge, "guidance": "2.0"})
        self.assertEqual(res.status_code, 413)
        body = res.json()
        self.assertEqual(body["type"], "request_too_large")

    @patch("web_cleaner.core.views.requests.post")
    def test_worker_json_error_returns_502(self, mock_post):
        image = _img_data_url()
        mask = _img_data_url(mask=True)

        fake = Mock()
        fake.ok = True
        fake.headers = {"content-type": "application/json"}
        fake.json.return_value = {"ok": False, "error": "AI error 500", "details": "upstream failed"}
        fake.text = '{"ok":false}'
        mock_post.return_value = fake

        res = self.client.post("/api/clean/", {"image": image, "mask": mask, "guidance": "2.0"})
        self.assertEqual(res.status_code, 502)
        self.assertEqual(res.json()["type"], "worker_error")


    @patch("web_cleaner.core.views.requests.post")
    def test_worker_http_error_returns_502(self, mock_post):
        image = _img_data_url(color=(10, 10, 10, 255), mask=False)
        mask = _img_data_url(mask=True)

        fake = Mock()
        fake.ok = False
        fake.status_code = 500
        fake.headers = {"content-type": "application/json", "x-upstream-request-id": "abc123"}
        fake.json.return_value = {"ok": False, "error": "AI error 500", "details": "bad upstream"}
        fake.text = '{"ok":false}'
        mock_post.return_value = fake

        res = self.client.post("/api/clean/", {"image": image, "mask": mask, "guidance": "2.0"})
        self.assertEqual(res.status_code, 502)
        self.assertEqual(res.json()["type"], "worker_error")

    @patch("web_cleaner.core.views.requests.post")
    def test_worker_image_success(self, mock_post):
        image = _img_data_url(color=(10, 20, 30, 255))
        mask = _img_data_url(mask=True)

        out = Image.new("RGBA", (512, 512), (200, 100, 50, 255))
        buf = io.BytesIO()
        out.save(buf, format="PNG")

        fake = Mock()
        fake.ok = True
        fake.headers = {"content-type": "image/png"}
        fake.content = buf.getvalue()
        fake.text = ""
        mock_post.return_value = fake

        res = self.client.post("/api/clean/", {"image": image, "mask": mask, "guidance": "2.0"})
        body = res.json()
        self.assertEqual(res.status_code, 200)
        self.assertTrue(body["ok"])
        self.assertTrue(body["result"].startswith("data:image/png;base64,"))
