export default {
  async fetch(request, env) {
    if (request.method === "OPTIONS") {
      return new Response(null, { headers: corsHeaders() });
    }

    try {
      if (request.method !== "POST") {
        return json({ ok: false, error: "Use POST" }, 405);
      }

      const body = await request.json().catch(() => null);
      if (!body || !body.prompt || !body.image) {
        return json({ ok: false, error: "Missing prompt or image" }, 400);
      }

      const model = typeof body.model === "string" && body.model.trim()
        ? body.model.trim()
        : "@cf/black-forest-labs/flux-2-klein-4b";

      const width = clampInt(body.width, 64, 1024, 512);
      const height = clampInt(body.height, 64, 1024, 512);
      const guidance = clampFloat(body.guidance, 1.0, 6.0, 2.0);

      const form = new FormData();
      form.append("input_image_0", dataUrlToBlob(body.image), "input0.png");
      if (body.image1) form.append("input_image_1", dataUrlToBlob(body.image1), "input1.png");
      if (body.image2) form.append("input_image_2", dataUrlToBlob(body.image2), "input2.png");
      if (body.image3) form.append("input_image_3", dataUrlToBlob(body.image3), "input3.png");

      form.append("prompt", String(body.prompt));
      form.append("width", String(width));
      form.append("height", String(height));
      form.append("guidance", String(guidance));

      if (body.seed !== undefined && body.seed !== null) {
        form.append("seed", String(parseInt(body.seed, 10)));
      }

      const url = `https://api.cloudflare.com/client/v4/accounts/${env.CF_ACCOUNT_ID}/ai/run/${model}`;
      const aiRes = await fetch(url, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${env.CF_API_TOKEN}`,
        },
        body: form,
      });

      const reqId = aiRes.headers.get("cf-ray") || aiRes.headers.get("x-request-id") || "";
      const ct = (aiRes.headers.get("content-type") || "").toLowerCase();

      if (!aiRes.ok) {
        const msg = await aiRes.text().catch(() => "");
        return json({
          ok: false,
          error: `AI error ${aiRes.status}`,
          model,
          request_id: reqId,
          content_type: ct,
          details: msg.slice(0, 12000),
        }, 500);
      }

      if (ct.includes("application/json")) {
        const data = await aiRes.json();
        const result = data?.result ?? data;

        const b64 =
          pickB64(result?.image) ||
          pickB64(result?.output?.image) ||
          pickB64(Array.isArray(result?.images) ? result.images[0] : null);

        if (b64) {
          const bytes = base64ToUint8(b64);
          return new Response(bytes, {
            headers: {
              ...corsHeaders(),
              "content-type": guessMimeFromBase64(b64),
              "cache-control": "no-store",
              ...(reqId ? { "x-upstream-request-id": reqId } : {}),
            },
          });
        }

        return json({ ok: true, note: "No image payload found", model, request_id: reqId, data }, 200);
      }

      if (ct.startsWith("image/")) {
        const bytes = await aiRes.arrayBuffer();
        return new Response(bytes, {
          headers: {
            ...corsHeaders(),
            "content-type": ct,
            "cache-control": "no-store",
            ...(reqId ? { "x-upstream-request-id": reqId } : {}),
          },
        });
      }

      const text = await aiRes.text().catch(() => "");
      return json({ ok: true, note: "Non-image response", content_type: ct, preview: text.slice(0, 2000) }, 200);
    } catch (err) {
      return json({ ok: false, error: err?.message ?? String(err), stack: err?.stack ?? null }, 500);
    }
  },
};

function pickB64(value) {
  if (typeof value !== "string") return null;
  const raw = value.trim();
  if (!raw) return null;
  if (raw.startsWith("data:")) {
    const parts = raw.split(",");
    return parts.length === 2 ? parts[1] : null;
  }
  return raw;
}

function clampInt(v, lo, hi, fallback) {
  const n = Number.parseInt(v, 10);
  if (!Number.isFinite(n)) return fallback;
  return Math.max(lo, Math.min(hi, n));
}

function clampFloat(v, lo, hi, fallback) {
  const n = Number.parseFloat(v);
  if (!Number.isFinite(n)) return fallback;
  return Math.max(lo, Math.min(hi, n));
}

function corsHeaders() {
  return {
    "access-control-allow-origin": "*",
    "access-control-allow-methods": "POST,OPTIONS",
    "access-control-allow-headers": "content-type",
  };
}

function json(obj, status = 200) {
  return new Response(JSON.stringify(obj), {
    status,
    headers: {
      ...corsHeaders(),
      "content-type": "application/json",
      "cache-control": "no-store",
    },
  });
}

function dataUrlToBlob(dataUrl) {
  const parts = String(dataUrl).split(",");
  if (parts.length !== 2) {
    throw new Error("Invalid data URL");
  }
  const meta = parts[0];
  const b64 = parts[1];
  const mm = meta.match(/^data:(.*);base64$/i);
  const mime = mm ? mm[1] : "application/octet-stream";
  return new Blob([base64ToUint8(b64)], { type: mime });
}

function guessMimeFromBase64(b64) {
  if (b64.startsWith("/9j/")) return "image/jpeg";
  if (b64.startsWith("iVBOR")) return "image/png";
  if (b64.startsWith("UklGR")) return "image/webp";
  return "application/octet-stream";
}

function base64ToUint8(b64) {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i += 1) out[i] = bin.charCodeAt(i);
  return out;
}
