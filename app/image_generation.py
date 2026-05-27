"""Image generation for the final scenario.

Uses OpenAI's image API when an API key is configured; otherwise returns
a "mock" result so the rest of the pipeline keeps working. Failures are
caught and surfaced via `ImageResult.error` - they never crash a run.
"""
from __future__ import annotations

import base64
import os
from typing import Optional

from . import config as _config_mod
from .config import Config
from .schemas import ImageResult


SAFETY_SUFFIX = (
    " Non-graphic, editorial style. No weapons close-ups, no targeting "
    "imagery, no gore, no operational military detail."
)


def build_image_prompt(scenario_title: str, scenario_summary: str) -> str:
    """Construct a safe editorial-illustration prompt."""
    base = (
        "A cinematic geopolitical editorial illustration of the Pacific region. "
        "Washington, Beijing, and Taipei connected by glowing trade routes, "
        "semiconductor circuits, and diplomatic chess pieces. "
        "Distant non-violent naval silhouettes, divided technology networks, "
        "serious dark-blue tone, high detail. "
        "Scenario theme: " + (scenario_title or "USA-China strategic rivalry").strip()
        + ". Summary: " + (scenario_summary or "").strip()
    )
    return base + SAFETY_SUFFIX


def generate_image(
    run_id: str,
    prompt: str,
    config: Optional[Config] = None,
) -> ImageResult:
    """Generate an image and save it under data/generated_images/<run_id>.png.

    Returns an `ImageResult` describing what happened. Never raises.
    """
    cfg = config or _config_mod.CONFIG

    result = ImageResult(enabled=cfg.enable_image_generation)
    if not cfg.enable_image_generation:
        return result

    out_dir = cfg.generated_images_dir
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, run_id + ".png")

    if cfg.mock_mode:
        # Write a tiny placeholder so the frontend has something to render.
        try:
            _write_placeholder_png(out_path)
            result.path = out_path
            result.generated = True
            result.mock = True
        except Exception as e:
            result.error = "mock_write_failed: " + str(e)
        return result

    try:
        from openai import OpenAI

        client = OpenAI(api_key=cfg.openai_api_key)
        resp = client.images.generate(
            model=cfg.openai_image_model,
            prompt=prompt,
            size="1024x1024",
            n=1,
        )
        data = resp.data[0]
        b64 = getattr(data, "b64_json", None)
        if b64:
            with open(out_path, "wb") as fh:
                fh.write(base64.b64decode(b64))
            result.path = out_path
            result.generated = True
            return result

        url = getattr(data, "url", None)
        if url:
            import urllib.request

            urllib.request.urlretrieve(url, out_path)  # nosec - trusted source
            result.path = out_path
            result.generated = True
            return result

        result.error = "no_image_data_returned"
        return result
    except Exception as e:
        result.error = type(e).__name__ + ": " + str(e)
        return result


# A 1x1 transparent PNG used as the mock-mode placeholder so the frontend
# has a real file to display when no API key is configured.
_PLACEHOLDER_PNG = base64.b64decode(
    b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def _write_placeholder_png(path: str) -> None:
    with open(path, "wb") as fh:
        fh.write(_PLACEHOLDER_PNG)
