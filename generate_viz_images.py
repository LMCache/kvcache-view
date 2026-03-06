"""Generate thumbnail images for the visualizations hub page via OpenAI gpt-image-1."""

import base64
import json
import os
import sys

VISUALIZATIONS = [
    {
        "id": "kvcache",
        "file": "thumbnails/kvcache.png",
        "prompt": (
            "Abstract dark visualization on navy-charcoal background. "
            "Glowing cyan and blue memory blocks stacking upward like a tower, "
            "representing GPU memory filling with KV cache data. "
            "Small orange particles streaming between layers. "
            "Futuristic data-center aesthetic. No text, no letters, no words."
        ),
    },
    {
        "id": "train",
        "file": "thumbnails/train.png",
        "prompt": (
            "Abstract dark visualization on deep charcoal background. "
            "A neural network training loop shown as concentric glowing rings "
            "in blue and purple, with gradient arrows flowing between them. "
            "Memory usage bars rising on the side in cyan. "
            "Futuristic scientific aesthetic. No text, no letters, no words."
        ),
    },
    {
        "id": "projections",
        "file": "thumbnails/projections.png",
        "prompt": (
            "Abstract dark visualization on navy background. "
            "An exponential growth curve rendered as a glowing cyan line "
            "shooting upward against a grid of faint blue lines, "
            "with timeline markers as small orange dots along the bottom. "
            "Data projection aesthetic. No text, no letters, no words."
        ),
    },
    {
        "id": "orchestrators",
        "file": "thumbnails/orchestrators.png",
        "prompt": (
            "Abstract dark visualization on charcoal-navy background. "
            "Four interconnected glowing nodes in orange, teal, purple, "
            "and blue, with data streams flowing between them like a network. "
            "Each node pulses with different intensity. "
            "Distributed systems aesthetic. No text, no letters, no words."
        ),
    },
    {
        "id": "engram",
        "file": "thumbnails/engram.png",
        "prompt": (
            "Abstract dark visualization on deep navy background. "
            "A Zipfian distribution curve glowing in cyan, with the tall head "
            "on the left tapering into a long tail on the right. "
            "Bright orange highlights on the most-accessed tokens at the peak. "
            "Statistical data aesthetic. No text, no letters, no words."
        ),
    },
    {
        "id": "agentic",
        "file": "thumbnails/agentic.png",
        "prompt": (
            "Abstract dark visualization on charcoal background. "
            "Multiple AI agent icons represented as glowing blue orbs "
            "connected by cyan data streams, with cache blocks being "
            "offloaded downward into tiered storage layers shown as "
            "translucent purple shelves. No text, no letters, no words."
        ),
    },
]


def _get_openai_api_key():
    """Get OpenAI API key from environment or auth file."""
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    try:
        with open(os.path.expanduser("~/.codex/auth.json")) as f:
            key = json.load(f).get("OPENAI_API_KEY")
    except Exception:
        pass
    return key


def generate_image(prompt, output_path):
    """Generate a thumbnail image via the OpenAI gpt-image-1 API."""
    key = _get_openai_api_key()
    if not key:
        print("[Image] Skipping: OPENAI_API_KEY not set", file=sys.stderr)
        return None

    import openai

    client = openai.OpenAI(api_key=key)

    try:
        print(f"[Image] Generating {output_path}...", file=sys.stderr)
        result = client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            n=1,
            size="1024x1024",
            quality="medium",
        )

        image_data = result.data[0]
        if hasattr(image_data, "b64_json") and image_data.b64_json:
            img_bytes = base64.b64decode(image_data.b64_json)
            with open(output_path, "wb") as f:
                f.write(img_bytes)
        elif hasattr(image_data, "url") and image_data.url:
            import urllib.request

            urllib.request.urlretrieve(image_data.url, output_path)
        else:
            print("[Image] No image data in response", file=sys.stderr)
            return None

        fsize = os.path.getsize(output_path)
        print(f"[Image] Generated: {output_path} ({fsize // 1024}KB)", file=sys.stderr)
        return output_path

    except Exception as e:
        print(f"[Image] Generation failed: {e}", file=sys.stderr)
        return None


def main():
    os.makedirs("thumbnails", exist_ok=True)

    force = "--force" in sys.argv
    target_id = None
    for arg in sys.argv[1:]:
        if arg != "--force":
            target_id = arg
            break

    for viz in VISUALIZATIONS:
        if target_id and viz["id"] != target_id:
            continue
        if not force and os.path.exists(viz["file"]):
            print(f"[Image] Skipping {viz['id']} (already exists)", file=sys.stderr)
            continue
        generate_image(viz["prompt"], viz["file"])


if __name__ == "__main__":
    main()
