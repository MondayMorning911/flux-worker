import runpod
import base64
import torch
from diffusers import FluxPipeline
from io import BytesIO

MODEL_ID = "black-forest-labs/FLUX.1-schnell"

print("🚀 Loading model...")

pipe = FluxPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    use_auth_token=True
).to("cuda")

pipe.enable_model_cpu_offload()

print("✅ Model loaded")


def handler(event):
    prompt = event["input"]["prompt"]

    image = pipe(
        prompt=prompt,
        num_inference_steps=20,
        guidance_scale=3.5,
        width=1024,
        height=1024
    ).images[0]

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode()

    return {"image": img_base64}


runpod.serverless.start({"handler": handler})