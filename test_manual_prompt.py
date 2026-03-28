"""
Test with manual prompt (bypass Florence2) + with/without clothing ref
"""
import json, time, urllib.request, urllib.error

COMFYUI_API = "http://127.0.0.1:8000"

def comfy_post(endpoint, data):
    req = urllib.request.Request(
        f"{COMFYUI_API}{endpoint}",
        data=json.dumps(data).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read())

def comfy_get_history(prompt_id):
    try:
        with urllib.request.urlopen(f"{COMFYUI_API}/history/{prompt_id}", timeout=10) as resp:
            data = json.loads(resp.read())
            return data.get(prompt_id)
    except:
        return None

def poll_result(prompt_id, timeout=300):
    start = time.time()
    while time.time() - start < timeout:
        result = comfy_get_history(prompt_id)
        if result:
            return result
        time.sleep(3)
    return None

def build_manual_prompt_workflow(person_img, clothing_img, face_img, full_prompt, prefix, use_clothing_ref=True):
    """Workflow with manual prompt - no Florence2."""
    workflow = {
        "1": {"class_type": "LoadImage", "inputs": {"image": person_img}},
        "2": {"class_type": "LoadImage", "inputs": {"image": clothing_img}},
        "3": {"class_type": "LoadImage", "inputs": {"image": face_img}},
        "4": {"class_type": "UNETLoader", "inputs": {
            "unet_name": "flux-2-klein-base-4b-fp8.safetensors", "weight_dtype": "fp8_e4m3fn"
        }},
        "5": {"class_type": "CLIPLoader", "inputs": {
            "clip_name": "qwen_3_4b.safetensors", "type": "flux2", "device": "default"
        }},
        "6": {"class_type": "VAELoader", "inputs": {"vae_name": "flux2-vae.safetensors"}},
        # Direct text - no Florence2
        "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["5", 0], "text": full_prompt}},
        "8": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["5", 0], "text": ""}},
        "9": {"class_type": "ImageScaleToTotalPixels", "inputs": {
            "image": ["1", 0], "upscale_method": "nearest-exact", "megapixels": 1.0, "resolution_steps": 1
        }},
        "11": {"class_type": "GetImageSize", "inputs": {"image": ["9", 0]}},
        "14": {"class_type": "VAEEncode", "inputs": {"pixels": ["9", 0], "vae": ["6", 0]}},
        "12": {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["7", 0], "latent": ["14", 0]}},
        "13": {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["8", 0], "latent": ["14", 0]}},
    }

    if use_clothing_ref:
        workflow["10"] = {"class_type": "ImageScaleToTotalPixels", "inputs": {
            "image": ["2", 0], "upscale_method": "nearest-exact", "megapixels": 1.0, "resolution_steps": 1
        }}
        workflow["17"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["10", 0], "vae": ["6", 0]}}
        workflow["15"] = {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["12", 0], "latent": ["17", 0]}}
        workflow["16"] = {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["13", 0], "latent": ["17", 0]}}
        cfg_pos, cfg_neg = ["15", 0], ["16", 0]
    else:
        cfg_pos, cfg_neg = ["12", 0], ["13", 0]

    workflow.update({
        "18": {"class_type": "CFGGuider", "inputs": {
            "model": ["4", 0], "positive": cfg_pos, "negative": cfg_neg, "cfg": 5.0
        }},
        "19": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "20": {"class_type": "Flux2Scheduler", "inputs": {
            "steps": 20, "width": ["11", 0], "height": ["11", 1]
        }},
        "21": {"class_type": "RandomNoise", "inputs": {"noise_seed": int(time.time()) % 999999}},
        "22": {"class_type": "EmptyFlux2LatentImage", "inputs": {
            "width": ["11", 0], "height": ["11", 1], "batch_size": 1
        }},
        "23": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["21", 0], "guider": ["18", 0], "sampler": ["19", 0],
            "sigmas": ["20", 0], "latent_image": ["22", 0]
        }},
        "24": {"class_type": "VAEDecode", "inputs": {"samples": ["23", 0], "vae": ["6", 0]}},
        "26": {"class_type": "VRAM_Debug", "inputs": {
            "empty_cache": True, "gc_collect": True, "unload_all_models": True, "any_input": ["24", 0]
        }},
        "27": {"class_type": "ReActorFaceSwap", "inputs": {
            "input_image": ["26", 0], "source_image": ["3", 0], "enabled": True,
            "swap_model": "inswapper_128.onnx", "facedetection": "retinaface_resnet50",
            "face_restore_model": "GFPGANv1.4.pth", "face_restore_visibility": 1.0,
            "codeformer_weight": 0.5, "detect_gender_input": "no", "detect_gender_source": "no",
            "input_faces_index": "0", "source_faces_index": "0", "console_log_level": 1
        }},
        "29": {"class_type": "SaveImage", "inputs": {"images": ["27", 0], "filename_prefix": prefix}},
    })
    return workflow


PERSON = "tg_1072576182_person_1774503611.jpg"
CLOTHING = "tg_1072576182_clothing_1774503616.jpg"

# Manual prompt - only describe clothing, zero background/scene description
PROMPT_E = (
    "Change ONLY the clothing. Keep the exact same face, hair, pose, body, background, and lighting. "
    "New outfit: grey-green mesh halter neck lingerie bralette with criss-cross straps and matching mesh panty"
)

PROMPT_F = (
    "Virtual try-on. Keep original background, pose, and body exactly. "
    "New garment: sheer mesh halter bralette with geometric criss-cross strap design, matching sheer mesh bikini bottom, grey-green color"
)

if __name__ == "__main__":
    tests = [
        # E: manual prompt + with ref
        ("testE_ref_manual",   True,  PROMPT_E),
        # F: manual prompt + no ref
        ("testF_noref_manual", False, PROMPT_F),
    ]

    for prefix, use_ref, prompt in tests:
        print(f"\n{'='*50}")
        print(f"Running: {prefix} (clothing_ref={use_ref})")
        print(f"Prompt: {prompt[:80]}...")
        print(f"{'='*50}")

        wf = build_manual_prompt_workflow(PERSON, CLOTHING, PERSON, prompt,
                                          prefix=prefix, use_clothing_ref=use_ref)
        try:
            resp = comfy_post("/prompt", {"prompt": wf})
        except urllib.error.HTTPError as he:
            body = he.read().decode()
            print(f"FAIL HTTP {he.code}: {body[:500]}")
            continue
        except Exception as e:
            print(f"FAIL: {e}")
            continue

        pid = resp.get("prompt_id")
        print(f"Submitted: {pid}")

        result = poll_result(pid, timeout=300)
        if result:
            images = result.get("outputs", {}).get("29", {}).get("images", [])
            if images:
                print(f"OK Done: {images[0]['filename']}")
            else:
                status = result.get("status", {})
                print(f"FAIL No output: {status.get('status_str')}")
                for m in status.get("messages", []):
                    if m[0] == "execution_error":
                        print(f"  {m[1].get('exception_message','')[:300]}")
        else:
            print("FAIL Timeout")

    print("\nDone!")
