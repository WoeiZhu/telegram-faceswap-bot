"""
ClothingChange_V1 — 換裝工作流

主路線（無限制）：build_klein_clothing_workflow
  Klein 4B + Florence2 自動描述服裝，無任何內容過濾
  適用所有服裝包含 NSFW / 性感 / 內衣

備用路線（Qwen，有安全過濾）：build_stage1_extract + build_stage2_transfer
  Qwen Image Edit 兩階段換裝，遇 NSFW 服裝可能被過濾

所需模型（Klein 路線）：
- diffusion_models/flux-2-klein-base-4b-fp8.safetensors
- text_encoders/qwen_3_4b.safetensors
- vae/flux2-vae.safetensors
- florence2-base（自動下載）

所需模型（Qwen 路線）：
- diffusion_models/qwen_image_edit_2509_fp8_e4m3fn.safetensors
- text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors
- vae/qwen_image_vae.safetensors
- loras/Qwen-Image-Lightning-4steps-V1.0.safetensors
- loras/extract-outfit_v3.safetensors
- loras/clothtransfer.safetensors
"""
import time


def build_stage1_extract(clothing_img: str) -> dict:
    """階段 1：從服裝圖提取純服裝到白底"""
    return {
        # ── 載入模型 ──
        "1": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": "qwen_image_edit_2509_fp8_e4m3fn.safetensors",
                "weight_dtype": "fp8_e4m3fn"
            }
        },
        # Lightning 4-step 加速 LoRA
        "2": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["1", 0],
                "lora_name": "Qwen-Image-Lightning-4steps-V1.0.safetensors",
                "strength_model": 1.0
            }
        },
        # Outfit Extractor LoRA
        "3": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["2", 0],
                "lora_name": "extract-outfit_v3.safetensors",
                "strength_model": 1.75
            }
        },
        # CLIP + VAE
        "4": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
                "type": "qwen_image",
                "device": "default"
            }
        },
        "5": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "qwen_image_vae.safetensors"}
        },
        # 載入服裝圖
        "6": {
            "class_type": "LoadImage",
            "inputs": {"image": clothing_img}
        },
        # 文字編碼 — 提取服裝
        "7": {
            "class_type": "TextEncodeQwenImageEditPlus",
            "inputs": {
                "clip": ["4", 0],
                "vae": ["5", 0],
                "image1": ["6", 0],
                "prompt": "extract the full body and the full outfit from front and from back onto a white background."
            }
        },
        # Negative conditioning
        "8": {
            "class_type": "ConditioningZeroOut",
            "inputs": {
                "conditioning": ["7", 0]
            }
        },
        # 空 latent
        "9": {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": 1024,
                "height": 1024,
                "batch_size": 1
            }
        },
        # KSampler
        "10": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["3", 0],
                "positive": ["7", 0],
                "negative": ["8", 0],
                "latent_image": ["9", 0],
                "seed": int(time.time()) % 999999,
                "steps": 6,
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0
            }
        },
        # VAE Decode
        "11": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["10", 0],
                "vae": ["5", 0]
            }
        },
        # 儲存提取結果
        "12": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["11", 0],
                "filename_prefix": "outfit_extracted"
            }
        },
    }


def build_stage2_transfer(person_img: str, extracted_outfit_img: str) -> dict:
    """階段 2：將提取的服裝轉移到目標人物"""
    return {
        # ── 載入模型 ──
        "1": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": "qwen_image_edit_2509_fp8_e4m3fn.safetensors",
                "weight_dtype": "fp8_e4m3fn"
            }
        },
        # Lightning 4-step 加速 LoRA
        "2": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["1", 0],
                "lora_name": "Qwen-Image-Lightning-4steps-V1.0.safetensors",
                "strength_model": 1.0
            }
        },
        # ClothTransfer LoRA
        "3": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["2", 0],
                "lora_name": "clothtransfer.safetensors",
                "strength_model": 1.0
            }
        },
        # CLIP + VAE
        "4": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
                "type": "qwen_image",
                "device": "default"
            }
        },
        "5": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "qwen_image_vae.safetensors"}
        },
        # 載入人物圖 + 提取的服裝圖
        "6": {
            "class_type": "LoadImage",
            "inputs": {"image": person_img}
        },
        "7": {
            "class_type": "LoadImage",
            "inputs": {"image": extracted_outfit_img}
        },
        # 縮放人物圖
        "8": {
            "class_type": "ImageScaleToTotalPixels",
            "inputs": {
                "image": ["6", 0],
                "upscale_method": "nearest-exact",
                "megapixels": 1.0,
                "resolution_steps": 1
            }
        },
        # 取得尺寸
        "9": {
            "class_type": "GetImageSize",
            "inputs": {"image": ["8", 0]}
        },
        # 文字編碼 — 轉移服裝
        "10": {
            "class_type": "TextEncodeQwenImageEditPlus",
            "inputs": {
                "clip": ["4", 0],
                "vae": ["5", 0],
                "image1": ["8", 0],
                "image2": ["7", 0],
                "prompt": "Transfer the outfit."
            }
        },
        # Negative conditioning
        "11": {
            "class_type": "ConditioningZeroOut",
            "inputs": {
                "conditioning": ["10", 0]
            }
        },
        # 空 latent（使用人物圖尺寸）
        "12": {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": ["9", 0],
                "height": ["9", 1],
                "batch_size": 1
            }
        },
        # KSampler
        "13": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["3", 0],
                "positive": ["10", 0],
                "negative": ["11", 0],
                "latent_image": ["12", 0],
                "seed": int(time.time()) % 999999,
                "steps": 8,
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0
            }
        },
        # VAE Decode
        "14": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["13", 0],
                "vae": ["5", 0]
            }
        },
        # 儲存結果
        "15": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["14", 0],
                "filename_prefix": "clothing_change_v1"
            }
        },
    }


# ═══════════════════════════════════════════════════════════════════
# Klein 無過濾換裝（主路線）
# 使用 Klein 4B + Florence2 自動服裝描述，無任何內容安全過濾
# 適用所有服裝類型包含性感/內衣/NSFW
# ═══════════════════════════════════════════════════════════════════

_CLOTHING_INSTRUCTION = (
    "Change this person's clothing. Keep the original face, hairstyle, "
    "pose, body shape, and background exactly the same. "
    "New outfit: "
)


def build_klein_clothing_workflow(person_img: str, clothing_img: str) -> dict:
    """
    Klein 4B 換裝（無內容過濾）

    Args:
        person_img:   人物圖（保留姿勢/背景/體型/臉）
        clothing_img: 服裝參考圖（Florence2 自動描述）
    """
    return {
        # ── 載入圖片 ──
        "1": {
            "class_type": "LoadImage",
            "inputs": {"image": person_img}
        },
        "2": {
            "class_type": "LoadImage",
            "inputs": {"image": clothing_img}
        },

        # ── 載入模型 ──
        "4": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": "flux-2-klein-base-4b-fp8.safetensors",
                "weight_dtype": "fp8_e4m3fn"
            }
        },
        "5": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": "qwen_3_4b.safetensors",
                "type": "flux2",
                "device": "default"
            }
        },
        "6": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "flux2-vae.safetensors"}
        },

        # ── Florence2 自動描述服裝 ──
        "31": {
            "class_type": "LayerMask: LoadFlorence2Model",
            "inputs": {"version": "base"}
        },
        "32": {
            "class_type": "LayerUtility: Florence2Image2Prompt",
            "inputs": {
                "florence2_model": ["31", 0],
                "image": ["2", 0],
                "task": "more detailed caption",
                "text_input": (
                    "Describe only the clothing and outfit in detail. "
                    "Focus on garment type, color, fabric, pattern, fit, style, and construction details. "
                    "Do not describe person, face, pose, or background."
                ),
                "max_new_tokens": 1024,
                "num_beams": 3,
                "do_sample": False,
                "fill_mask": False
            }
        },

        # ── 組合 Prompt ──
        "35": {
            "class_type": "Text String",
            "inputs": {"text": _CLOTHING_INSTRUCTION}
        },
        "34": {
            "class_type": "CR Text Concatenate",
            "inputs": {
                "text1": ["35", 0],
                "text2": ["32", 0],
                "separator": ""
            }
        },

        # ── CLIP Text Encode ──
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["5", 0], "text": ["34", 0]}
        },
        "8": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["5", 0], "text": ""}
        },

        # ── 縮放圖片（1MP）──
        "9": {
            "class_type": "ImageScaleToTotalPixels",
            "inputs": {
                "image": ["1", 0],
                "upscale_method": "nearest-exact",
                "megapixels": 1.0,
                "resolution_steps": 1
            }
        },
        "10": {
            "class_type": "ImageScaleToTotalPixels",
            "inputs": {
                "image": ["2", 0],
                "upscale_method": "nearest-exact",
                "megapixels": 1.0,
                "resolution_steps": 1
            }
        },
        "11": {
            "class_type": "GetImageSize",
            "inputs": {"image": ["9", 0]}
        },

        # ── VAE Encode ──
        "14": {
            "class_type": "VAEEncode",
            "inputs": {"pixels": ["9", 0], "vae": ["6", 0]}
        },
        "17": {
            "class_type": "VAEEncode",
            "inputs": {"pixels": ["10", 0], "vae": ["6", 0]}
        },

        # ── ReferenceLatent（人物 index=10 → 服裝 index=20）──
        "12": {
            "class_type": "ReferenceLatent",
            "inputs": {
                "conditioning": ["7", 0],
                "latent": ["14", 0],
                "ref_index_scale": 10.0
            }
        },
        "13": {
            "class_type": "ReferenceLatent",
            "inputs": {
                "conditioning": ["12", 0],
                "latent": ["17", 0],
                "ref_index_scale": 20.0
            }
        },
        "15": {
            "class_type": "ReferenceLatent",
            "inputs": {
                "conditioning": ["8", 0],
                "latent": ["14", 0],
                "ref_index_scale": 10.0
            }
        },
        "16": {
            "class_type": "ReferenceLatent",
            "inputs": {
                "conditioning": ["15", 0],
                "latent": ["17", 0],
                "ref_index_scale": 20.0
            }
        },

        # ── Sampler ──
        "18": {
            "class_type": "CFGGuider",
            "inputs": {
                "model": ["4", 0],
                "positive": ["13", 0],
                "negative": ["16", 0],
                "cfg": 5.0
            }
        },
        "19": {
            "class_type": "KSamplerSelect",
            "inputs": {"sampler_name": "euler"}
        },
        "20": {
            "class_type": "Flux2Scheduler",
            "inputs": {
                "steps": 20,
                "width": ["11", 0],
                "height": ["11", 1]
            }
        },
        "21": {
            "class_type": "RandomNoise",
            "inputs": {"noise_seed": int(time.time()) % 999999999}
        },
        "22": {
            "class_type": "EmptyFlux2LatentImage",
            "inputs": {
                "width": ["11", 0],
                "height": ["11", 1],
                "batch_size": 1
            }
        },
        "23": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["21", 0],
                "guider": ["18", 0],
                "sampler": ["19", 0],
                "sigmas": ["20", 0],
                "latent_image": ["22", 0]
            }
        },

        # ── VAE Decode ──
        "24": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["23", 0], "vae": ["6", 0]}
        },

        # ── 儲存結果 ──
        "29": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["24", 0],
                "filename_prefix": "clothing_change_klein"
            }
        },
    }
