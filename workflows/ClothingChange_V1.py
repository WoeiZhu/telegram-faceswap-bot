"""
ClothingChange_V1 — Qwen Image Edit 兩階段換裝工作流

階段 1：Outfit Extractor — 從服裝參考圖提取純服裝（白底）
階段 2：Outfit Transfer — 將提取的服裝轉移到目標人物

所需模型：
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
