"""
ClothingChange_V2 — FLUX.1-fill + Redux 視覺參考換裝工作流

核心技術：
  - FLUX.1-fill-dev（官方 inpainting 模型）
  - FLUX Redux（flux1-redux-dev）直接以服裝圖作視覺條件，不靠文字轉換
  - LayerMask: SegmentAnythingUltra V2 自動偵測服裝遮罩
  - InpaintModelConditioning（FLUX Fill 正確做法）

所需模型（已全部安裝）：
  - diffusion_models/flux1-fill-dev.safetensors
  - style_models/flux1-redux-dev.safetensors
  - clip_vision/sigclip_vision_patch14_384.safetensors
  - text_encoders/t5xxl_fp16.safetensors
  - text_encoders/clip_l.safetensors
  - vae/ae.safetensors

效能（RTX 4070 SUPER 12GB）：
  - 預估每張約 60~120 秒
"""
import time

# GroundingDINO 偵測服裝的文字提示
_GROUNDING_PROMPT = (
    "clothing . shirt . pants . dress . jacket . top . "
    "skirt . outfit . bikini . underwear . coat . blouse"
)

# 換裝指令前綴，後接 Florence2 自動描述的服裝款式
_CLOTHING_INSTRUCTION = (
    "Change this person's clothing. Keep the original face, hairstyle, "
    "pose, body shape, and background exactly the same. "
    "New outfit: "
)


def build_flux_inpaint_clothing_workflow(person_img: str, clothing_img: str) -> dict:
    """
    FLUX.1-fill + Redux 換裝（視覺參考 + 自動遮罩 + 局部重繪）

    Args:
        person_img:   人物圖（保留姿勢/背景/體型/臉）
        clothing_img: 服裝參考圖（Redux 直接視覺參考，不需要文字描述）
    """
    return {
        # ── 載入圖片 ─────────────────────────────────────────────────
        "1": {
            "class_type": "LoadImage",
            "inputs": {"image": person_img}
        },
        "2": {
            "class_type": "LoadImage",
            "inputs": {"image": clothing_img}
        },

        # ── 縮放人物圖至 1MP（確保尺寸為 64 的倍數）────────────────────
        "3": {
            "class_type": "ImageScaleToTotalPixels",
            "inputs": {
                "image": ["1", 0],
                "upscale_method": "lanczos",
                "megapixels": 1.0,
                "resolution_steps": 64
            }
        },

        # ── 載入主模型 ────────────────────────────────────────────────
        "10": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": "flux1-fill-dev.safetensors",
                "weight_dtype": "fp8_e4m3fn"
            }
        },
        "11": {
            "class_type": "DualCLIPLoader",
            "inputs": {
                "clip_name1": "t5xxl_fp16.safetensors",
                "clip_name2": "clip_l.safetensors",
                "type": "flux",
                "device": "default"
            }
        },
        "12": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "ae.safetensors"}
        },

        # ── Redux 視覺參考：直接用服裝圖作條件 ───────────────────────
        # CLIPVision 編碼服裝圖
        "13": {
            "class_type": "CLIPVisionLoader",
            "inputs": {"clip_name": "sigclip_vision_patch14_384.safetensors"}
        },
        "14": {
            "class_type": "CLIPVisionEncode",
            "inputs": {
                "clip_vision": ["13", 0],
                "image": ["2", 0],
                "crop": "center"
            }
        },
        # Redux Style Model
        "15": {
            "class_type": "StyleModelLoader",
            "inputs": {"style_model_name": "flux1-redux-dev.safetensors"}
        },

        # ── SegmentAnythingUltra V2 自動生成服裝遮罩 ──────────────────
        "22": {
            "class_type": "LayerMask: SegmentAnythingUltra V2",
            "inputs": {
                "image": ["3", 0],
                "sam_model": "sam_vit_b (375MB)",
                "grounding_dino_model": "GroundingDINO_SwinT_OGC (694MB)",
                "threshold": 0.3,
                "detail_method": "GuidedFilter",
                "detail_erode": 6,
                "detail_dilate": 6,
                "black_point": 0.15,
                "white_point": 0.99,
                "process_detail": False,
                "prompt": _GROUNDING_PROMPT,
                "device": "cuda",
                "max_megapixels": 2.0,
                "cache_model": True
            }
        },
        # 遮罩邊緣膨脹，讓重繪邊界融合更自然
        "23": {
            "class_type": "GrowMask",
            "inputs": {
                "mask": ["22", 1],
                "expand": 12,
                "tapered_corners": True
            }
        },

        # ── Florence2 自動描述服裝款式 ────────────────────────────────
        "30": {
            "class_type": "LayerMask: LoadFlorence2Model",
            "inputs": {"version": "base"}
        },
        "31": {
            "class_type": "LayerUtility: Florence2Image2Prompt",
            "inputs": {
                "florence2_model": ["30", 0],
                "image": ["2", 0],
                "task": "more detailed caption",
                "text_input": (
                    "Describe only the clothing and outfit in detail. "
                    "Focus on garment type, color, fabric, pattern, fit, style, "
                    "and construction details. "
                    "Do not describe person, face, pose, or background."
                ),
                "max_new_tokens": 1024,
                "num_beams": 3,
                "do_sample": False,
                "fill_mask": False
            }
        },
        # 指令前綴 + Florence2 描述組合
        "35": {
            "class_type": "Text String",
            "inputs": {"text": _CLOTHING_INSTRUCTION}
        },
        "34": {
            "class_type": "CR Text Concatenate",
            "inputs": {"text1": ["35", 0], "text2": ["31", 0], "separator": ""}
        },

        # ── Text Encode ───────────────────────────────────────────────
        "40": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["11", 0], "text": ["34", 0]}
        },
        "41": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["11", 0], "text": ""}
        },

        # ── FluxGuidance ──────────────────────────────────────────────
        "42": {
            "class_type": "FluxGuidance",
            "inputs": {
                "conditioning": ["40", 0],
                "guidance": 3.5
            }
        },

        # ── Redux 視覺條件注入（strength=0.5 輔助，文字為主）───────────
        # Redux 負責顏色/材質，文字 prompt 負責款式細節
        "43": {
            "class_type": "StyleModelApply",
            "inputs": {
                "conditioning": ["42", 0],
                "style_model": ["15", 0],
                "clip_vision_output": ["14", 0],
                "strength": 0.5,
                "strength_type": "multiply"
            }
        },

        # ── InpaintModelConditioning（FLUX Fill 專用）────────────────
        "50": {
            "class_type": "InpaintModelConditioning",
            "inputs": {
                "positive": ["43", 0],
                "negative": ["41", 0],
                "vae": ["12", 0],
                "pixels": ["3", 0],
                "mask": ["23", 0],
                "noise_mask": True
            }
        },

        # ── Sampler ──────────────────────────────────────────────────
        "60": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["10", 0],
                "positive": ["50", 0],
                "negative": ["50", 1],
                "latent_image": ["50", 2],
                "seed": int(time.time()) % 999999999,
                "steps": 30,
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0
            }
        },

        # ── VAE Decode ───────────────────────────────────────────────
        "70": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["60", 0],
                "vae": ["12", 0]
            }
        },

        # ── 儲存結果 ─────────────────────────────────────────────────
        "80": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["70", 0],
                "filename_prefix": "clothing_change_v2"
            }
        },
    }
