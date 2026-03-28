"""
KleinVTON_V1 — Klein 4B VTON + Florence2 自動服裝描述 + ReActor 換臉

流程：
1. 載入人物圖、服裝參考圖、臉孔圖
2. Florence2 自動描述服裝圖（更詳細的服裝描述）
3. Prompt = 指令模板 + Florence2 描述
4. Klein 4B + ReferenceLatent 換裝
   - RefLatent(人物, index=10) → RefLatent(服裝, index=20)
   - CFGGuider → SamplerCustomAdvanced → VAEDecode
5. VRAM_Debug 清理 GPU 記憶體
6. ReActorFaceSwap 換臉（保留人物原臉或換上臉孔圖的臉）
7. ReActorRestoreFace 臉部修復

所需模型：
- diffusion_models/flux-2-klein-base-4b-fp8.safetensors
- text_encoders/qwen_3_4b.safetensors
- vae/flux2-vae.safetensors
- insightface/inswapper_128.onnx
- facerestore_models/GFPGANv1.4.pth
- florence2-base（自動下載 ~2.4GB）

參數：CFG=5.0 | Steps=20 | Sampler=euler | 1MP
"""
import time


INSTRUCTION = (
    "Change this person's clothing. Keep the original face, hairstyle, "
    "pose, body shape, and background exactly the same. "
    "New outfit: "
)


def build_klein_vton_workflow(person_img: str, clothing_img: str, face_img: str) -> dict:
    """
    Klein VTON + ReActor 換臉工作流

    Args:
        person_img:   人物圖（要換裝的人，保留姿勢/背景/體型）
        clothing_img: 服裝參考圖（Florence2 自動描述）
        face_img:     臉孔來源圖（ReActor 換臉）
    """
    return {
        # ═══════════════════════════════════════════
        # 載入圖片
        # ═══════════════════════════════════════════
        "1": {
            "class_type": "LoadImage",
            "inputs": {"image": person_img}
        },
        "2": {
            "class_type": "LoadImage",
            "inputs": {"image": clothing_img}
        },
        "3": {
            "class_type": "LoadImage",
            "inputs": {"image": face_img}
        },

        # ═══════════════════════════════════════════
        # 載入模型
        # ═══════════════════════════════════════════
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

        # ═══════════════════════════════════════════
        # Florence2 自動描述服裝
        # ═══════════════════════════════════════════
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

        # ═══════════════════════════════════════════
        # 組合 Prompt
        # ═══════════════════════════════════════════
        "35": {
            "class_type": "Text String",
            "inputs": {"text": INSTRUCTION}
        },
        "34": {
            "class_type": "CR Text Concatenate",
            "inputs": {
                "text1": ["35", 0],
                "text2": ["32", 0],
                "separator": ""
            }
        },

        # ═══════════════════════════════════════════
        # Clip Text Encode（正向 + 負向）
        # ═══════════════════════════════════════════
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": ["5", 0],
                "text": ["34", 0]
            }
        },
        "8": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": ["5", 0],
                "text": ""
            }
        },

        # ═══════════════════════════════════════════
        # 縮放圖片（1MP）
        # ═══════════════════════════════════════════
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

        # ═══════════════════════════════════════════
        # VAE Encode
        # ═══════════════════════════════════════════
        "14": {
            "class_type": "VAEEncode",
            "inputs": {
                "pixels": ["9", 0],
                "vae": ["6", 0]
            }
        },
        "17": {
            "class_type": "VAEEncode",
            "inputs": {
                "pixels": ["10", 0],
                "vae": ["6", 0]
            }
        },

        # ═══════════════════════════════════════════
        # ReferenceLatent（人物 index=10 → 服裝 index=20）
        # ═══════════════════════════════════════════
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

        # ═══════════════════════════════════════════
        # Sampler 設定
        # ═══════════════════════════════════════════
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

        # ═══════════════════════════════════════════
        # VAE Decode
        # ═══════════════════════════════════════════
        "24": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["23", 0],
                "vae": ["6", 0]
            }
        },

        # ═══════════════════════════════════════════
        # VRAM 清理（換臉前先釋放 GPU 記憶體）
        # ═══════════════════════════════════════════
        "26": {
            "class_type": "VRAM_Debug",
            "inputs": {
                "empty_cache": True,
                "gc_collect": True,
                "unload_all_models": True,
                "any_input": ["24", 0]
            }
        },

        # ═══════════════════════════════════════════
        # ReActor 換臉
        # ═══════════════════════════════════════════
        "27": {
            "class_type": "ReActorFaceSwap",
            "inputs": {
                "enabled": True,
                "input_image": ["26", 0],
                "source_image": ["3", 0],
                "swap_model": "inswapper_128.onnx",
                "facedetection": "retinaface_resnet50",
                "face_restore_model": "GFPGANv1.4.pth",
                "face_restore_visibility": 1.0,
                "codeformer_weight": 0.5,
                "detect_gender_input": "no",
                "detect_gender_source": "no",
                "input_faces_index": "0",
                "source_faces_index": "0",
                "console_log_level": 1
            }
        },

        # ═══════════════════════════════════════════
        # 臉部修復
        # ═══════════════════════════════════════════
        "28": {
            "class_type": "ReActorRestoreFace",
            "inputs": {
                "image": ["27", 0],
                "facedetection": "retinaface_resnet50",
                "model": "GFPGANv1.4.pth",
                "visibility": 1.0,
                "codeformer_weight": 0.5
            }
        },

        # ═══════════════════════════════════════════
        # 儲存結果
        # ═══════════════════════════════════════════
        "29": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["28", 0],
                "filename_prefix": "klein_vton_v1"
            }
        },
    }
