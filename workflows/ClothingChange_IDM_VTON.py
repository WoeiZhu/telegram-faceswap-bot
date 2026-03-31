"""
ClothingChange_IDM_VTON — IDM-VTON 專用換裝工作流

流程：
1. 載入人物圖 + 服裝圖，縮放至 768x1024
2. DWPose 骨架擷取（全身偵測）
3. DensePose 3D 人體拓樸
4. GroundingDINO + SAM 自動偵測上衣遮罩
5. 遮罩擴展 + 羽化（避免接縫）
6. IDM-VTON 核心生成（40 步高品質）
7. FaceDetailer 臉部修復（SD1.5 RealisticVision + YOLOv8）

所需模型：
- yisol/IDM-VTON（核心 VTON 模型，fp16）
- sam_vit_h_4b8939.pth（SAM 分割）
- GroundingDINO_SwinT_OGC（物件偵測）
- sd15/realisticVisionV60B1_v51VAE.safetensors（臉部修復用 SD1.5）
- bbox/face_yolov8m.pt（人臉偵測）
- yolox_l.onnx + dw-ll_ucoco_384_bs5.torchscript.pt（DWPose）
- densepose_r50_fpn_dl.torchscript（DensePose）

效能：預估每張約 60~180 秒（視 GPU 而定）
"""


def build_idm_vton_workflow(person_img: str, garment_img: str) -> dict:
    """
    IDM-VTON 換裝工作流

    Args:
        person_img:  人物圖檔名（要換裝的人）
        garment_img: 服裝圖檔名（目標服裝）
    """
    return {
        # ── 載入圖片 ─────────────────────────────────────────────────
        "1": {
            "class_type": "LoadImage",
            "inputs": {
                "image": person_img,
                "upload": "image"
            },
            "_meta": {"title": "人物照片 (Person Image)"}
        },
        "2": {
            "class_type": "LoadImage",
            "inputs": {
                "image": garment_img,
                "upload": "image"
            },
            "_meta": {"title": "服裝照片 (Garment Image)"}
        },

        # ── 縮放至 768x1024 ──────────────────────────────────────────
        "11": {
            "class_type": "ImageResize+",
            "inputs": {
                "width": 768,
                "height": 1024,
                "interpolation": "lanczos",
                "method": "keep proportion",
                "condition": "always",
                "multiple_of": 8,
                "image": ["1", 0]
            },
            "_meta": {"title": "人物圖縮放至 768x1024"}
        },
        "12": {
            "class_type": "ImageResize+",
            "inputs": {
                "width": 768,
                "height": 1024,
                "interpolation": "lanczos",
                "method": "keep proportion",
                "condition": "always",
                "multiple_of": 8,
                "image": ["2", 0]
            },
            "_meta": {"title": "服裝圖縮放至 768x1024"}
        },

        # ── DWPose 骨架擷取（全身偵測）───────────────────────────────
        "3": {
            "class_type": "DWPreprocessor",
            "inputs": {
                "detect_hand": "enable",
                "detect_body": "enable",
                "detect_face": "enable",
                "resolution": 1024,
                "bbox_detector": "yolox_l.onnx",
                "pose_estimator": "dw-ll_ucoco_384_bs5.torchscript.pt",
                "image": ["11", 0]
            },
            "_meta": {"title": "骨架擷取 (DWPose) - 全身偵測"}
        },

        # ── DensePose 3D 人體拓樸 ────────────────────────────────────
        "4": {
            "class_type": "DensePosePreprocessor",
            "inputs": {
                "model": "densepose_r50_fpn_dl.torchscript",
                "cmap": "Viridis",
                "resolution": 1024,
                "image": ["11", 0]
            },
            "_meta": {"title": "3D 人體拓樸 (DensePose)"}
        },

        # ── IDM-VTON 核心模型載入 ────────────────────────────────────
        "5": {
            "class_type": "IDM_VTON_ModelLoader",
            "inputs": {
                "model_name": "yisol/IDM-VTON",
                "dtype": "fp16",
                "download_if_missing": True
            },
            "_meta": {"title": "IDM-VTON 核心模型載入 (fp16)"}
        },

        # ── SAM + GroundingDINO 自動遮罩 ─────────────────────────────
        "6": {
            "class_type": "SAMLoader",
            "inputs": {
                "model_name": "sam_vit_h_4b8939.pth",
                "device_mode": "AUTO"
            },
            "_meta": {"title": "SAM 模型載入"}
        },
        "7": {
            "class_type": "GroundingDinoModelLoader (segment anything)",
            "inputs": {
                "model_name": "GroundingDINO_SwinT_OGC (1.2GB)"
            },
            "_meta": {"title": "GroundingDINO 模型載入"}
        },
        "8": {
            "class_type": "GroundingDinoSAMSegment (segmentator)",
            "inputs": {
                "prompt": "upper body clothes . shirt . t-shirt . jacket . coat . top . blouse . sweater",
                "threshold": 0.25,
                "sam_model": ["6", 0],
                "grounding_dino_model": ["7", 0],
                "image": ["11", 0]
            },
            "_meta": {"title": "自動化遮罩 - 上衣偵測"}
        },

        # ── 遮罩擴展 + 羽化 ─────────────────────────────────────────
        "13": {
            "class_type": "GrowMaskWithBlur",
            "inputs": {
                "expand": 12,
                "incremental_expandrate": 0,
                "tapered_corners": True,
                "flip_input": False,
                "blur_radius": 8,
                "lerp_alpha": 0.6,
                "decay_factor": 1.0,
                "fill_holes": True,
                "mask": ["8", 1]
            },
            "_meta": {"title": "遮罩擴展 + 羽化 (避免接縫)"}
        },

        # ── IDM-VTON 核心生成 ────────────────────────────────────────
        "9": {
            "class_type": "IDM_VTON",
            "inputs": {
                "garment_desc": "a photo of a high quality garment, clean background, studio lighting",
                "negative_prompt": "monochrome, lowres, bad anatomy, worst quality, low quality, blurry, artifacts, deformed, disfigured",
                "num_inference_steps": 50,
                "guidance_scale": 3.0,
                "seed": 0,
                "pipe": ["5", 0],
                "image": ["11", 0],
                "garment_img": ["12", 0],
                "pose_img": ["3", 0],
                "densepose_img": ["4", 0],
                "mask": ["13", 0]
            },
            "_meta": {"title": "IDM-VTON 核心生成 (50步 高品質)"}
        },

        # ── FaceDetailer 臉部修復 ────────────────────────────────────
        "20": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {
                "ckpt_name": "sd15/realisticVisionV60B1_v51VAE.safetensors"
            },
            "_meta": {"title": "SD1.5 寫實模型 (臉部修復用)"}
        },
        "21": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": "sharp face, detailed eyes, natural skin texture, high quality portrait, same face as original, preserve facial features",
                "clip": ["20", 1]
            },
            "_meta": {"title": "臉部正面提示詞"}
        },
        "22": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": "blurry, deformed face, extra fingers, bad anatomy, disfigured, poorly drawn face, different face, changed face",
                "clip": ["20", 1]
            },
            "_meta": {"title": "臉部負面提示詞"}
        },
        "23": {
            "class_type": "UltralyticsDetectorProvider",
            "inputs": {
                "model_name": "bbox/face_yolov8m.pt"
            },
            "_meta": {"title": "人臉偵測器 (YOLOv8)"}
        },
        "14": {
            "class_type": "FaceDetailer",
            "inputs": {
                "guide_size": 384,
                "guide_size_for": True,
                "max_size": 1024,
                "seed": 0,
                "steps": 20,
                "cfg": 7.0,
                "sampler_name": "euler_ancestral",
                "scheduler": "normal",
                "denoise": 0.2,
                "feather": 5,
                "noise_mask": True,
                "force_inpaint": True,
                "bbox_threshold": 0.5,
                "bbox_dilation": 10,
                "bbox_crop_factor": 3.0,
                "sam_detection_hint": "center-1",
                "sam_dilation": 0,
                "sam_threshold": 0.93,
                "sam_bbox_expansion": 0,
                "sam_mask_hint_threshold": 0.7,
                "sam_mask_hint_use_negative": "False",
                "drop_size": 10,
                "wildcard": "",
                "cycle": 1,
                "inpaint_model": False,
                "noise_mask_feather": 20,
                "image": ["9", 0],
                "model": ["20", 0],
                "clip": ["20", 1],
                "vae": ["20", 2],
                "positive": ["21", 0],
                "negative": ["22", 0],
                "bbox_detector": ["23", 0],
                "sam_model_opt": ["6", 0]
            },
            "_meta": {"title": "臉部修復 (FaceDetailer)"}
        },

        # ── 儲存結果 ─────────────────────────────────────────────────
        "10": {
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": "VTON_IDM_Final",
                "images": ["14", 0]
            },
            "_meta": {"title": "儲存最終結果 (含臉部修復)"}
        },
    }
