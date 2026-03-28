"""
FaceOff_V1 — ReActor Face Swap（直接換臉，保留原圖一切）

流程：
1. 載入人物圖（保留身體/背景/服裝）+ 臉孔來源圖（要換上的臉）
2. ReActorFaceSwap 執行換臉（inswapper_128）
3. ReActorRestoreFace 修復臉部品質（GFPGANv1.4）

所需模型：
- insightface/inswapper_128.onnx (ReActor 換臉)
- facerestore_models/GFPGANv1.4.pth (臉部修復)

特點：
- 保留原圖的身體、姿勢、服裝、髮型、背景
- 只替換臉部特徵
- 速度快（約 5~15 秒）
"""


def build_faceoff_workflow(person_img: str, face_img: str) -> dict:
    """
    ReActor Face Swap — 直接換臉

    Args:
        person_img: 人物圖檔名（要被換臉的人，保留身體/背景）
        face_img:   臉孔來源圖檔名（要換上去的臉）
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
            "inputs": {"image": face_img}
        },

        # ═══════════════════════════════════════════
        # ReActor 換臉
        # ═══════════════════════════════════════════
        "10": {
            "class_type": "ReActorFaceSwap",
            "inputs": {
                "enabled": True,
                "input_image": ["1", 0],      # 人物圖（保留身體/背景）
                "source_image": ["2", 0],     # 臉孔來源（要換上的臉）
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
        # 臉部修復（額外一輪，提升品質）
        # ═══════════════════════════════════════════
        "20": {
            "class_type": "ReActorRestoreFace",
            "inputs": {
                "image": ["10", 0],           # SWAPPED_IMAGE
                "facedetection": "retinaface_resnet50",
                "model": "GFPGANv1.4.pth",
                "visibility": 1.0,
                "codeformer_weight": 0.5
            }
        },

        # ═══════════════════════════════════════════
        # 儲存結果
        # ═══════════════════════════════════════════
        "70": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["20", 0],
                "filename_prefix": "faceoff_v1"
            }
        },
    }
