#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：EchoMimic
@File    ：audio2vid.py
@Author  roy
@Date    ：2024/3/4 17:43 
'''
import argparse
import os

import random
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote

import cv2
import numpy as np
import torch
from diffusers import AutoencoderKL, DDIMScheduler
from omegaconf import OmegaConf
from PIL import Image

from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d_echo import EchoUNet3DConditionModel
from src.models.whisper.audio2feature import load_audio_model
from src.pipelines.pipeline_echo_mimic import Audio2VideoPipeline
from src.utils.util import save_videos_grid, crop_and_pad
from src.models.face_locator import FaceLocator
from moviepy.editor import VideoFileClip, AudioFileClip
from facenet_pytorch import MTCNN

from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.responses import FileResponse


from starlette.middleware.cors import CORSMiddleware  #引入 CORS中间件模块

#设置允许访问的域名
origins = ["*"]  #"*"，即为所有。

from pydantic import BaseModel

import uvicorn


from typing import Generator

app = FastAPI()

app.add_middleware(
    CORSMiddleware, 
    allow_origins=origins,  #设置允许的origins来源
    allow_credentials=True,
    allow_methods=["*"],  # 设置允许跨域的http方法，比如 get、post、put等。
    allow_headers=["*"])  #允许跨域的headers，可以用来鉴别来源等作用。

class Audio2VidRequest(BaseModel):
    ref_image_path: str
    audio_path: str
    output_path: str
    seed: int = 2581  # 提供默认值
    steps: int = 10

ffmpeg_path = os.getenv('FFMPEG_PATH')
if ffmpeg_path is None and platform.system() in ['Linux', 'Darwin']:
    try:
        result = subprocess.run(['which', 'ffmpeg'], capture_output=True, text=True)
        if result.returncode == 0:
            ffmpeg_path = result.stdout.strip()
            print(f"FFmpeg is installed at: {ffmpeg_path}")
        else:
            print("FFmpeg is not installed. Please download ffmpeg-static and export to FFMPEG_PATH.")
            print("For example: export FFMPEG_PATH=/musetalk/ffmpeg-4.4-amd64-static")
    except Exception as e:
        pass

if ffmpeg_path is not None and ffmpeg_path not in os.getenv('PATH'):
    print("Adding FFMPEG_PATH to PATH")
    os.environ["PATH"] = f"{ffmpeg_path}:{os.environ['PATH']}"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/prompts/animation.yaml")
    parser.add_argument("-W", type=int, default=512)
    parser.add_argument("-H", type=int, default=512)
    parser.add_argument("-L", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=420)
    parser.add_argument("--facemusk_dilation_ratio", type=float, default=0.1)
    parser.add_argument("--facecrop_dilation_ratio", type=float, default=0.5)

    parser.add_argument("--context_frames", type=int, default=12)
    parser.add_argument("--context_overlap", type=int, default=3)

    parser.add_argument("--cfg", type=float, default=2.5)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    return args

def select_face(det_bboxes, probs):
    ## max face from faces that the prob is above 0.8
    ## box: xyxy
    if det_bboxes is None or probs is None:
        return None
    filtered_bboxes = []
    for bbox_i in range(len(det_bboxes)):
        if probs[bbox_i] > 0.8:
            filtered_bboxes.append(det_bboxes[bbox_i])
    if len(filtered_bboxes) == 0:
        return None

    sorted_bboxes = sorted(filtered_bboxes, key=lambda x:(x[3]-x[1]) * (x[2] - x[0]), reverse=True)
    return sorted_bboxes[0]


@app.post("/audio2vid/")
async def convert_audio_to_video(request_data: Audio2VidRequest):
    try:
        # 直接使用请求数据中的路径调用 audio2vid 函数
        videoUrl = audio2vid(
            ref_image_path=request_data.ref_image_path,
            audio_path=request_data.audio_path,
            output_path=request_data.output_path,
            seed=request_data.seed,
            steps=request_data.steps
        )
        
        return {"message": "Conversion successful", "video_url": videoUrl}
    except Exception as e:
        # 如果 audio2vid 函数抛出异常，返回 HTTP 500 内部服务器错误
        raise HTTPException(status_code=500, detail=str(e))

def audio2vid(ref_image_path: str, audio_path: str, output_path: str, seed: int=2581, steps: int=10):
    
    print("audio2vid")
    args = parse_args()

    config = OmegaConf.load(args.config)
    if config.weight_dtype == "fp16":
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32

    device = args.device
    if device.__contains__("cuda") and not torch.cuda.is_available():
        device = "cpu"

    inference_config_path = config.inference_config
    infer_config = OmegaConf.load(inference_config_path)


    ############# model_init started #############

    ## vae init
    vae = AutoencoderKL.from_pretrained(
        config.pretrained_vae_path,
    ).to("cuda", dtype=weight_dtype)

    ## reference net init
    reference_unet = UNet2DConditionModel.from_pretrained(
        config.pretrained_base_model_path,
        subfolder="unet",
    ).to(dtype=weight_dtype, device=device)
    reference_unet.load_state_dict(
        torch.load(config.reference_unet_path, map_location="cpu"),
    )

    ## denoising net init
    if os.path.exists(config.motion_module_path):
        ### stage1 + stage2
        denoising_unet = EchoUNet3DConditionModel.from_pretrained_2d(
            config.pretrained_base_model_path,
            config.motion_module_path,
            subfolder="unet",
            unet_additional_kwargs=infer_config.unet_additional_kwargs,
        ).to(dtype=weight_dtype, device=device)
    else:
        ### only stage1
        denoising_unet = EchoUNet3DConditionModel.from_pretrained_2d(
            config.pretrained_base_model_path,
            "",
            subfolder="unet",
            unet_additional_kwargs={
                "use_motion_module": False,
                "unet_use_temporal_attention": False,
                "cross_attention_dim": infer_config.unet_additional_kwargs.cross_attention_dim
            }
        ).to(dtype=weight_dtype, device=device)
    denoising_unet.load_state_dict(
        torch.load(config.denoising_unet_path, map_location="cpu"),
        strict=False
    )

    ## face locator init
    face_locator = FaceLocator(320, conditioning_channels=1, block_out_channels=(16, 32, 96, 256)).to(
        dtype=weight_dtype, device="cuda"
    )
    face_locator.load_state_dict(torch.load(config.face_locator_path))

    ### load audio processor params
    audio_processor = load_audio_model(model_path=config.audio_model_path, device=device)

    ### load face detector params
    face_detector = MTCNN(image_size=320, margin=0, min_face_size=20, thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True, device=device)

    ############# model_init finished #############

    width, height = args.W, args.H
    sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
    scheduler = DDIMScheduler(**sched_kwargs)

    pipe = Audio2VideoPipeline(
        vae=vae,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        audio_guider=audio_processor,
        face_locator=face_locator,
        scheduler=scheduler,
    )
    pipe = pipe.to("cuda", dtype=weight_dtype)

    date_str = datetime.now().strftime("%Y%m%d")
    time_str = datetime.now().strftime("%H%M")
    save_dir_name = f"{output_path}" # f"{time_str}--seed_{args.seed}-{args.W}x{args.H}"
    save_dir = Path(f"output/{date_str}/{save_dir_name}")
    save_dir.mkdir(exist_ok=True, parents=True)

    # for ref_image_path in config["test_cases"].keys():
    # ref_image_path = "./assets/test_imgs/a.png"
    audio_paths = ["./assets/test_audios/alay.wav", "./assets/test_audios/chunnuanhuakai.wav"]
    # audio_path = "./assets/test_audios/alay.wav"
    # for audio_path in audio_paths:

    if seed is not None and seed > -1:
        generator = torch.manual_seed(seed)
    else:
        generator = torch.manual_seed(random.randint(100, 1000000))

    ref_name = Path(ref_image_path).stem
    audio_name = Path(audio_path).stem
    final_fps = args.fps

    #### face musk prepare
    face_img = cv2.imread(ref_image_path)
    face_mask = np.zeros((face_img.shape[0], face_img.shape[1])).astype('uint8')

    det_bboxes, probs = face_detector.detect(face_img)
    select_bbox = select_face(det_bboxes, probs)
    if select_bbox is None:
        face_mask[:, :] = 255
    else:
        xyxy = select_bbox[:4]
        xyxy = np.round(xyxy).astype('int')
        rb, re, cb, ce = xyxy[1], xyxy[3], xyxy[0], xyxy[2]
        r_pad = int((re - rb) * args.facemusk_dilation_ratio)
        c_pad = int((ce - cb) * args.facemusk_dilation_ratio)
        face_mask[rb - r_pad : re + r_pad, cb - c_pad : ce + c_pad] = 255

        #### face crop
        r_pad_crop = int((re - rb) * args.facecrop_dilation_ratio)
        c_pad_crop = int((ce - cb) * args.facecrop_dilation_ratio)
        crop_rect = [max(0, cb - c_pad_crop), max(0, rb - r_pad_crop), min(ce + c_pad_crop, face_img.shape[1]), min(re + c_pad_crop, face_img.shape[0])]
        # print(crop_rect)
        face_img = crop_and_pad(face_img, crop_rect)
        face_mask = crop_and_pad(face_mask, crop_rect)
        face_img = cv2.resize(face_img, (args.W, args.H))
        face_mask = cv2.resize(face_mask, (args.W, args.H))

    ref_image_pil = Image.fromarray(face_img[:, :, [2, 1, 0]])
    face_mask_tensor = torch.Tensor(face_mask).to(dtype=weight_dtype, device="cuda").unsqueeze(0).unsqueeze(0).unsqueeze(0) / 255.0

    video = pipe(
        ref_image_pil,
        audio_path,
        face_mask_tensor,
        width,
        height,
        args.L,
        steps,
        args.cfg,
        generator=generator,
        audio_sample_rate=args.sample_rate,
        context_frames=args.context_frames,
        fps=final_fps,
        context_overlap=args.context_overlap
    ).videos

    # 定义文件名变量
    file_name_base = f"{audio_name}" # {ref_name}_{audio_name}_{args.H}x{args.W}_{int(args.cfg)}_{time_str}

    # 使用 file_name_base 构建视频文件名
    video_filename = f"{save_dir}/{file_name_base}.mp4"

    # 使用 file_name_base 构建带音频的视频文件名
    video_with_audio_filename = f"{save_dir}/{file_name_base}_withaudio.mp4"
    video = video
    save_videos_grid(
        video,
        video_filename,
        n_rows=1,
        fps=final_fps,
    )
    video_clip = VideoFileClip(video_filename)

    # video_clip = VideoFileClip(f"{save_dir}/{ref_name}_{audio_name}_{args.H}x{args.W}_{int(args.cfg)}_{time_str}.mp4")
    audio_clip = AudioFileClip(audio_path)
    video_clip = video_clip.set_audio(audio_clip)
    # 写入包含音频的视频文件
    video_clip.write_videofile(
        video_with_audio_filename,
        codec="libx264",
        audio_codec="aac"
    )
    # video_clip.write_videofile(f"{save_dir}/{ref_name}_{audio_name}_{args.H}x{args.W}_{int(args.cfg)}_{time_str}_withaudio.mp4", codec="libx264", audio_codec="aac")
    # print(f"{save_dir}/{ref_name}_{audio_name}_{args.H}x{args.W}_{int(args.cfg)}_{time_str}_withaudio.mp4")
    print(f"Video saved as: {video_with_audio_filename}")
    return video_with_audio_filename

# 假设音频文件存储在服务器的这个目录下
# 定义基础变量，指向存储视频的根目录
BASE_VIDEO_DIR = "output"

@app.get("/getvideo")
async def get_video(path: str):
    # 构建完整的文件路径
    # 从请求的查询参数中获取文件路径

    # 如果没有提供文件路径，返回错误信息
    # if not filepath:
    #     return jsonify({'error': 'No filepath provided'}), 400
    print(f"filename: {path}")
    # 解码 URL 路径
    decoded_path = unquote(path)
    file_path = os.path.join(BASE_VIDEO_DIR, decoded_path)
    print(f"file_path: {file_path}")
    # 检查文件是否存在
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail=f"File {decoded_path} not found")

    # 返回视频文件
    return FileResponse(file_path, media_type="video/mp4")

# if __name__ == "__main__":
#     main()
if __name__ == "__main__":

    
    # chat = load_chat_tts_model(source="local", local_path="models")

    uvicorn.run(app,host='0.0.0.0',port=9881,workers=1)

# ```
# pip install torch==2.2.2+cu121 torchvision==0.17.2+cu121 torchaudio==2.2.2+cu121 -f https://download.pytorch.org/whl/torch_stable.html
# ```