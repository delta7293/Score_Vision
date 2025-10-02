import os
import json
import time
from typing import Optional, Dict, Any, List
import supervision as sv
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
import asyncio
from pathlib import Path
from loguru import logger

from fiber.logging_utils import get_logger
from miner.core.models.config import Config
from miner.core.configuration import factory_config
from miner.dependencies import get_config, verify_request, blacklist_low_stake
from sports.configs.soccer import SoccerPitchConfiguration
from miner.utils.device import get_optimal_device
from miner.utils.model_manager import ModelManager
from miner.utils.video_processor import VideoProcessor
from miner.utils.shared import miner_lock
from miner.utils.video_downloader import download_video


logger = get_logger(__name__)

CONFIG = SoccerPitchConfiguration()

# Global model manager instance
model_manager = None

def get_model_manager(config: Config = Depends(get_config)) -> ModelManager:
    global model_manager
    if model_manager is None:
        model_manager = ModelManager(device=config.device)
        model_manager.load_all_models()
    return model_manager

async def process_soccer_video_dual_gpu(
    video_path: str,
    model_manager: ModelManager,
    batch_size: int = 8
) -> Dict[str, Any]:
    """Process a soccer video using 2 GPUs (optimized)."""
    start_time = time.time()

    try:
        video_processor = VideoProcessor(
            device="cuda",  # 멀티 GPU 지원
            cuda_timeout=10800.0,
            mps_timeout=10800.0,
            cpu_timeout=10800.0
        )

        if not await video_processor.ensure_video_readable(video_path):
            raise HTTPException(
                status_code=400,
                detail="Video file is not readable or corrupted"
            )

        # GPU0 → pitch 모델
        pitch_model = model_manager.get_model("pitch")
        pitch_model.to("cuda:0")  # 모델을 GPU0으로 이동

        # Player 모델을 GPU0, GPU1에 각각 로드
        player_model0 = model_manager.get_model("player")
        player_model0.to("cuda:0")  # 모델을 GPU0으로 이동

        player_model1 = model_manager.get_model("player")
        player_model1.to("cuda:1")  # 모델을 GPU1으로 이동

        tracker = sv.ByteTrack()

        tracking_data = {"frames": []}
        batch_frames: List[Any] = []
        batch_indices: List[int] = []

        async for frame_number, frame in video_processor.stream_frames(video_path):
            batch_frames.append(frame)
            batch_indices.append(frame_number)

            if len(batch_frames) >= batch_size:
                # ----- GPU0: pitch 모델 -----
                pitch_task = asyncio.to_thread(
                    pitch_model, batch_frames, verbose=False
                )

                # ----- GPU0 & GPU1: player 모델 배치 분산 -----
                half = len(batch_frames) // 2
                frames0, frames1 = batch_frames[:half], batch_frames[half:]

                player_task0 = asyncio.to_thread(
                    player_model0, frames0, imgsz=960, verbose=False
                )
                player_task1 = asyncio.to_thread(
                    player_model1, frames1, imgsz=960, verbose=False
                )

                # 동시에 실행
                pitch_results, results0, results1 = await asyncio.gather(
                    pitch_task, player_task0, player_task1
                )

                player_results = results0 + results1

                # ----- 결과 처리 -----
                for idx, (pitch_res, player_res) in enumerate(zip(pitch_results, player_results)):
                    frame_num = batch_indices[idx]

                    keypoints = sv.KeyPoints.from_ultralytics(pitch_res[0])
                    detections = sv.Detections.from_ultralytics(player_res[0])
                    detections = tracker.update_with_detections(detections)

                    frame_data = {
                        "frame_number": int(frame_num),
                        "keypoints": keypoints.xy[0].tolist() if keypoints and keypoints.xy is not None else [],
                        "objects": [
                            {
                                "id": int(tid),
                                "bbox": [float(x) for x in bbox],
                                "class_id": int(cid)
                            }
                            for tid, bbox, cid in zip(
                                detections.tracker_id or [],
                                detections.xyxy or [],
                                detections.class_id or []
                            )
                        ]
                    }
                    tracking_data["frames"].append(frame_data)

                    if frame_num % 100 == 0:
                        elapsed = time.time() - start_time
                        fps = frame_num / elapsed if elapsed > 0 else 0
                        logger.info(f"Processed {frame_num} frames in {elapsed:.1f}s ({fps:.2f} fps)")

                batch_frames.clear()
                batch_indices.clear()

        # 남은 프레임 처리
        if batch_frames:
            pitch_task = asyncio.to_thread(pitch_model, batch_frames, verbose=False)
            half = len(batch_frames) // 2
            frames0, frames1 = batch_frames[:half], batch_frames[half:]

            player_task0 = asyncio.to_thread(player_model0, frames0, imgsz=960, verbose=False)
            player_task1 = asyncio.to_thread(player_model1, frames1, imgsz=960, verbose=False)

            pitch_results, results0, results1 = await asyncio.gather(
                pitch_task, player_task0, player_task1
            )
            player_results = results0 + results1

            for idx, (pitch_res, player_res) in enumerate(zip(pitch_results, player_results)):
                frame_num = batch_indices[idx]

                keypoints = sv.KeyPoints.from_ultralytics(pitch_res[0])
                detections = sv.Detections.from_ultralytics(player_res[0])
                detections = tracker.update_with_detections(detections)

                frame_data = {
                    "frame_number": int(frame_num),
                    "keypoints": keypoints.xy[0].tolist() if keypoints and keypoints.xy is not None else [],
                    "objects": [
                        {
                            "id": int(tid),
                            "bbox": [float(x) for x in bbox],
                            "class_id": int(cid)
                        }
                        for tid, bbox, cid in zip(
                            detections.tracker_id or [],
                            detections.xyxy or [],
                            detections.class_id or []
                        )
                    ]
                }
                tracking_data["frames"].append(frame_data)

        # 처리 시간 로그
        processing_time = time.time() - start_time
        tracking_data["processing_time"] = processing_time
        total_frames = len(tracking_data["frames"])
        fps = total_frames / processing_time if processing_time > 0 else 0

        logger.info(
            f"Completed processing {total_frames} frames in {processing_time:.1f}s "
            f"({fps:.2f} fps) using dual GPUs"
        )

        return tracking_data

    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Video processing error: {str(e)}")

async def process_challenge(
    request: Request,
    config: Config = Depends(get_config),
    model_manager: ModelManager = Depends(get_model_manager),
):
    logger.info("Attempting to acquire miner lock...")
    async with miner_lock:
        logger.info("Miner lock acquired, processing challenge...")
        try:
            challenge_data = await request.json()
            challenge_id = challenge_data.get("challenge_id")
            video_url = challenge_data.get("video_url")
            
            logger.info(f"Received challenge data: {json.dumps(challenge_data, indent=2)}")
            
            if not video_url:
                raise HTTPException(status_code=400, detail="No video URL provided")
            
            logger.info(f"Processing challenge {challenge_id} with video {video_url}")
            
            video_path = await download_video(video_url)
            
            try:
                tracking_data = await process_soccer_video(
                    video_path,
                    model_manager
                )
                
                response = {
                    "challenge_id": challenge_id,
                    "frames": tracking_data["frames"],
                    "processing_time": tracking_data["processing_time"]
                }
                
                logger.info(f"Completed challenge {challenge_id} in {tracking_data['processing_time']:.2f} seconds")
                return response
                
            finally:
                try:
                    os.unlink(video_path)
                except:
                    pass
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error processing soccer challenge: {str(e)}")
            logger.exception("Full error traceback:")
            raise HTTPException(status_code=500, detail=f"Challenge processing error: {str(e)}")
        finally:
            logger.info("Releasing miner lock...")

# Create router with dependencies
router = APIRouter()
router.add_api_route(
    "/challenge",
    process_challenge,
    tags=["soccer"],
    dependencies=[Depends(blacklist_low_stake), Depends(verify_request)],
    methods=["POST"],
)
