# app/api/v1/endpoints/youtube.py
from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List

from app.services.youtube_service import download_video_info, YouTubeVideoService
from app.schemas.youtube import VideoRequest, VideoInfoResponse, VideoFormatsResponse, SubtitlesResponse

router = APIRouter()


@router.post("/info", response_model=VideoInfoResponse)
async def get_video_info(request: VideoRequest):
    """
    Get comprehensive video information from YouTube URL

    - **url**: YouTube video URL (supports various formats)
    - **quality**: Preferred quality (optional: 'highest', 'lowest', '720p', '1080p', etc.)
    """
    try:
        # Validate URL format
        if not any(domain in request.url.lower() for domain in ['youtube.com', 'youtu.be']):
            raise HTTPException(status_code=400, detail="Invalid YouTube URL")

        result = await download_video_info(request.url)

        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])

        return VideoInfoResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/info/{video_id}", response_model=VideoInfoResponse)
async def get_video_info_by_id(video_id: str, quality: Optional[str] = "highest"):
    """
    Get video information by YouTube video ID

    - **video_id**: YouTube video ID (11 characters)
    - **quality**: Preferred quality (optional)
    """
    try:
        youtube_url = f"https://www.youtube.com/watch?v={video_id}"
        result = await download_video_info(youtube_url)

        if not result["success"]:
            raise HTTPException(status_code=404, detail="Video not found or unavailable")

        return VideoInfoResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/formats", response_model=VideoFormatsResponse)
async def get_video_formats(request: VideoRequest):
    """
    Get all available formats for a YouTube video

    Returns detailed information about all available video and audio formats
    """
    try:
        async with YouTubeVideoService() as service:
            video_info = await service.get_video_info(request.url)

            # Organize formats by type
            video_formats = []
            audio_formats = []

            for fmt in video_info.formats:
                format_data = {
                    "itag": fmt.itag,
                    "quality": fmt.quality,
                    "resolution": fmt.resolution,
                    "ext": fmt.ext,
                    "url": fmt.url,
                    "filesize": fmt.filesize,
                    "fps": fmt.fps
                }

                if fmt.resolution:  # Has video
                    video_formats.append(format_data)
                else:  # Audio only
                    audio_formats.append(format_data)

            return VideoFormatsResponse(
                success=True,
                video_id=video_info.video_id,
                title=video_info.title,
                total_formats=len(video_info.formats),
                video_formats=video_formats,
                audio_formats=audio_formats
            )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/subtitles", response_model=SubtitlesResponse)
async def get_video_subtitles(
        request: VideoRequest,
        language: Optional[str] = Query("en", description="Language code (e.g., 'en', 'es', 'fr')"),
        auto_generated: Optional[bool] = Query(True, description="Include auto-generated subtitles"),
        format_type: Optional[str] = Query("json", description="Format: 'json', 'srt', 'vtt'")
):
    """
    Get subtitles for a YouTube video with timestamps

    - **url**: YouTube video URL
    - **language**: Language code (default: 'en')
    - **auto_generated**: Include auto-generated subtitles (default: True)
    - **format_type**: Output format - 'json', 'srt', or 'vtt' (default: 'json')
    """
    try:
        # Validate URL format
        if not any(domain in request.url.lower() for domain in ['youtube.com', 'youtu.be']):
            raise HTTPException(status_code=400, detail="Invalid YouTube URL")

        async with YouTubeVideoService() as service:
            subtitles_data = await service.get_video_subtitles(
                request.url,
                language=language,
                auto_generated=auto_generated,
                format_type=format_type
            )

            if not subtitles_data["success"]:
                raise HTTPException(status_code=404, detail=subtitles_data["error"])

            return SubtitlesResponse(**subtitles_data)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/subtitles/{video_id}")
async def get_subtitles_by_id(
        video_id: str,
        language: Optional[str] = Query("en", description="Language code"),
        auto_generated: Optional[bool] = Query(True, description="Include auto-generated subtitles"),
        format_type: Optional[str] = Query("json", description="Format: 'json', 'srt', 'vtt'")
):
    """
    Get subtitles by YouTube video ID

    - **video_id**: YouTube video ID (11 characters)
    - **language**: Language code (default: 'en')
    - **auto_generated**: Include auto-generated subtitles (default: True)
    - **format_type**: Output format (default: 'json')
    """
    try:
        youtube_url = f"https://www.youtube.com/watch?v={video_id}"

        async with YouTubeVideoService() as service:
            subtitles_data = await service.get_video_subtitles(
                youtube_url,
                language=language,
                auto_generated=auto_generated,
                format_type=format_type
            )

            if not subtitles_data["success"]:
                raise HTTPException(status_code=404, detail=subtitles_data["error"])

            return SubtitlesResponse(**subtitles_data)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/subtitles/languages/{video_id}")
async def get_available_subtitle_languages(video_id: str):
    """
    Get all available subtitle languages for a video

    - **video_id**: YouTube video ID (11 characters)
    """
    try:
        youtube_url = f"https://www.youtube.com/watch?v={video_id}"

        async with YouTubeVideoService() as service:
            languages = await service.get_available_subtitle_languages(youtube_url)

            if not languages["success"]:
                raise HTTPException(status_code=404, detail=languages["error"])

            return {
                "success": True,
                "video_id": video_id,
                "available_languages": languages["languages"],
                "total_languages": len(languages["languages"])
            }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")