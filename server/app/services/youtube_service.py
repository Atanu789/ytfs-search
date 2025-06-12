import asyncio
import logging
import re
import json
import os
import tempfile
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import requests
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
import assemblyai as aai
import yt_dlp
from typing import Dict, Any
logger = logging.getLogger(__name__)


@dataclass
class VideoFormat:
  
    itag: int
    mime_type: str
    quality: str
    fps: Optional[int]
    resolution: Optional[str]
    video_codec: Optional[str]
    audio_codec: Optional[str]
    filesize: Optional[int]
    url: str


@dataclass
class CaptionTrack:
   
    language_code: str
    language_name: str
    is_auto_generated: bool
    is_translatable: bool


@dataclass
class VideoInfo:
   
    video_id: str
    title: str
    description: str
    duration: Optional[int]
    view_count: Optional[int]
    author: str
    publish_date: Optional[str]
    thumbnail_url: str
    formats: List[VideoFormat]
    available_captions: List[CaptionTrack]
    tags: List[str]
    category: Optional[str]


@dataclass
class TranscriptionResult:
  
    success: bool
    text: Optional[str]
    segments: Optional[List[Dict]]
    source: str 
    language: Optional[str]
    confidence: Optional[float]
    duration: Optional[float]
    error: Optional[str] = None


class AssemblyAIService:
    

    def __init__(self, api_key: str):
       
        if not api_key:
            raise ValueError("AssemblyAI API key is required")

        aai.settings.api_key = api_key
        self.transcriber = aai.Transcriber()

    async def transcribe_audio_url(self, audio_url: str,
                                   config: Optional[aai.TranscriptionConfig] = None) -> TranscriptionResult:
      
        try:
            if not config:
                config = aai.TranscriptionConfig(
                    speech_model=aai.SpeechModel.best,
                    language_detection=True,
                    punctuate=True,
                    format_text=True
                )

            transcript = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.transcriber.transcribe(audio_url, config=config)
            )

            if transcript.status == aai.TranscriptStatus.error:
                return TranscriptionResult(
                    success=False,
                    text=None,
                    segments=None,
                    source="assemblyai",
                    language=None,
                    confidence=None,
                    duration=None,
                    error=f"AssemblyAI transcription failed: {transcript.error}"
                )

            segments = []
            if hasattr(transcript, 'words') and transcript.words:
                current_segment = {"text": "", "start": 0, "end": 0}
                segment_words = []

                for word in transcript.words:
                    if not segment_words:
                        current_segment["start"] = word.start / 1000.0  

                    segment_words.append(word.text)
                    current_segment["end"] = word.end / 1000.0

                    if len(segment_words) >= 8:
                        current_segment["text"] = " ".join(segment_words)
                        current_segment["duration"] = current_segment["end"] - current_segment["start"]
                        segments.append(current_segment.copy())

                        segment_words = []
                        current_segment = {"text": "", "start": 0, "end": 0}

                if segment_words:
                    current_segment["text"] = " ".join(segment_words)
                    current_segment["duration"] = current_segment["end"] - current_segment["start"]
                    segments.append(current_segment)

            return TranscriptionResult(
                success=True,
                text=transcript.text,
                segments=segments if segments else None,
                source="assemblyai",
                language=getattr(transcript, 'language_code', None),
                confidence=getattr(transcript, 'confidence', None),
                duration=getattr(transcript, 'audio_duration', None),
                error=None
            )

        except Exception as e:
            logger.error(f"AssemblyAI transcription error: {e}")
            return TranscriptionResult(
                success=False,
                text=None,
                segments=None,
                source="assemblyai",
                language=None,
                confidence=None,
                duration=None,
                error=str(e)
            )


class YouTubeAudioExtractor:
 

    def __init__(self):
        self.ydl_opts = {
            'format': 'bestaudio/best',
            'extractaudio': True,
            'audioformat': 'mp3',
            'outtmpl': '%(title)s.%(ext)s',
            'quiet': True,
            'no_warnings': True,
            # Add retry parameters
            'retries': 3,
            'fragment_retries': 3,
            'skip_unavailable_fragments': True,
            'extractor_args': {
                'youtube': {
                    'skip': ['dash', 'hls']
                }
            }
        }

    async def extract_audio_url(self, video_url: str) -> Optional[str]:
      
        try:
      
            def extract_info():
                with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                    try:
                        info = ydl.extract_info(video_url, download=False)
                        if not info:
                            return None

                        audio_formats = [f for f in info.get('formats', [])
                                     if f.get('acodec') != 'none' and f.get('vcodec') == 'none']

                        if not audio_formats:
                            audio_formats = [f for f in info.get('formats', [])
                                         if f.get('acodec') != 'none']

                        if audio_formats:
                            def get_bitrate(fmt):
                                abr = fmt.get('abr')
                                tbr = fmt.get('tbr')
                                return abr if isinstance(abr, (int, float)) else (
                                    tbr if isinstance(tbr, (int, float)) else 0)

                            best_audio = max(audio_formats, key=get_bitrate)
                            return best_audio.get('url')

                        return None
                    except yt_dlp.DownloadError as e:
                        logger.error(f"Download error: {e}")
                        return None
                return None

            audio_url = await asyncio.get_event_loop().run_in_executor(None, extract_info)
            return audio_url

        except Exception as e:
            logger.error(f"Failed to extract audio URL: {e}")
            return None

class YouTubeVideoService:
   
    def __init__(self, assemblyai_api_key: Optional[str] = None):
        self.session = requests.Session()
       
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })

        self.common_languages = [
            'en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh', 'zh-CN', 'zh-TW',
            'ar', 'hi', 'hi-IN', 'th', 'vi', 'id', 'ms', 'tl', 'sv', 'no', 'da', 'fi', 'nl', 'pl',
            'tr', 'cs', 'hu', 'ro', 'bg', 'hr', 'sk', 'sl', 'et', 'lv', 'lt', 'uk', 'he'
        ]

        self.assemblyai_service = None
        self.audio_extractor = None

        if assemblyai_api_key:
            try:
                self.assemblyai_service = AssemblyAIService(assemblyai_api_key)
                self.audio_extractor = YouTubeAudioExtractor()
                logger.info("AssemblyAI fallback enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize AssemblyAI service: {e}")

    def extract_video_id(self, url: str) -> Optional[str]:
       
        patterns = [
            r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)',
            r'youtube\.com\/v\/([^&\n?#]+)',
            r'youtube\.com\/watch\?.*v=([^&\n?#]+)',
            r'(?:youtube\.com\/shorts\/)([^&\n?#]+)'
        ]

        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)

        return None

    async def get_video_metadata(self, video_id: str) -> Dict:
        """Get comprehensive video metadata using multiple APIs"""
        metadata = {}

      
        try:
            oembed_url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
            response = self.session.get(oembed_url, timeout=15)
            response.raise_for_status()

            data = response.json()
            metadata.update({
                'title': data.get('title', f'Video {video_id}'),
                'author': data.get('author_name', 'Unknown Author'),
                'thumbnail_url': data.get('thumbnail_url', f'https://img.youtube.com/vi/{video_id}/maxresdefault.jpg'),
            })
        except Exception as e:
            logger.warning(f"oEmbed API failed: {e}")

        try:
   
            watch_url = f"https://www.youtube.com/watch?v={video_id}"
            response = self.session.get(watch_url, timeout=15)

            if response.status_code == 200:
                html_content = response.text

                
                title_match = re.search(r'"title":"([^"]+)"', html_content)
                if title_match and 'title' not in metadata:
                    metadata['title'] = title_match.group(1).encode().decode('unicode_escape')

                view_match = re.search(r'"viewCount":"(\d+)"', html_content)
                if view_match:
                    metadata['view_count'] = int(view_match.group(1))

               
                duration_match = re.search(r'"lengthSeconds":"(\d+)"', html_content)
                if duration_match:
                    metadata['duration'] = int(duration_match.group(1))

                desc_match = re.search(r'"shortDescription":"([^"]*)"', html_content)
                if desc_match:
                    metadata['description'] = desc_match.group(1).encode().decode('unicode_escape')

               
                date_match = re.search(r'"uploadDate":"([^"]+)"', html_content)
                if date_match:
                    metadata['publish_date'] = date_match.group(1)

                
                tags_match = re.search(r'"keywords":\[([^\]]+)\]', html_content)
                if tags_match:
                    try:
                        tags_str = '[' + tags_match.group(1) + ']'
                        metadata['tags'] = json.loads(tags_str)
                    except:
                        metadata['tags'] = []

        except Exception as e:
            logger.warning(f"Failed to extract additional metadata: {e}")

        metadata.setdefault('title', f'Video {video_id}')
        metadata.setdefault('author', 'Unknown Author')
        metadata.setdefault('description', '')
        metadata.setdefault('duration', None)
        metadata.setdefault('view_count', None)
        metadata.setdefault('publish_date', None)
        metadata.setdefault('thumbnail_url', f'https://img.youtube.com/vi/{video_id}/maxresdefault.jpg')
        metadata.setdefault('tags', [])
        metadata.setdefault('category', None)

        return metadata

    async def get_available_captions(self, video_id: str) -> List[CaptionTrack]:
        """Get list of all available caption tracks for a video"""
        caption_tracks = []

        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

            for transcript in transcript_list:
                caption_tracks.append(CaptionTrack(
                    language_code=transcript.language_code,
                    language_name=transcript.language,
                    is_auto_generated=transcript.is_generated,
                    is_translatable=transcript.is_translatable
                ))

        except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable) as e:
            logger.info(f"No captions available for video {video_id}: {e}")
        except Exception as e:
            logger.warning(f"Error getting caption list: {e}")

        return caption_tracks

    async def get_video_info(self, url: str) -> VideoInfo:
        """Get comprehensive video information"""
        try:
            video_id = self.extract_video_id(url)
            if not video_id:
                raise ValueError("Invalid YouTube URL")

            metadata_task = self.get_video_metadata(video_id)
            captions_task = self.get_available_captions(video_id)

            metadata, available_captions = await asyncio.gather(
                metadata_task, captions_task, return_exceptions=True
            )

            if isinstance(metadata, Exception):
                logger.error(f"Failed to get metadata: {metadata}")
                metadata = {'title': f'Video {video_id}', 'author': 'Unknown'}

            if isinstance(available_captions, Exception):
                logger.error(f"Failed to get captions list: {available_captions}")
                available_captions = []

            return VideoInfo(
                video_id=video_id,
                title=metadata['title'],
                description=metadata['description'],
                duration=metadata['duration'],
                view_count=metadata['view_count'],
                author=metadata['author'],
                publish_date=metadata['publish_date'],
                thumbnail_url=metadata['thumbnail_url'],
                formats=[],  
                available_captions=available_captions,
                tags=metadata['tags'],
                category=metadata['category']
            )

        except Exception as e:
            logger.error(f"Error getting video info: {e}")
            raise

    async def get_captions(
            self,
            url: str,
            languages: Optional[List[str]] = None,
            prefer_manual: bool = True,
            format_type: str = "json",
            translate_to: Optional[str] = None,
            use_assemblyai_fallback: bool = True
    ) -> Dict[str, any]:
       
        try:
            video_id = self.extract_video_id(url)
            if not video_id:
                return {"success": False, "error": "Invalid YouTube URL"}

         
            youtube_result = await self._get_youtube_captions(
                video_id, languages, prefer_manual, format_type, translate_to
            )

            if youtube_result["success"]:
                youtube_result["transcription_source"] = "youtube_captions"
                return youtube_result

            if use_assemblyai_fallback and self.assemblyai_service and self.audio_extractor:
                logger.info("YouTube captions failed, trying AssemblyAI fallback...")

                assemblyai_result = await self._get_assemblyai_transcription(
                    url, format_type
                )

                if assemblyai_result["success"]:
                    assemblyai_result["transcription_source"] = "assemblyai"
                    assemblyai_result["fallback_reason"] = youtube_result["error"]
                    return assemblyai_result
                else:
                   
                    return {
                        "success": False,
                        "error": f"YouTube captions failed: {youtube_result['error']}. AssemblyAI fallback also failed: {assemblyai_result['error']}",
                        "youtube_error": youtube_result["error"],
                        "assemblyai_error": assemblyai_result["error"]
                    }
            else:
            
                if not self.assemblyai_service:
                    youtube_result["note"] = "AssemblyAI fallback not available (no API key provided)"
                return youtube_result

        except Exception as e:
            logger.error(f"Error getting captions: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _get_youtube_captions(
            self,
            video_id: str,
            languages: Optional[List[str]],
            prefer_manual: bool,
            format_type: str,
            translate_to: Optional[str]
    ) -> Dict[str, any]:
        """Get captions from YouTube's native caption system"""
        try:
           
            try:
                transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            except (TranscriptsDisabled, VideoUnavailable):
                return {
                    "success": False,
                    "error": "Captions are disabled or video is unavailable for this video"
                }
            except NoTranscriptFound:
                return {
                    "success": False,
                    "error": "No captions found for this video"
                }

            if not languages:
                languages = ['en']  

            selected_transcript = None
            selected_language = None

            for lang in languages:
                try:
                    if prefer_manual:
                  
                        try:
                            selected_transcript = transcript_list.find_manually_created_transcript([lang])
                            selected_language = lang
                            break
                        except NoTranscriptFound:
                            try:
                                selected_transcript = transcript_list.find_generated_transcript([lang])
                                selected_language = lang
                                break
                            except NoTranscriptFound:
                                continue
                    else:
                       
                        selected_transcript = transcript_list.find_transcript([lang])
                        selected_language = lang
                        break
                except NoTranscriptFound:
                    continue

           
            if not selected_transcript:
                available_transcripts = list(transcript_list)
                if available_transcripts:
  
                    manual_transcripts = [t for t in available_transcripts if not t.is_generated]
                    if manual_transcripts and prefer_manual:
                        selected_transcript = manual_transcripts[0]
                    else:
                        selected_transcript = available_transcripts[0]
                    selected_language = selected_transcript.language_code

            if not selected_transcript:
                return {
                    "success": False,
                    "error": "No suitable captions found for the requested languages"
                }

            
            if translate_to and translate_to != selected_language:
                try:
                    selected_transcript = selected_transcript.translate(translate_to)
                    selected_language = translate_to
                except Exception as e:
                    logger.warning(f"Translation to {translate_to} failed: {e}")
                   
            try:
                transcript_data = selected_transcript.fetch()
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Failed to fetch caption data: {str(e)}"
                }

        
            formatted_captions = []
            for item in transcript_data:
                caption_dict = {
                    'text': item.get('text', '') if isinstance(item, dict) else getattr(item, 'text', ''),
                    'start': float(item.get('start', 0.0)) if isinstance(item, dict) else float(
                        getattr(item, 'start', 0.0)),
                    'duration': float(item.get('duration', 0.0)) if isinstance(item, dict) else float(
                        getattr(item, 'duration', 0.0))
                }
                formatted_captions.append(caption_dict)

            formatted_text = None
            if format_type != "json":
                formatted_text = self._format_captions(formatted_captions, format_type)

            total_duration = max([c['start'] + c['duration'] for c in formatted_captions]) if formatted_captions else 0

            return {
                "success": True,
                "video_id": video_id,
                "language": selected_language,
                "language_name": selected_transcript.language,
                "is_auto_generated": selected_transcript.is_generated,
                "format_type": format_type,
                "total_segments": len(formatted_captions),
                "duration": total_duration,
                "subtitles": formatted_captions if format_type == "json" else None,
                "text": formatted_text if format_type != "json" else None,
                "was_translated": translate_to is not None and translate_to != selected_transcript.language_code
            }

        except Exception as e:
            logger.error(f"Error getting YouTube captions: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _get_assemblyai_transcription(self, url: str, format_type: str = "json") -> Dict[str, Any]:
        """Get transcription using AssemblyAI as fallback with local download"""
        try:
            video_id = self.extract_video_id(url)
            if not video_id:
                return {"success": False, "error": "Invalid YouTube URL"}

            logger.info("Extracting audio URL...")

           
            audio_url = await self.audio_extractor.extract_audio_url(url)
            if not audio_url:
                return {
                    "success": False,
                    "error": "Failed to extract audio URL from video"
                }

            logger.info("Downloading audio file temporarily...")

            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_file:
                temp_path = tmp_file.name

                try:
                   
                    response = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: requests.get(audio_url, stream=True, timeout=30)
                    )
                    response.raise_for_status()

                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            tmp_file.write(chunk)

                    tmp_file.flush()

                    logger.info(f"Audio downloaded to {temp_path} ({os.path.getsize(temp_path) / 1024 / 1024:.2f} MB)")

                    
                    logger.info("Uploading to AssemblyAI...")
                    upload_url = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self.assemblyai_service.transcriber.upload_file(temp_path)
                    )

                    logger.info(f"AssemblyAI upload URL: {upload_url}")

                   
                    transcription_result = await self.assemblyai_service.transcribe_audio_url(
                        upload_url,
                        config=aai.TranscriptionConfig(
                            speech_model=aai.SpeechModel.best,
                            language_detection=True
                        )
                    )

              
                    try:
                        os.unlink(temp_path)
                    except Exception as e:
                        logger.warning(f"Failed to delete temp file: {e}")

             
                    if not transcription_result.success:
                        return {
                            "success": False,
                            "error": transcription_result.error
                        }

                    formatted_captions = []
                    if transcription_result.segments:
                        formatted_captions = [
                            {
                                'text': segment['text'],
                                'start': segment['start'],
                                'duration': segment.get('duration', segment['end'] - segment['start'])
                            }
                            for segment in transcription_result.segments
                        ]
                    else:
                        formatted_captions = [{
                            'text': transcription_result.text,
                            'start': 0.0,
                            'duration': transcription_result.duration or 0.0
                        }]

                    formatted_text = None
                    if format_type != "json":
                        formatted_text = self._format_captions(formatted_captions, format_type)

                    return {
                        "success": True,
                        "video_id": video_id,
                        "language": transcription_result.language or "auto-detected",
                        "language_name": transcription_result.language or "Auto-detected",
                        "is_auto_generated": True,
                        "format_type": format_type,
                        "total_segments": len(formatted_captions),
                        "duration": transcription_result.duration,
                        "confidence": transcription_result.confidence,
                        "subtitles": formatted_captions if format_type == "json" else None,
                        "text": formatted_text if format_type != "json" else transcription_result.text,
                        "was_translated": False,
                        "transcription_method": "assemblyai"
                    }

                except Exception as e:
                    try:
                        os.unlink(temp_path)
                    except:
                        pass
                    logger.exception("Error during audio download or transcription")
                    raise e

        except Exception as e:
            logger.error(f"AssemblyAI transcription error: {e}")
            return {
                "success": False,
                "error": f"AssemblyAI transcription failed: {str(e)}"
            }
    async def get_all_available_captions(self, url: str) -> Dict[str, any]:
        """Get information about all available caption tracks"""
        try:
            video_id = self.extract_video_id(url)
            if not video_id:
                return {"success": False, "error": "Invalid YouTube URL"}

            available_captions = await self.get_available_captions(video_id)

            result = {
                "success": True,
                "video_id": video_id,
                "available_languages": [
                    {
                        "code": track.language_code,
                        "name": track.language_name,
                        "auto_generated": track.is_auto_generated,
                        "translatable": track.is_translatable
                    }
                    for track in available_captions
                ],
                "total_tracks": len(available_captions)
            }

        
            if self.assemblyai_service:
                result["assemblyai_fallback_available"] = True
                result["note"] = "AssemblyAI fallback available for videos without captions"
            else:
                result["assemblyai_fallback_available"] = False
                result["note"] = "AssemblyAI fallback not configured"

            return result

        except Exception as e:
            logger.error(f"Error getting available captions: {e}")
            return {"success": False, "error": str(e)}

    def _format_captions(self, captions: List[Dict], format_type: str) -> str:
        """Format captions into requested format"""
        if format_type == "txt":
            return "\n".join([c.get("text", "") for c in captions])
        elif format_type == "srt":
            return self._format_as_srt(captions)
        elif format_type == "vtt":
            return self._format_as_vtt(captions)
        else:
            return ""

    def _format_as_srt(self, captions: List[Dict]) -> str:
        """Format captions as SRT"""
        srt_content = []
        for i, caption in enumerate(captions, 1):
            start = self._seconds_to_srt_time(caption.get("start", 0))
            end = self._seconds_to_srt_time(caption.get("start", 0) + caption.get("duration", 0))
            srt_content.append(f"{i}\n{start} --> {end}\n{caption.get('text', '')}\n")
        return "\n".join(srt_content)

    def _format_as_vtt(self, captions: List[Dict]) -> str:
        """Format captions as WebVTT"""
        vtt_content = ["WEBVTT\n"]
        for caption in captions:
            start = self._seconds_to_vtt_time(caption.get("start", 0))
            end = self._seconds_to_vtt_time(caption.get("start", 0) + caption.get("duration", 0))
            vtt_content.append(f"{start} --> {end}\n{caption.get('text', '')}\n")
        return "\n".join(vtt_content)

    def _seconds_to_srt_time(self, seconds: float) -> str:
        """Convert seconds to SRT time format (HH:MM:SS,mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        millis = int((secs - int(secs)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{int(secs):02d},{millis:03d}"

    def _seconds_to_vtt_time(self, seconds: float) -> str:
        """Convert seconds to VTT time format (HH:MM:SS.mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


async def main():
    """Example usage of the YouTube Video Service"""
   
    ASSEMBLYAI_API_KEY = "9b6fce9eb17b4087981066119b25f81b"

  
    service = YouTubeVideoService(assemblyai_api_key=ASSEMBLYAI_API_KEY)

 
    video_url = "https://www.youtube.com/watch?v=aW2oassRy64"

    try:
    
        print("Getting video information...")
        video_info = await service.get_video_info(video_url)
        print(f"Title: {video_info.title}")
        print(f"Author: {video_info.author}")
        print(f"Duration: {video_info.duration} seconds")
        print(f"Available captions: {len(video_info.available_captions)}")

      
        print("\nGetting available captions...")
        available_captions = await service.get_all_available_captions(video_url)
        if available_captions["success"]:
            for lang in available_captions["available_languages"]:
                print(f"- {lang['name']} ({lang['code']}) - Auto: {lang['auto_generated']}")

        print("\nGetting captions...")
        captions_result = await service.get_captions(
            video_url,
            languages=['en'],
            format_type='txt',
            use_assemblyai_fallback=True
        )

        if captions_result["success"]:
            print(f"Captions retrieved successfully!")
            print(f"Language: {captions_result['language']}")
            print(f"Source: {captions_result.get('transcription_source', 'youtube_captions')}")
            print(f"Total segments: {captions_result['total_segments']}")

           
            text = captions_result.get('text', '')
            if text:
                print(f"Text preview: {text[:200]}...")
        else:
            print(f"Failed to get captions: {captions_result['error']}")

    except Exception as e:
        print(f"Error: {e}")



async def get_video_transcript_only(url: str, assemblyai_api_key: str = None) -> str:

    service = YouTubeVideoService(assemblyai_api_key=assemblyai_api_key)

    result = await service.get_captions(
        url,
        languages=['en'],
        format_type='txt',
        use_assemblyai_fallback=bool(assemblyai_api_key)
    )

    if result["success"]:
        return result.get('text', '')
    else:
        raise Exception(f"Failed to get transcript: {result['error']}")


async def download_captions_as_file(url: str, output_file: str, format_type: str = 'srt',
                                    assemblyai_api_key: str = None):
    """
    Download captions and save to file
    """
    service = YouTubeVideoService(assemblyai_api_key=assemblyai_api_key)

    result = await service.get_captions(
        url,
        languages=['en'],
        format_type=format_type,
        use_assemblyai_fallback=bool(assemblyai_api_key)
    )

    if result["success"]:
        content = result.get('text', '')
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Captions saved to {output_file}")
        return True
    else:
        print(f"Failed to download captions: {result['error']}")
        return False


if __name__ == "__main__":
 
    asyncio.run(main())
