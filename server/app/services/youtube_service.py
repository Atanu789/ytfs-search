import asyncio
import logging
import re
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import requests
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound, VideoUnavailable

logger = logging.getLogger(__name__)


@dataclass
class VideoFormat:
    """Represents a video format with quality and URL information"""
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
    """Represents a caption track with language information"""
    language_code: str
    language_name: str
    is_auto_generated: bool
    is_translatable: bool


@dataclass
class VideoInfo:
    """Contains comprehensive video information"""
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


class YouTubeVideoService:
    """Enhanced service for YouTube video information and subtitle extraction"""

    def __init__(self):
        self.session = requests.Session()
        # Updated user agent for better compatibility
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })

        # Common language codes for better multilingual support
        self.common_languages = [
            'en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh', 'zh-CN', 'zh-TW',
            'ar', 'hi','hi-IN', 'th', 'vi', 'id', 'ms', 'tl', 'sv', 'no', 'da', 'fi', 'nl', 'pl',
            'tr', 'cs', 'hu', 'ro', 'bg', 'hr', 'sk', 'sl', 'et', 'lv', 'lt', 'uk', 'he'
        ]

    def extract_video_id(self, url: str) -> Optional[str]:
        """Extract video ID from various YouTube URL formats"""
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

        # Try oEmbed API first (most reliable)
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

        # Try YouTube's internal API for additional details
        try:
            # This endpoint sometimes provides more detailed information
            watch_url = f"https://www.youtube.com/watch?v={video_id}"
            response = self.session.get(watch_url, timeout=15)

            if response.status_code == 200:
                html_content = response.text

                # Extract additional metadata from HTML using regex
                title_match = re.search(r'"title":"([^"]+)"', html_content)
                if title_match and 'title' not in metadata:
                    metadata['title'] = title_match.group(1).encode().decode('unicode_escape')

                # Extract view count
                view_match = re.search(r'"viewCount":"(\d+)"', html_content)
                if view_match:
                    metadata['view_count'] = int(view_match.group(1))

                # Extract duration
                duration_match = re.search(r'"lengthSeconds":"(\d+)"', html_content)
                if duration_match:
                    metadata['duration'] = int(duration_match.group(1))

                # Extract description
                desc_match = re.search(r'"shortDescription":"([^"]*)"', html_content)
                if desc_match:
                    metadata['description'] = desc_match.group(1).encode().decode('unicode_escape')

                # Extract upload date
                date_match = re.search(r'"uploadDate":"([^"]+)"', html_content)
                if date_match:
                    metadata['publish_date'] = date_match.group(1)

                # Extract tags
                tags_match = re.search(r'"keywords":\[([^\]]+)\]', html_content)
                if tags_match:
                    try:
                        tags_str = '[' + tags_match.group(1) + ']'
                        metadata['tags'] = json.loads(tags_str)
                    except:
                        metadata['tags'] = []

        except Exception as e:
            logger.warning(f"Failed to extract additional metadata: {e}")

        # Fill in defaults for missing fields
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

            # Get metadata and available captions concurrently
            metadata_task = self.get_video_metadata(video_id)
            captions_task = self.get_available_captions(video_id)

            metadata, available_captions = await asyncio.gather(
                metadata_task, captions_task, return_exceptions=True
            )

            # Handle exceptions from concurrent tasks
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
                formats=[],  # We're removing pytube, so no download formats
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
            translate_to: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Get video captions with enhanced multilingual support

        Args:
            url: YouTube video URL
            languages: List of preferred language codes (e.g., ['en', 'es', 'fr'])
            prefer_manual: Whether to prefer manually created over auto-generated
            format_type: Output format ('json', 'txt', 'srt', 'vtt')
            translate_to: Language code to translate captions to
        """
        try:
            video_id = self.extract_video_id(url)
            if not video_id:
                return {"success": False, "error": "Invalid YouTube URL"}

            # Get available transcripts
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

            # Determine target languages
            if not languages:
                languages = ['en']  # Default to English

            # Find the best available transcript
            selected_transcript = None
            selected_language = None

            # Strategy 1: Try to find exact language matches
            for lang in languages:
                try:
                    if prefer_manual:
                        # Try manual first, then auto-generated
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
                        # Try any transcript in this language
                        selected_transcript = transcript_list.find_transcript([lang])
                        selected_language = lang
                        break
                except NoTranscriptFound:
                    continue

            # Strategy 2: If no exact match, try to find any available transcript
            if not selected_transcript:
                available_transcripts = list(transcript_list)
                if available_transcripts:
                    # Prefer manual over auto-generated if we have a choice
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

            # Apply translation if requested
            if translate_to and translate_to != selected_language:
                try:
                    selected_transcript = selected_transcript.translate(translate_to)
                    selected_language = translate_to
                except Exception as e:
                    logger.warning(f"Translation to {translate_to} failed: {e}")
                    # Continue with original language

            # Fetch transcript data
            try:
                transcript_data = selected_transcript.fetch()
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Failed to fetch caption data: {str(e)}"
                }

            # Process and format transcript data
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

            # Format output
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
            logger.error(f"Error getting captions: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def get_all_available_captions(self, url: str) -> Dict[str, any]:
        """Get information about all available caption tracks"""
        try:
            video_id = self.extract_video_id(url)
            if not video_id:
                return {"success": False, "error": "Invalid YouTube URL"}

            available_captions = await self.get_available_captions(video_id)

            return {
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
    """Example usage of the Enhanced YouTube Video Service"""
    # Test URLs with different caption scenarios
    test_urls = [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",  # Rick Roll - usually has captions
        "https://www.youtube.com/watch?v=5qo4-D4Mn3E",  # Me at the zoo - first YouTube video
    ]

    service = YouTubeVideoService()

    for url in test_urls:
        print(f"\n{'=' * 60}")
        print(f"Testing URL: {url}")
        print('=' * 60)

        try:
            # Get video information
            print("\n1. Getting video information...")
            video_info = await service.get_video_info(url)
            print(f"   Title: {video_info.title}")
            print(f"   Author: {video_info.author}")
            print(f"   Duration: {video_info.duration}s" if video_info.duration else "   Duration: Unknown")
            print(f"   View Count: {video_info.view_count:,}" if video_info.view_count else "   View Count: Unknown")
            print(f"   Available Caption Tracks: {len(video_info.available_captions)}")

            # List available captions
            print("\n2. Available caption languages:")
            if video_info.available_captions:
                for track in video_info.available_captions:
                    status = "Auto-generated" if track.is_auto_generated else "Manual"
                    translatable = "Translatable" if track.is_translatable else "Not translatable"
                    print(f"   - {track.language_code} ({track.language_name}) - {status}, {translatable}")
            else:
                print("   No captions available")

            # Try to get captions in multiple languages
            print("\n3. Testing caption retrieval...")

            # Test 1: English captions
            captions = await service.get_captions(url, languages=['en'])
            if captions["success"]:
                print(f"   ✓ English captions found ({captions['total_segments']} segments)")
                print(f"     Language: {captions['language']} ({captions['language_name']})")
                print(f"     Auto-generated: {captions['is_auto_generated']}")

                # Show ALL captions (full length)
                if captions["subtitles"]:
                    print("\n     FULL CAPTIONS:")
                    print("     " + "="*50)
                    for i, sub in enumerate(captions["subtitles"], 1):
                        timestamp = f"[{sub['start']:.1f}s - {sub['start'] + sub['duration']:.1f}s]"
                        print(f"     {i:3d}. {timestamp:20} {sub['text']}")
                    print("     " + "="*50)
            else:
                print(f"   ✗ English captions failed: {captions['error']}")

            # Test 2: Multiple language preference
            captions_multi = await service.get_captions(url, languages=['es', 'fr', 'en'])
            if captions_multi["success"]:
                print(f"   ✓ Multi-language search found: {captions_multi['language']}")
            else:
                print(f"   ✗ Multi-language search failed: {captions_multi['error']}")

            # Test 3: Translation (if captions exist)
            if captions["success"]:
                translated = await service.get_captions(url, languages=['en'], translate_to='es')
                if translated["success"]:
                    print(f"   ✓ Translation to Spanish successful")
                    print(f"     Was translated: {translated['was_translated']}")
                else:
                    print(f"   ✗ Translation failed: {translated['error']}")

            # Test 4: Different formats
            if captions["success"]:
                print("\n4. Testing different formats...")
                for fmt in ['txt', 'srt', 'vtt']:
                    formatted = await service.get_captions(url, languages=['en'], format_type=fmt)
                    if formatted["success"]:
                        print(f"   ✓ {fmt.upper()} format: {len(formatted['text'])} chars")
                        print(f"     FULL {fmt.upper()} OUTPUT:")
                        print("     " + "-"*40)
                        # Print the full formatted text
                        lines = formatted["text"].split('\n')
                        for line in lines:
                            print(f"     {line}")
                        print("     " + "-"*40)

        except Exception as e:
            print(f"Error processing {url}: {e}")

    print(f"\n{'=' * 60}")
    print("Testing complete!")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    asyncio.run(main())