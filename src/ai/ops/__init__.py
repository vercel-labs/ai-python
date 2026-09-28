"""Model operations beyond LLM chat: media generation and friends."""

from .audio import AudioParams, AudioPrompt, generate_audio
from .embeddings import EmbedParams, embed
from .evaluation import (
    BooleanAnswer,
    BooleanCriteria,
    BooleanQuestion,
    ChoiceAnswer,
    ChoiceQuestion,
    EvaluationInput,
    EvaluationParams,
    ScoreAnswer,
    ScoreQuestion,
    experimental_evaluate,
)
from .images import ImageParams, ImagePrompt, generate_image
from .items import Item, Warning
from .reranking import RankedDocument, RerankParams, rerank
from .transcriptions import (
    TranscribeParams,
    Transcription,
    TranscriptionSegment,
    transcribe,
)
from .videos import FrameImage, VideoParams, VideoPrompt, generate_video

__all__ = [
    "AudioParams",
    "AudioPrompt",
    "BooleanAnswer",
    "BooleanCriteria",
    "BooleanQuestion",
    "ChoiceAnswer",
    "ChoiceQuestion",
    "EmbedParams",
    "EvaluationInput",
    "EvaluationParams",
    "FrameImage",
    "ImageParams",
    "ImagePrompt",
    "Item",
    "RankedDocument",
    "RerankParams",
    "ScoreAnswer",
    "ScoreQuestion",
    "TranscribeParams",
    "Transcription",
    "TranscriptionSegment",
    "VideoParams",
    "VideoPrompt",
    "Warning",
    "embed",
    "experimental_evaluate",
    "generate_audio",
    "generate_image",
    "generate_video",
    "rerank",
    "transcribe",
]
