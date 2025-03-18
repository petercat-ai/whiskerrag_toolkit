from typing import List, Optional

from pydantic import BaseModel, Field, model_validator


class BaseSplitConfig(BaseModel):
    """Base split configuration class"""

    chunk_size: int = Field(default=1500, ge=1)
    chunk_overlap: int = Field(default=150, ge=0)

    @model_validator(mode="after")
    def validate_overlap(self) -> "BaseSplitConfig":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return self


class MarkdownSplitConfig(BaseSplitConfig):
    """Markdown document split configuration"""

    separators: Optional[List[str]] = Field(default=None)
    split_regex: Optional[str] = Field(default=None)
    heading_levels: List[int] = Field(
        default=[1, 2],
        description="Heading levels to consider, e.g., [1, 2] means splitting only at # and ##",
    )
    split_by_heading: bool = Field(
        default=True, description="Whether to split by headings"
    )
    remove_markdown_chars: bool = Field(
        default=False, description="Whether to remove Markdown syntax characters"
    )


class PDFSplitConfig(BaseSplitConfig):
    """PDF document split configuration"""

    split_by_page: bool = Field(default=False, description="Whether to split by pages")
    keep_layout: bool = Field(
        default=True, description="Whether to preserve the original layout"
    )
    extract_images: bool = Field(default=False, description="Whether to extract images")
    table_extract_mode: str = Field(
        default="text", description="Table extraction mode: 'text' or 'structure'"
    )


class TextSplitConfig(BaseSplitConfig):
    """Plain text split configuration"""

    separators: Optional[List[str]] = Field(default=None)
    split_regex: Optional[str] = Field(default=None)
    strip_whitespace: bool = Field(
        default=True, description="Whether to remove extra whitespace"
    )


class JSONSplitConfig(BaseSplitConfig):
    """JSON document split configuration"""

    split_level: int = Field(
        default=1, description="Depth level for JSON splitting", ge=1
    )
    preserve_structure: bool = Field(
        default=True, description="Whether to preserve JSON structure"
    )
    array_handling: str = Field(
        default="split",
        description="Array handling mode: 'split' or 'merge'",
    )
    key_filters: Optional[List[str]] = Field(
        default=None, description="List of keys to process; processes all keys if None"
    )

    @model_validator(mode="after")
    def validate_array_handling(self) -> "JSONSplitConfig":
        valid_handlers = ["split", "merge"]
        if self.array_handling not in valid_handlers:
            raise ValueError(f"array_handling must be one of {valid_handlers}")
        return self
