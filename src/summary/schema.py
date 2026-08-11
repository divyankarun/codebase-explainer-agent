"""Pydantic schemas for structured architecture summary outputs."""

from pydantic import BaseModel, Field


class ArchitectureSummary(BaseModel):
    main_modules: list[str] = Field(description="List of key modules or directories in the codebase")
    entry_points: list[str] = Field(description="Key entry point files or initialization paths")
    tech_stack: list[str] = Field(description="Primary technologies, libraries, and frameworks used")
    how_it_connects: str = Field(description="High-level architectural overview of how modules interact")
    notable_patterns: list[str] = Field(description="Design patterns, conventions, or unique architectural choices")
