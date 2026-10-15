# [PACKAGE DOC] Package for the v0.1 generation spec: Pydantic models describing pipeline input. Re-
# exports the main input models from spec.input (AuthSpec, BootstrapSpec, GenerationInput, GPUSpec,
# LLMSpec, OutputSpec, PipelineName, PipelineSpec, QASpec, RepoSpec, SandboxSpec) and two options
# models from spec.options (PRDiffOptions, PRRuntimeOptions). Other options models live in
# spec.options and spec.recipe_options. The output task model lives in emitter/harbor.py.
"""v0.1 spec — input/output Pydantic models."""

from repo2rlenv.spec.input import (
    AuthSpec,
    BootstrapSpec,
    GenerationInput,
    GPUSpec,
    LLMSpec,
    OutputSpec,
    PipelineName,
    PipelineSpec,
    QASpec,
    RepoSpec,
    SandboxSpec,
)
from repo2rlenv.spec.options import (
    PRDiffOptions,
    PRRuntimeOptions,
)

__all__ = [
    "AuthSpec",
    "BootstrapSpec",
    "GPUSpec",
    "GenerationInput",
    "LLMSpec",
    "OutputSpec",
    "PRDiffOptions",
    "PRRuntimeOptions",
    "PipelineName",
    "PipelineSpec",
    "QASpec",
    "RepoSpec",
    "SandboxSpec",
]
