# [MODULE DOC] Pydantic schemas (extra fields forbidden) used as LLM response formats for the CLI-
# Gym recipe: the inversion goal, the destruction/recovery scripts, and the repair instruction.
"""Typed environment-inversion goals and reproducible filesystem changes."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class InversionGoal(BaseModel):
    # [CLASS DOC] LLM-proposed environment-damage goal: title (5-120 chars), category,
    # selected_tests (1-50 test ids, must be a subset of the sampled healthy tests; checked by the
    # pipeline), description (40-12000 chars), expected_result, recovery_strategy.
    model_config = ConfigDict(extra="forbid")
    title: str = Field(min_length=5, max_length=120)
    category: str
    selected_tests: list[str] = Field(min_length=1, max_length=50)
    description: str = Field(min_length=40, max_length=12000)
    expected_result: str
    recovery_strategy: str


class Inversion(BaseModel):
    # [CLASS DOC] Pair of bash scripts implementing a goal: destruction_shell and recovery_shell
    # (each 20-12000 chars, must start with "#!/bin/bash" + newline) and explanation (40-6000
    # chars).
    model_config = ConfigDict(extra="forbid")
    destruction_shell: str = Field(min_length=20, max_length=12000)
    recovery_shell: str = Field(min_length=20, max_length=12000)
    explanation: str = Field(min_length=40, max_length=6000)

    @model_validator(mode="after")
    def bash_scripts(self):
        # [FUNCTION DOC] After-validator: raises ValueError unless both scripts start with the exact
        # "#!/bin/bash\n" shebang line. Returns self.
        if any(
            not script.startswith("#!/bin/bash\n")
            for script in (self.destruction_shell, self.recovery_shell)
        ):
            raise ValueError("Inversion and recovery must start with a bash shebang")
        return self


class RepairInstruction(BaseModel):
    # [CLASS DOC] LLM-authored task instruction for the repair task: instruction (100-10000 chars).
    model_config = ConfigDict(extra="forbid")
    instruction: str = Field(min_length=100, max_length=10000)
