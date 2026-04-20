from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PromptSection:
    name: str
    title: str
    path: Path
    description: str
    tags: tuple[str, ...]


class PromptLibrary:
    def __init__(self, base_dir: Path | None = None) -> None:
        prompts_dir = base_dir or (Path(__file__).resolve().parent.parent / "prompts")
        self._base_dir = prompts_dir
        self._core_sections = (
            PromptSection(
                name="core_agent",
                title="Core Agent",
                path=self._base_dir / "core_agent.md",
                description="Core planner instructions and operational sales rules.",
                tags=("core", "planner", "sales"),
            ),
            PromptSection(
                name="business_rules",
                title="Business Rules",
                path=self._base_dir / "business_rules.md",
                description="Commercial brief, segmentation, qualification, and CTA rules.",
                tags=("business", "qualification", "sales"),
            ),
        )
        self._knowledge_sections = (
            PromptSection(
                name="wabog_company",
                title="Wabog Company",
                path=self._base_dir / "wabog_company.md",
                description="What Wabog does, who it is for, and value proposition.",
                tags=("company", "product", "positioning", "value"),
            ),
            PromptSection(
                name="wabog_pricing",
                title="Wabog Pricing",
                path=self._base_dir / "wabog_pricing.md",
                description="Pricing guidance, constraints, and commercial routing rules.",
                tags=("pricing", "plans", "quote", "cost"),
            ),
            PromptSection(
                name="wabog_integrations",
                title="Wabog Integrations",
                path=self._base_dir / "wabog_integrations.md",
                description="Documented integration guidance and safe phrasing.",
                tags=("integrations", "whatsapp", "implementation"),
            ),
            PromptSection(
                name="wabog_faq",
                title="Wabog FAQ",
                path=self._base_dir / "wabog_faq.md",
                description="FAQ and constraints for undocumented claims.",
                tags=("faq", "implementation", "support", "security"),
            ),
        )
        self._image_prompt = PromptSection(
            name="media_image_prompt",
            title="Media Image Prompt",
            path=self._base_dir / "media_image_prompt.md",
            description="Vision prompt for commercial image summarization.",
            tags=("media", "vision"),
        )
        self._knowledge_by_name = {section.name: section for section in self._knowledge_sections}

    def get_core_sections(self) -> list[PromptSection]:
        return list(self._core_sections)

    def get_knowledge_sections(self) -> list[PromptSection]:
        return list(self._knowledge_sections)

    def get_image_prompt(self) -> str:
        return self.read_section(self._image_prompt.name)

    def read_section(self, name: str) -> str:
        section = self._find_section(name)
        return section.path.read_text(encoding="utf-8").strip()

    def render_prompt_scaffold(self) -> str:
        parts = [
            "# Core Agent",
            self.read_section("core_agent"),
            "",
            "# Business Rules",
            self.read_section("business_rules"),
            "",
            "# Knowledge Context",
            "{{KNOWLEDGE_CONTEXT}}",
            "",
            "Contact:",
            "{{CONTACT}}",
            "",
            "Recent messages:",
            "{{RECENT_MESSAGES}}",
            "",
            "Semantic memories:",
            "{{SEMANTIC_MEMORIES}}",
            "",
            "User message:",
            "{{USER_MESSAGE}}",
        ]
        return "\n".join(parts).strip()

    def get_playground_context(self) -> dict:
        return {
            "core_sections": [
                {
                    "name": section.name,
                    "title": section.title,
                    "content": self.read_section(section.name),
                }
                for section in self._core_sections
            ],
            "knowledge_sections": [
                {
                    "name": section.name,
                    "title": section.title,
                    "description": section.description,
                    "content": self.read_section(section.name),
                }
                for section in self._knowledge_sections
            ],
        }

    def get_sections_by_name(self, section_names: list[str]) -> list[PromptSection]:
        unique_names: list[str] = []
        for name in section_names:
            normalized = name.strip()
            if normalized and normalized in self._knowledge_by_name and normalized not in unique_names:
                unique_names.append(normalized)
        return [self._knowledge_by_name[name] for name in unique_names]

    def search_sections(self, query: str, limit: int = 3) -> list[PromptSection]:
        query_terms = {term.lower() for term in query.split() if term.strip()}
        if not query_terms:
            return []
        scored: list[tuple[int, PromptSection]] = []
        for section in self._knowledge_sections:
            haystack = " ".join((section.name, section.title, section.description, " ".join(section.tags))).lower()
            score = sum(2 for term in query_terms if term in haystack)
            content = self.read_section(section.name).lower()
            score += sum(1 for term in query_terms if term in content)
            if score:
                scored.append((score, section))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [section for _, section in scored[:limit]]

    def _find_section(self, name: str) -> PromptSection:
        for section in (*self._core_sections, *self._knowledge_sections, self._image_prompt):
            if section.name == name:
                return section
        raise KeyError(f"Unknown prompt section: {name}")
