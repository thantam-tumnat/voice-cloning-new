import json
import logging
from typing import List, Tuple, Any, Optional
from app.config import settings
from app.models import Segment, Tone, AnnotateResponse, LLMAnnotationResult, LLMTagConversionResult
from app.prompts import (
    SYSTEM_PROMPT,
    FEW_SHOT_EXAMPLES,
    ANNOTATE_TOOL,
    TAG_CONVERSION_SYSTEM_PROMPT,
    CONVERT_TAG_TOOL,
)
from app.validator import validate_and_build_segments, ValidationError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Anthropic Helpers
# ---------------------------------------------------------------------------

def build_anthropic_system_blocks() -> list:
    """Build Anthropic system message with prompt caching enabled."""
    return [
        {
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"}
        }
    ]


def build_anthropic_messages(clauses: List[str], guidance: Optional[str] = None) -> list:
    """Construct Anthropic conversation messages with few-shot examples."""
    messages = []
    for eg in FEW_SHOT_EXAMPLES:
        messages.append({
            "role": "user",
            "content": json.dumps(eg["input"], ensure_ascii=False)
        })
        messages.append({
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": f"call_{eg['output']['labels'][0]['i']}",
                    "name": "annotate_clauses",
                    "input": eg["output"]
                }
            ]
        })

    user_payload: dict[str, Any] = {
        "clauses": [{"i": idx, "text": clause} for idx, clause in enumerate(clauses)]
    }
    if guidance and guidance.strip():
        user_payload["user_tone_guidance"] = guidance.strip()

    messages.append({
        "role": "user",
        "content": json.dumps(user_payload, ensure_ascii=False)
    })
    return messages


# ---------------------------------------------------------------------------
# Gemini (Google AI Studio) Helpers
# ---------------------------------------------------------------------------

def build_gemini_prompt(clauses: List[str], guidance: Optional[str] = None) -> str:
    """Build prompt for Gemini containing instructions, few-shot examples, and target clauses."""
    few_shot_text = ""
    for idx, eg in enumerate(FEW_SHOT_EXAMPLES, 1):
        few_shot_text += f"\n--- Example {idx} ---\nInput:\n{json.dumps(eg['input'], ensure_ascii=False)}\nOutput:\n{json.dumps(eg['output'], ensure_ascii=False)}\n"

    target_payload: dict[str, Any] = {
        "clauses": [{"i": idx, "text": clause} for idx, clause in enumerate(clauses)]
    }
    if guidance and guidance.strip():
        target_payload["user_tone_guidance"] = guidance.strip()

    return f"{few_shot_text}\n--- Target Task ---\nInput:\n{json.dumps(target_payload, ensure_ascii=False)}\nOutput:"


# ---------------------------------------------------------------------------
# Main Annotator Engine
# ---------------------------------------------------------------------------

def detect_provider(model_name: Optional[str]) -> str:
    """Infer provider based on model name prefix or fallback to configured provider."""
    if not model_name or not model_name.strip():
        return settings.llm_provider.lower()
    m = model_name.strip().lower()
    if any(m.startswith(prefix) for prefix in ["qwen", "deepseek", "gpt-", "o1", "o3", "chatgpt"]):
        return "openai"
    if any(m.startswith(prefix) for prefix in ["claude"]):
        return "anthropic"
    if any(m.startswith(prefix) for prefix in ["gemini", "gemma"]):
        return "gemini"
    return settings.llm_provider.lower()


class Annotator:
    def __init__(
        self,
        anthropic_client: Optional[Any] = None,
        gemini_client: Optional[Any] = None,
        openai_client: Optional[Any] = None,
    ):
        self._anthropic_client = anthropic_client
        self._gemini_client = gemini_client
        self._openai_client = openai_client

    def get_anthropic_client(self):
        if self._anthropic_client is not None:
            return self._anthropic_client
        import anthropic
        return anthropic.Anthropic(api_key=settings.anthropic_api_key or "dummy-key")

    def get_gemini_client(self):
        if self._gemini_client is not None:
            return self._gemini_client
        from google import genai
        return genai.Client(api_key=settings.effective_gemini_api_key or "dummy-key")

    def get_openai_client(self):
        if self._openai_client is not None:
            return self._openai_client
        from openai import OpenAI
        return OpenAI(
            api_key=settings.effective_openai_api_key or "dummy-key",
            base_url=settings.openai_base_url or None,
        )

    def _call_anthropic(self, client: Any, model: str, clauses: List[str], guidance: Optional[str] = None) -> List[Any]:
        """Execute structured tool use call via Anthropic."""
        system_blocks = build_anthropic_system_blocks()
        messages = build_anthropic_messages(clauses, guidance=guidance)

        response = client.messages.create(
            model=model,
            max_tokens=2048,
            temperature=0,
            system=system_blocks,
            messages=messages,
            tools=[ANNOTATE_TOOL],
            tool_choice={"type": "tool", "name": "annotate_clauses"}
        )

        for block in response.content:
            if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == "annotate_clauses":
                input_data = getattr(block, "input", {})
                if isinstance(input_data, dict) and "labels" in input_data:
                    return input_data["labels"]
                elif isinstance(input_data, list):
                    return input_data

        raise ValidationError("Anthropic LLM did not invoke annotate_clauses tool properly")

    def _call_gemini(self, client: Any, model: str, clauses: List[str], guidance: Optional[str] = None) -> List[Any]:
        """Execute Structured Output JSON call via Google AI Studio (Gemini)."""
        from google.genai import types

        prompt = build_gemini_prompt(clauses, guidance=guidance)
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=LLMAnnotationResult,
            temperature=0.0,
        )

        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=config
        )

        if not response.text:
            raise ValidationError("Gemini returned empty response")

        try:
            parsed = LLMAnnotationResult.model_validate_json(response.text)
            return [label.model_dump() for label in parsed.labels]
        except Exception as e:
            # Fallback to json dict parsing
            try:
                data = json.loads(response.text)
                if isinstance(data, dict) and "labels" in data:
                    return data["labels"]
                elif isinstance(data, list):
                    return data
            except Exception:
                pass
            raise ValidationError(f"Failed to parse Gemini output: {e}")

    def _call_openai(self, client: Any, model: str, clauses: List[str], guidance: Optional[str] = None) -> List[Any]:
        """Execute Structured JSON call via OpenAI-compatible API (e.g. 9arm Gateway)."""
        prompt = build_gemini_prompt(clauses, guidance=guidance)

        response = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            temperature=0.0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT + "\nOutput valid JSON with key 'labels' matching the required schema."},
                {"role": "user", "content": prompt}
            ]
        )
        content = response.choices[0].message.content
        if not content:
            raise ValidationError("OpenAI returned empty response")

        try:
            parsed = LLMAnnotationResult.model_validate_json(content)
            return [label.model_dump() for label in parsed.labels]
        except Exception:
            try:
                data = json.loads(content)
                if isinstance(data, dict) and "labels" in data:
                    return data["labels"]
                elif isinstance(data, list):
                    return data
            except Exception:
                pass
            raise ValidationError(f"Failed to parse OpenAI output: {content[:200]}")

    def _run_provider(self, provider: str, model: str, clauses: List[str], guidance: Optional[str] = None) -> List[Any]:
        if provider == "gemini":
            client = self.get_gemini_client()
            return self._call_gemini(client, model, clauses, guidance=guidance)
        elif provider == "openai":
            client = self.get_openai_client()
            return self._call_openai(client, model, clauses, guidance=guidance)
        else:
            client = self.get_anthropic_client()
            return self._call_anthropic(client, model, clauses, guidance=guidance)

    def annotate(
        self,
        original_text: str,
        clauses: List[str],
        guidance: Optional[str] = None,
        custom_model: Optional[str] = None
    ) -> AnnotateResponse:
        """
        Annotate clauses with emotional tones.
        Executes primary model -> escalation model -> fallback neutral.
        """
        if not clauses:
            return AnnotateResponse(
                original=original_text,
                segments=[],
                model_used="none",
                fallback=False
            )

        provider = detect_provider(custom_model)
        if provider == "gemini":
            primary_model = (custom_model.strip() if custom_model and custom_model.strip() else settings.gemini_model)
            escalate_model = settings.gemini_escalate_model
        elif provider == "openai":
            primary_model = (custom_model.strip() if custom_model and custom_model.strip() else settings.openai_model)
            escalate_model = settings.openai_escalate_model
        else:
            primary_model = (custom_model.strip() if custom_model and custom_model.strip() else settings.llm_model)
            escalate_model = settings.llm_escalate_model

        attempts: List[dict] = []

        # Attempt 1: Primary Model
        try:
            raw_labels = self._run_provider(provider, primary_model, clauses, guidance=guidance)
            segments = validate_and_build_segments(
                original_text=original_text,
                clauses=clauses,
                raw_labels=raw_labels,
                max_segments=settings.max_segments
            )
            attempts.append({"model": primary_model, "provider": provider, "status": "success"})
            return AnnotateResponse(
                original=original_text,
                segments=segments,
                model_used=primary_model,
                fallback=False,
                attempts=attempts
            )
        except Exception as err:
            err_msg = str(err)
            logger.warning(f"Primary model {primary_model} ({provider}) failed: {err_msg}. Escalating to {escalate_model}")
            attempts.append({"model": primary_model, "provider": provider, "status": "failed", "error": err_msg})

        # Attempt 2: Escalation Model (only if different from primary)
        if escalate_model and escalate_model != primary_model:
            try:
                raw_labels = self._run_provider(provider, escalate_model, clauses, guidance=guidance)
                segments = validate_and_build_segments(
                    original_text=original_text,
                    clauses=clauses,
                    raw_labels=raw_labels,
                    max_segments=settings.max_segments
                )
                attempts.append({"model": escalate_model, "provider": provider, "status": "success"})
                return AnnotateResponse(
                    original=original_text,
                    segments=segments,
                    model_used=escalate_model,
                    fallback=False,
                    attempts=attempts
                )
            except Exception as err:
                err_msg = str(err)
                logger.error(f"Escalation model {escalate_model} ({provider}) failed: {err_msg}. Falling back to neutral.")
                attempts.append({"model": escalate_model, "provider": provider, "status": "failed", "error": err_msg})

        # Attempt 3: Safe Fallback
        fallback_segments = [
            Segment(
                text=original_text,
                tone=Tone.NEUTRAL,
                intensity=2
            )
        ]
        error_summary = "; ".join([f"[{a['model']}]: {a.get('error', 'unknown error')}" for a in attempts if a.get('status') == 'failed'])
        return AnnotateResponse(
            original=original_text,
            segments=fallback_segments,
            model_used="fallback-neutral",
            fallback=True,
            error="LLM annotation failed (e.g. Quota Exceeded or API error)",
            error_detail=error_summary,
            attempts=attempts
        )

    def convert_style_tag(
        self,
        tag: str,
        intensity: int = 2,
        custom_model: Optional[str] = None
    ) -> Optional[dict]:
        """Convert a free-form emotion tag (e.g. 'sad and cry') to a VoxCPM-compatible instruction."""
        if not tag or not tag.strip():
            return None

        clean_tag = tag.strip()
        provider = detect_provider(custom_model)
        if provider == "gemini":
            primary_model = (custom_model.strip() if custom_model and custom_model.strip() else settings.gemini_model)
            escalate_model = settings.gemini_escalate_model
        elif provider == "openai":
            primary_model = (custom_model.strip() if custom_model and custom_model.strip() else settings.openai_model)
            escalate_model = settings.openai_escalate_model
        else:
            primary_model = (custom_model.strip() if custom_model and custom_model.strip() else settings.llm_model)
            escalate_model = settings.llm_escalate_model

        for model in filter(None, [primary_model, escalate_model]):
            try:
                if provider == "gemini":
                    from google.genai import types
                    client = self.get_gemini_client()
                    config = types.GenerateContentConfig(
                        system_instruction=TAG_CONVERSION_SYSTEM_PROMPT,
                        response_mime_type="application/json",
                        response_schema=LLMTagConversionResult,
                        temperature=0.0,
                    )
                    prompt = f"Convert this emotion tag to a VoxCPM2 style instruction: '{clean_tag}' (default intensity: {intensity})"
                    response = client.models.generate_content(
                        model=model,
                        contents=prompt,
                        config=config
                    )
                    if response.text:
                        parsed = LLMTagConversionResult.model_validate_json(response.text)
                        instr = parsed.instruction.strip()
                        if not instr.startswith("("):
                            instr = f"({instr}"
                        if not instr.endswith(")"):
                            instr = f"{instr})"
                        return {
                            "instruction": instr,
                            "tone": parsed.tone,
                            "intensity": parsed.intensity,
                        }
                elif provider == "openai":
                    client = self.get_openai_client()
                    prompt = f"Convert this emotion tag to a VoxCPM2 style instruction: '{clean_tag}' (default intensity: {intensity})"
                    response = client.chat.completions.create(
                        model=model,
                        response_format={"type": "json_object"},
                        temperature=0.0,
                        messages=[
                            {"role": "system", "content": TAG_CONVERSION_SYSTEM_PROMPT + "\nReturn JSON object with keys: instruction, tone, intensity."},
                            {"role": "user", "content": prompt}
                        ]
                    )
                    content = response.choices[0].message.content
                    if content:
                        parsed = json.loads(content)
                        instr = str(parsed.get("instruction", "")).strip()
                        if instr:
                            if not instr.startswith("("):
                                instr = f"({instr}"
                            if not instr.endswith(")"):
                                instr = f"{instr})"
                            tone_str = str(parsed.get("tone", "neutral")).lower()
                            tone_enum = Tone(tone_str) if tone_str in Tone._value2member_map_ else Tone.NEUTRAL
                            return {
                                "instruction": instr,
                                "tone": tone_enum,
                                "intensity": int(parsed.get("intensity", intensity)),
                            }
                else:
                    client = self.get_anthropic_client()
                    response = client.messages.create(
                        model=model,
                        max_tokens=512,
                        temperature=0,
                        system=TAG_CONVERSION_SYSTEM_PROMPT,
                        messages=[
                            {"role": "user", "content": f"Convert this emotion tag to a VoxCPM2 style instruction: '{clean_tag}' (default intensity: {intensity})"}
                        ],
                        tools=[CONVERT_TAG_TOOL],
                        tool_choice={"type": "tool", "name": "convert_style_tag"}
                    )
                    for block in response.content:
                        if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == "convert_style_tag":
                            input_data = getattr(block, "input", {})
                            instr = str(input_data.get("instruction", "")).strip()
                            if instr:
                                if not instr.startswith("("):
                                    instr = f"({instr}"
                                if not instr.endswith(")"):
                                    instr = f"{instr})"
                                tone_str = str(input_data.get("tone", "neutral")).lower()
                                tone_enum = Tone(tone_str) if tone_str in Tone._value2member_map_ else Tone.NEUTRAL
                                return {
                                    "instruction": instr,
                                    "tone": tone_enum,
                                    "intensity": int(input_data.get("intensity", intensity)),
                                }
            except Exception as e:
                logger.warning(f"[convert_style_tag] Failed with model {model} ({provider}): {e}")
                continue

        return None

    def list_available_models(self, refresh: bool = False) -> dict:
        """Fetch all available models from configured providers (Gemini, Anthropic, OpenAI/9arm)."""
        if refresh:
            import app.config as app_cfg
            app_cfg.settings = app_cfg.Settings()
            self._openai_client = None
            self._gemini_client = None
            self._anthropic_client = None
            self._cached_models = None
        elif getattr(self, "_cached_models", None) is not None:
            return self._cached_models

        providers: dict[str, Any] = {}

        # 1. Google Gemini
        has_gemini = bool(settings.effective_gemini_api_key and settings.effective_gemini_api_key != "dummy-key")
        gemini_models = []
        if has_gemini:
            try:
                client = self.get_gemini_client()
                raw_models = client.models.list()
                for m in raw_models:
                    actions = getattr(m, "supported_actions", []) or getattr(m, "supported_generation_methods", []) or []
                    name = m.name.replace("models/", "") if hasattr(m, "name") else str(m)
                    # Filter out non-content models (embeddings, tts, image generators, audio clips)
                    if not any(x in name.lower() for x in ["tts", "image", "embedding", "aqa", "clip"]):
                        is_rec = (name == settings.gemini_model)
                        label = name + (" (⚡ แนะนำ: Google)" if is_rec else "")
                        gemini_models.append({
                            "id": name,
                            "name": label,
                            "recommended": is_rec,
                        })
            except Exception as e:
                logger.warning(f"Could not list Gemini models live: {e}")

        if not gemini_models:
            for m_id in [
                "gemini-3.6-flash",
                "gemini-3.7-flash",
                "gemini-3.5-flash-lite",
                "gemini-3.1-flash-lite",
                "gemini-flash-lite-latest",
                "gemini-flash-latest",
                "gemini-pro-latest",
            ]:
                is_rec = (m_id == settings.gemini_model)
                gemini_models.append({
                    "id": m_id,
                    "name": m_id + (" (⚡ แนะนำ: Google)" if is_rec else ""),
                    "recommended": is_rec,
                })

        providers["gemini"] = {
            "available": has_gemini,
            "default": settings.gemini_model,
            "escalate": settings.gemini_escalate_model,
            "models": sorted(gemini_models, key=lambda x: (not x["recommended"], x["id"])),
        }

        # 2. 9arm Gateway / OpenAI-Compatible
        has_openai = bool(settings.effective_openai_api_key and settings.effective_openai_api_key != "dummy-key")
        openai_models = []
        if has_openai:
            try:
                client = self.get_openai_client()
                raw_data = client.models.list().data
                for m in raw_data:
                    m_id = m.id if hasattr(m, "id") else str(m)
                    is_rec = (m_id == settings.openai_model)
                    label = f"{m_id}" + (" (⚡ แนะนำ: 9arm Gateway)" if is_rec else " (9arm Gateway)")
                    openai_models.append({
                        "id": m_id,
                        "name": label,
                        "recommended": is_rec,
                    })
            except Exception as e:
                logger.warning(f"Could not list OpenAI/9arm models live: {e}")

        if not openai_models:
            for m_id in [settings.openai_model, settings.openai_escalate_model]:
                if m_id:
                    is_rec = (m_id == settings.openai_model)
                    openai_models.append({
                        "id": m_id,
                        "name": f"{m_id} (9arm Gateway)",
                        "recommended": is_rec,
                    })

        providers["openai"] = {
            "available": has_openai,
            "default": settings.openai_model,
            "escalate": settings.openai_escalate_model,
            "base_url": settings.openai_base_url,
            "models": sorted(openai_models, key=lambda x: (not x["recommended"], x["id"])),
        }

        # 3. Anthropic Claude
        has_anthropic = bool(settings.anthropic_api_key and settings.anthropic_api_key != "dummy-key")
        claude_models = [
            {"id": "claude-haiku-4-5", "name": "claude-haiku-4-5 (⚡ แนะนำ: เร็ว & ประหยัด)", "recommended": True},
            {"id": "claude-sonnet-5", "name": "claude-sonnet-5 (✨ โมเดลแม่นยำสูง)", "recommended": False},
            {"id": "claude-opus-4-6", "name": "claude-opus-4-6", "recommended": False},
        ]
        providers["anthropic"] = {
            "available": has_anthropic,
            "default": settings.llm_model,
            "escalate": settings.llm_escalate_model,
            "models": claude_models,
        }

        def_model = settings.gemini_model
        if settings.llm_provider == "openai" and has_openai:
            def_model = settings.openai_model
        elif settings.llm_provider == "anthropic" and has_anthropic:
            def_model = settings.llm_model

        result = {
            "providers": providers,
            "current_provider": settings.llm_provider,
            "default_model": def_model,
        }
        self._cached_models = result
        return result


annotator = Annotator()

