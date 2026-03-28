from __future__ import annotations

import json
from urllib import error, parse, request

import pandas as pd
import streamlit as st

from core.i18n import t
from core.insights import answer_chat_question, build_chat_context, insight_sentences

PROVIDER_OPTIONS = ["rule-based", "openai", "gemini"]
OPENAI_DEFAULT_MODEL = "gpt-4o-mini"
GEMINI_DEFAULT_MODEL = "gemini-2.5-flash"


def _provider_label(provider: str) -> str:
    return t(f"chatbot.provider.{provider}")


def _set_chatbot_status(message: str, tone: str = "info") -> None:
    st.session_state["chatbot_status_message"] = message
    st.session_state["chatbot_status_tone"] = tone


def _sync_chatbot_status() -> None:
    provider = st.session_state.get("chatbot_provider", "rule-based")
    if provider == "rule-based":
        _set_chatbot_status(t("chatbot.status.rule_based"), "info")
        return

    key_name = "openai_api_key" if provider == "openai" else "gemini_api_key"
    api_key = st.session_state.get(key_name, "").strip()
    if api_key:
        _set_chatbot_status(t("chatbot.status.using_provider", provider=_provider_label(provider)), "success")
    else:
        _set_chatbot_status(
            t("chatbot.status.fallback_missing_key", provider=_provider_label(provider)),
            "warning",
        )


def _render_status_line() -> None:
    message = st.session_state.get("chatbot_status_message", "")
    tone = st.session_state.get("chatbot_status_tone", "info")

    if not message:
        _sync_chatbot_status()
        message = st.session_state.get("chatbot_status_message", "")
        tone = st.session_state.get("chatbot_status_tone", "info")

    if tone == "success":
        st.success(message)
    elif tone == "warning":
        st.warning(message)
    else:
        st.info(message)


def _build_system_prompt(frame: pd.DataFrame) -> str:
    response_language = "Vietnamese" if st.session_state.get("language") == "vi" else "English"
    summary_lines = insight_sentences(frame)
    summary_block = "\n".join(f"- {line}" for line in summary_lines) if summary_lines else "- No summary available."
    visible_columns = ", ".join(str(column) for column in frame.columns[:12]) if not frame.empty else "No columns"
    dataset_context = build_chat_context(frame)

    return (
        "You are an Airbnb analytics assistant inside a Streamlit dashboard. "
        f"Answer in {response_language}. Keep the answer concise, factual, and grounded only in the provided dataset summary. "
        "If the user asks beyond this context, clearly say that the current chatbot only has summary-level context from the dashboard.\n\n"
        f"Rows in current dataset: {len(frame):,}\n"
        f"Visible columns: {visible_columns}\n"
        "Dataset summary:\n"
        f"{summary_block}\n\n"
        "Dataset context:\n"
        f"{dataset_context}"
    )


def _post_json(url: str, headers: dict[str, str], payload: dict[str, object]) -> dict[str, object]:
    body = json.dumps(payload).encode("utf-8")
    http_request = request.Request(url, data=body, headers=headers, method="POST")

    try:
        with request.urlopen(http_request, timeout=30) as response:
            raw = response.read().decode("utf-8")
    except error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="ignore")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as decode_error:
            raise RuntimeError(raw or str(exc)) from decode_error

        if isinstance(parsed.get("error"), dict):
            message = parsed["error"].get("message") or raw
            raise RuntimeError(message) from exc
        raise RuntimeError(raw or str(exc)) from exc
    except error.URLError as exc:
        raise RuntimeError(str(exc.reason)) from exc

    try:
        parsed_response = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Provider returned a non-JSON response.") from exc

    if not isinstance(parsed_response, dict):
        raise RuntimeError("Provider returned an unexpected response shape.")
    return parsed_response


def _call_openai(question: str, frame: pd.DataFrame) -> str:
    api_key = st.session_state.get("openai_api_key", "").strip()
    model = OPENAI_DEFAULT_MODEL
    payload = {
        "model": model,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": _build_system_prompt(frame)},
            {"role": "user", "content": question},
        ],
    }
    response = _post_json(
        "https://api.openai.com/v1/chat/completions",
        {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        payload,
    )
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("OpenAI returned no choices.")

    message = choices[0].get("message", {})
    content = message.get("content", "")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("OpenAI returned an empty message.")
    return content.strip()


def _call_gemini(question: str, frame: pd.DataFrame) -> str:
    api_key = st.session_state.get("gemini_api_key", "").strip()
    model = GEMINI_DEFAULT_MODEL
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{parse.quote(model, safe='')}:generateContent?key={parse.quote(api_key, safe='')}"
    )
    payload = {
        "systemInstruction": {
            "parts": [
                {
                    "text": _build_system_prompt(frame),
                }
            ]
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": question}],
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
        },
    }
    response = _post_json(
        url,
        {
            "Content-Type": "application/json",
        },
        payload,
    )
    candidates = response.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise RuntimeError("Gemini returned no candidates.")

    parts = candidates[0].get("content", {}).get("parts", [])
    text_parts = [part.get("text", "") for part in parts if isinstance(part, dict)]
    final_text = "".join(text_parts).strip()
    if not final_text:
        raise RuntimeError("Gemini returned an empty message.")
    return final_text


def _generate_assistant_reply(question: str, frame: pd.DataFrame) -> str:
    provider = st.session_state.get("chatbot_provider", "rule-based")
    fallback_reply = answer_chat_question(question, frame)

    if provider == "rule-based":
        _set_chatbot_status(t("chatbot.status.rule_based"), "info")
        return fallback_reply

    key_name = "openai_api_key" if provider == "openai" else "gemini_api_key"
    api_key = st.session_state.get(key_name, "").strip()
    if not api_key:
        _set_chatbot_status(
            t("chatbot.status.fallback_missing_key", provider=_provider_label(provider)),
            "warning",
        )
        st.info(t("chatbot.provider.missing_key", provider=_provider_label(provider)))
        return fallback_reply

    try:
        if provider == "openai":
            reply = _call_openai(question, frame)
        else:
            reply = _call_gemini(question, frame)
        _set_chatbot_status(t("chatbot.status.using_provider", provider=_provider_label(provider)), "success")
        return reply
    except Exception as exc:
        _set_chatbot_status(
            t("chatbot.status.fallback_failed", provider=_provider_label(provider)),
            "warning",
        )
        st.warning(t("chatbot.provider.failed", provider=_provider_label(provider), error=str(exc)))
        return fallback_reply


def _render_provider_settings() -> None:
    with st.expander(t("chatbot.settings.title"), expanded=False):
        st.caption(t("chatbot.settings.caption"))

        st.selectbox(
            t("chatbot.provider.label"),
            options=PROVIDER_OPTIONS,
            format_func=_provider_label,
            key="chatbot_provider",
        )

        provider = st.session_state.get("chatbot_provider", "rule-based")
        if provider == "openai":
            st.text_input(
                t("chatbot.provider.openai_key"),
                type="password",
                key="openai_api_key",
                placeholder="sk-...",
            )
        elif provider == "gemini":
            st.text_input(
                t("chatbot.provider.gemini_key"),
                type="password",
                key="gemini_api_key",
                placeholder="AIza...",
            )

        st.caption(t("chatbot.provider.network_note"))


def _render_chat_text(content: str) -> None:
    st.markdown(content.replace("$", r"\$"))


def render_page(frame: pd.DataFrame) -> None:
    st.title(t("chatbot.title"))
    st.caption(t("chatbot.caption"))
    _render_provider_settings()
    _sync_chatbot_status()

    quick_prompts = [
        t("chatbot.quick.price"),
        t("chatbot.quick.room_mix"),
        t("chatbot.quick.expensive_area"),
        t("chatbot.quick.availability"),
    ]

    quick_prompt = None
    prompt_cols = st.columns(4)
    for column, prompt_text in zip(prompt_cols, quick_prompts):
        if column.button(prompt_text, use_container_width=True):
            quick_prompt = prompt_text

    status_placeholder = st.empty()

    user_prompt = st.chat_input(t("chatbot.input"))
    final_prompt = user_prompt or quick_prompt

    if final_prompt:
        st.session_state["chat_history"].append({"role": "user", "content": final_prompt})
        assistant_reply = _generate_assistant_reply(final_prompt, frame)
        st.session_state["chat_history"].append({"role": "assistant", "content": assistant_reply})

    with status_placeholder.container():
        _render_status_line()

    for message in st.session_state["chat_history"]:
        with st.chat_message(message["role"]):
            _render_chat_text(message["content"])
