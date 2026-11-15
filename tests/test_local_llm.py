import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import llm_client
import local_llm


def test_backend_selection(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("MHAI_LLM", raising=False)
    assert llm_client.backend() == "" and not llm_client.is_available()
    monkeypatch.setenv("MHAI_LLM", "ollama")
    assert llm_client.backend() == "ollama" and isinstance(llm_client._client(), local_llm.OllamaClient)


def test_payload_translates_anthropic_style_arguments():
    p = local_llm.build_payload(model="claude-sonnet-5-5", max_tokens=4000,
                                system=[{"type": "text", "text": "A"}, {"type": "text", "text": "B"}],
                                messages=[{"role": "user", "content": "xin chào"}],
                                output_config={"format": {"type": "json_schema", "schema": {"type": "object"}}, "effort": "low"})
    assert p["model"] == local_llm.DEFAULT_MODEL and p["messages"][0] == {"role": "system", "content": "A\n\nB"}
    assert p["messages"][1]["content"] == "xin chào" and p["format"] == {"type": "object"} and p["options"]["num_predict"] <= 2048


class _FakeResp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def test_create_and_stream_parse_ollama_responses(monkeypatch):
    monkeypatch.setattr(local_llm, "_post", lambda p: _FakeResp(json.dumps({"message": {"content": " Chào bạn "}}).encode())
                        if not p["stream"] else _FakeResp(b'{"message":{"content":"Ch"}}\n{"message":{"content":"ao"},"done":true}\n'))
    c = local_llm.OllamaClient()
    assert llm_client._text(c.messages.create(model="x", max_tokens=10, messages=[])) == "Chào bạn"
    with c.messages.stream(model="x", max_tokens=10, messages=[]) as s:
        assert "".join(s.text_stream) == "Chao"
