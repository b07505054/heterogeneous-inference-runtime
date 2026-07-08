import json

from benchmark.backends.openai_compatible import OpenAICompatibleBackend, OpenAICompatibleConfig
from benchmark.metrics import openai_latency_metrics
from benchmark.runner import BenchmarkRunner


class FakeResponse:
    def __init__(self, lines=None, body=None):
        self.lines = lines or []
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def __iter__(self):
        return iter(self.lines)

    def read(self):
        return self.body or b"{}"


def test_openai_backend_streaming_metrics_with_mocked_response():
    def fake_urlopen(request, timeout):
        if request.full_url.endswith("/v1/models"):
            return FakeResponse(body=json.dumps({"data": [{"id": "m"}]}).encode("utf-8"))
        lines = [
            b'data: {"choices":[{"delta":{"content":"a"}}]}\n\n',
            b'data: {"choices":[{"delta":{"content":"b"}}]}\n\n',
            b"data: [DONE]\n\n",
        ]
        return FakeResponse(lines=lines)

    backend = OpenAICompatibleBackend(
        OpenAICompatibleConfig(base_url="http://test", model="m", concurrency=1),
        urlopen=fake_urlopen,
    )
    metadata = backend.fetch_model_metadata()
    runner = BenchmarkRunner(
        measure_fn=backend.execute,
        finalize_fn=lambda rows: {
            "metrics": openai_latency_metrics(rows),
            "request_results": rows,
        },
    )
    runner.run([{"messages": [{"role": "user", "content": "hi"}]}])
    result = runner.finalize()
    assert metadata["ok"] is True
    assert result["metrics"]["success_count"] == 1
    assert result["metrics"]["error_count"] == 0
    assert result["metrics"]["total_output_tokens"] == 2
    assert result["request_results"][0]["ttft_ms"] is not None


def test_openai_backend_records_errors_with_mocked_response():
    def fake_urlopen(request, timeout):
        raise TimeoutError("slow")

    backend = OpenAICompatibleBackend(
        OpenAICompatibleConfig(base_url="http://test", model="m", concurrency=1),
        urlopen=fake_urlopen,
    )
    runner = BenchmarkRunner(
        measure_fn=backend.execute,
        finalize_fn=lambda rows: {
            "metrics": openai_latency_metrics(rows),
            "request_results": rows,
        },
    )
    runner.run([{"prompt": "hi"}])
    result = runner.finalize()
    assert result["metrics"]["success_count"] == 0
    assert result["metrics"]["error_count"] == 1
    assert result["request_results"][0]["error_type"] == "TimeoutError"


def test_openai_backend_does_not_duplicate_v1_when_base_url_includes_v1():
    seen_urls = []

    def fake_urlopen(request, timeout):
        seen_urls.append(request.full_url)
        if request.full_url.endswith("/models"):
            return FakeResponse(body=json.dumps({"data": [{"id": "m"}]}).encode("utf-8"))
        return FakeResponse(lines=[b"data: [DONE]\n\n"])

    backend = OpenAICompatibleBackend(
        OpenAICompatibleConfig(base_url="http://test/v1", model="m", concurrency=1),
        urlopen=fake_urlopen,
    )

    backend.fetch_model_metadata()
    backend.execute({"messages": [{"role": "user", "content": "hi"}]})

    assert seen_urls == ["http://test/v1/models", "http://test/v1/chat/completions"]
    assert all("/v1/v1/" not in url for url in seen_urls)


def test_openai_backend_appends_v1_when_base_url_excludes_v1():
    seen_urls = []

    def fake_urlopen(request, timeout):
        seen_urls.append(request.full_url)
        if request.full_url.endswith("/models"):
            return FakeResponse(body=json.dumps({"data": [{"id": "m"}]}).encode("utf-8"))
        return FakeResponse(lines=[b"data: [DONE]\n\n"])

    backend = OpenAICompatibleBackend(
        OpenAICompatibleConfig(base_url="http://test", model="m", concurrency=1),
        urlopen=fake_urlopen,
    )

    backend.fetch_model_metadata()
    backend.execute({"messages": [{"role": "user", "content": "hi"}]})

    assert seen_urls == ["http://test/v1/models", "http://test/v1/chat/completions"]
    assert all("/v1/v1/" not in url for url in seen_urls)


def test_openai_backend_does_not_duplicate_v1_with_trailing_slash():
    seen_urls = []

    def fake_urlopen(request, timeout):
        seen_urls.append(request.full_url)
        if request.full_url.endswith("/models"):
            return FakeResponse(body=json.dumps({"data": [{"id": "m"}]}).encode("utf-8"))
        return FakeResponse(lines=[b"data: [DONE]\n\n"])

    backend = OpenAICompatibleBackend(
        OpenAICompatibleConfig(base_url="http://test/v1/", model="m", concurrency=1),
        urlopen=fake_urlopen,
    )

    backend.fetch_model_metadata()
    backend.execute({"messages": [{"role": "user", "content": "hi"}]})

    assert seen_urls == ["http://test/v1/models", "http://test/v1/chat/completions"]
    assert all("/v1/v1/" not in url for url in seen_urls)
