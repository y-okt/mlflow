import json
import pytest
from unittest.mock import patch, MagicMock

import mlflow
from mlflow.smolagents.autolog import SmolagentsSpanProcessor
from mlflow.tracing.provider import _get_tracer

import smolagents
from smolagents import CodeAgent, InferenceClientModel, ChatMessage

from tests.tracing.helper import get_traces


def global_autolog():
    # Libraries used within tests or crewai library
    mlflow.autolog(exclude_flavors=["openai", "litellm", "langchain"])
    mlflow.utils.import_hooks.notify_module_loaded(smolagents)


def clear_autolog_state():
    from mlflow.utils.autologging_utils import AUTOLOGGING_INTEGRATIONS

    for key in AUTOLOGGING_INTEGRATIONS.keys():
        AUTOLOGGING_INTEGRATIONS[key].clear()
    mlflow.utils.import_hooks._post_import_hooks = {}


@pytest.fixture(params=[mlflow.smolagents.autolog, global_autolog])
def autolog(request):
    clear_autolog_state()

    yield request.param

    clear_autolog_state()


# def test_override_of_span_processor():
#     autolog()

#     from mlflow.tracing.provider import _MLFLOW_TRACER_PROVIDER

#     span_processor = _MLFLOW_TRACER_PROVIDER._active_span_processor._span_processors[0]
#     assert isinstance(span_processor, SmolagentsSpanProcessor)

#     # Ensure the tracer is not reset on _get_tracer invocation
#     _ = _get_tracer(__name__)
#     span_processor = _MLFLOW_TRACER_PROVIDER._active_span_processor._span_processors[0]
#     assert isinstance(span_processor, SmolagentsSpanProcessor)

DUMMY_OUTPUT = ChatMessage(
    role="user",
    content='[{"type": "text", "text": "Explain quantum mechanics in simple terms."}]'
)


def test_smolagents_invoke_simple(monkeypatch, autolog):
    # with patch.object(InferenceClientModel, "__call__", return_value=DUMMY_OUTPUT):
    autolog()
    monkeypatch.setattr(
        "smolagents.InferenceClientModel.__call__",
        lambda self, *args, **kwargs: DUMMY_OUTPUT,
    )
    model = InferenceClientModel(model_id="gpt-3.5-turbo", token="test_id")
    agent = CodeAgent(tools=[], model=model, add_base_tools=True)
    agent.run(
        "Could you give me the 118th number in the Fibonacci sequence?"
    )

    traces = get_traces()
    assert len(traces) == 1
    trace = traces[0]
    assert trace.info.request_id
    assert trace.info.experiment_id == "0"
    assert trace.info.timestamp_ms > 0
    assert isinstance(trace.data.spans, list) and len(trace.data.spans) >= 2

    root_span = next((s for s in trace.data.spans if s.parent_id is None), None)
    child_span = next((s for s in trace.data.spans if s.parent_id == root_span.span_id), None)

    assert root_span is not None and "mlflow.traceRequestId" in root_span.attributes

    inputs = child_span.attributes["mlflow.spanInputs"]
    if isinstance(inputs, str):
        inputs = json.loads(inputs)
    assert isinstance(inputs, dict)

    outputs = child_span.attributes["mlflow.spanOutputs"]
    if isinstance(outputs, str):
        outputs = json.loads(outputs)
    assert isinstance(outputs, list)
