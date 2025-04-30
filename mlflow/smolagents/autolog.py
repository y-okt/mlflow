from typing import Optional

from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan as OTelReadableSpan
from opentelemetry.sdk.trace import Span as OTelSpan
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.trace import set_tracer_provider

from mlflow import MlflowClient
from mlflow.entities import SpanType
from mlflow.exceptions import MlflowException
from mlflow.tracing.export.mlflow import MlflowSpanExporter
from mlflow.tracing.provider import detach_span_from_context, set_span_in_context
from mlflow.tracing.trace_manager import InMemoryTraceManager
from mlflow.tracking import MlflowClient


def _get_span_type(span: OTelSpan) -> str:
    print("get_span_type called", span.attributes)
    # TODO: Implement span type
    return SpanType.CHAT_MODEL


class SmolagentsSpanProcessor(SimpleSpanProcessor):
    def __init__(self):
        self._client = MlflowClient()
        self._trace_manager = InMemoryTraceManager.get_instance()
        self.span_exporter = MlflowSpanExporter()
        self._span_token_dict = {}
        self._trace_id_to_request_id_dict = {}
        self._otel_span_id_to_mlflow_span = {}

    def on_start(self, span: OTelSpan, parent_context: Optional[Context] = None) -> None:
        otel_parent_id = span.parent.span_id if span.parent else None
        span_type = _get_span_type(span)
        print("otel_parent_id", otel_parent_id)
        if otel_parent_id is not None:
            parent_mlflow_span = self._otel_span_id_to_mlflow_span.get(otel_parent_id, None)
            print("parent_mlflow_span", parent_mlflow_span.request_id, parent_mlflow_span.parent_id)
            mlflow_span = self._client.start_span(
                name=span.name,
                request_id=parent_mlflow_span.request_id,
                parent_id=parent_mlflow_span.parent_id,
                span_type=span_type,
                # TODO: inplement inputs and attributes
            )
        else:
            mlflow_span = self._client.start_trace(
                name=span.name,
                span_type=span_type,
                # TODO: inputs, attributes, tags
            )
        self._otel_span_id_to_mlflow_span[span.context.span_id] = mlflow_span
        token = set_span_in_context(mlflow_span)
        self._span_token_dict[span.context.span_id] = token
        self._trace_id_to_request_id_dict[span.context.trace_id] = mlflow_span.request_id

    def on_end(self, span: OTelReadableSpan) -> None:
        request_id = self._trace_id_to_request_id_dict.get(span.context.trace_id, None)
        if request_id is None:
            raise MlflowException(
                f"Request ID for trace {span.context.trace_id} is not found. Cannot end the span."
            )
        try:
            self._client.end_span(
                request_id=request_id,
                span_id=span.context.span_id,
                # TODO: outputs, attributes, status
            )
        finally:
            token = self._span_token_dict.pop(span.context.span_id, None)
            for key, value in self._span_token_dict.items():
                print("key", key, "value", value)
            if not token:
                raise MlflowException(
                    f"Token for span {span.context.span_id} is not found. "
                    "Cannot detach the span from context."
                )
            detach_span_from_context(token)


def _wrap_mlflow_processor_with_smolagents_processor():
    print("Wrap mlflow processor with smolagents processor")
    from mlflow.tracing.provider import _MLFLOW_TRACER_PROVIDER, _setup_tracer_provider

    if not _MLFLOW_TRACER_PROVIDER or not hasattr(
        _MLFLOW_TRACER_PROVIDER, "_active_span_processor"
    ):
        print("MLFLOW_TRACER_PROVIDER is not initialized. Setting up tracer provider.")
        _setup_tracer_provider()

    from mlflow.tracing.provider import _MLFLOW_TRACER_PROVIDER, _MLFLOW_TRACER_PROVIDER_INITIALIZED

    multi_processor = _MLFLOW_TRACER_PROVIDER._active_span_processor
    wrapped = SmolagentsSpanProcessor()
    multi_processor._span_processors = (wrapped,)

    print("Wrapped span processor with SmolagentsSpanProcessor.")

    _MLFLOW_TRACER_PROVIDER_INITIALIZED.done = True


def _set_tracer_provider_with_mlflow_tracer_provider():
    from mlflow.tracing.provider import _MLFLOW_TRACER_PROVIDER

    if _MLFLOW_TRACER_PROVIDER:
        set_tracer_provider(_MLFLOW_TRACER_PROVIDER)
        print("Set tracer provider with MLflow tracer provider.")
    else:
        raise ValueError("MLflow tracer provider not initialized. Please initialize it first.")


def _set_smolagents_instrumentor():
    trace_provider = TracerProvider()
    trace_provider.add_span_processor(SmolagentsSpanProcessor())

    SmolagentsInstrumentor().instrument(tracer_provider=trace_provider)


def setup_smolagents_tracing():
    _wrap_mlflow_processor_with_smolagents_processor()
    _set_tracer_provider_with_mlflow_tracer_provider()
    _set_smolagents_instrumentor()
