import logging
import json
from typing import Optional, Union

from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan as OTelReadableSpan
from opentelemetry.sdk.trace import Span as OTelSpan, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
from opentelemetry.trace import set_tracer_provider

import mlflow
from mlflow import MlflowClient
from mlflow.entities import SpanType
from mlflow.tracing.export.mlflow import MlflowSpanExporter
from mlflow.tracing.processor.mlflow import MlflowSpanProcessor
from mlflow.tracing.provider import set_span_in_context, detach_span_from_context
from mlflow.tracing.trace_manager import InMemoryTraceManager
from mlflow.tracing.constant import SpanAttributeKey
from mlflow.tracking import MlflowClient
from mlflow.tracing.utils import encode_span_id
from mlflow.exceptions import MlflowException


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

    def on_start(self, span: OTelSpan, parent_context: Optional[Context] = None) -> None:
        try:
            parent = mlflow.get_current_active_span()
        except Exception as e:
            parent = None
        print('parent', parent)
        span_type = _get_span_type(span)
        if parent:
            mlflow_span = self._client.start_span(
                name=span.name,
                request_id=parent.request_id,
                parent_id=parent.span_id,
                span_type=span_type,
                # TODO: inplement inputs and attributes
            )
        else:
            mlflow_span = self._client.start_trace(
                name=span.name,
                span_type=span_type,
                # TODO: inputs, attributes, tags
            )
        token = set_span_in_context(mlflow_span)
        self._span_token_dict[span.context.span_id] = token

    def on_end(self, span: OTelReadableSpan) -> None:
        if span._parent is None:
            request_id = str(span.context.trace_id)  # Use otel-generated trace_id as request_id
        else:
            request_id = self._trace_manager.get_request_id_from_trace_id(span.context.trace_id)
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
