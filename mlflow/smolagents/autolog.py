import logging
import json
from typing import Optional, Union

from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan as OTelReadableSpan
from opentelemetry.sdk.trace import Span as OTelSpan
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
from opentelemetry.trace import set_tracer_provider

import mlflow
from mlflow import MlflowClient
from mlflow.entities import SpanType
from mlflow.tracing.processor.mlflow import MlflowSpanProcessor
from mlflow.tracing.trace_manager import InMemoryTraceManager
from mlflow.tracing.constant import SpanAttributeKey
from mlflow.tracking import MlflowClient
from mlflow.tracing.utils import encode_span_id

_logger = logging.getLogger(__name__)


def _get_span_type(span: OTelSpan) -> str:
    print("get_span_type called", span.attributes)
    # if hasattr(span, "attributes") and (operation := span.attributes.get("mlflow.operation")): # TODO: 正しいoperation
    #     span_type_mapping = {
    #         "chat.completions": SpanType.CHAT_MODEL
    #     }
    #     span_type = span_type_mapping.get(operation)
    #     print("spantype", span_type)
    return SpanType.CHAT_MODEL


class SmolagentsSpanProcessor(SimpleSpanProcessor):
    def __init__(self, base_span_processor: Union[SimpleSpanProcessor, BatchSpanProcessor]):
        self._base_span_processor = base_span_processor
        self.span_exporter = getattr(base_span_processor, "span_exporter", None)

        if not self.span_exporter:
            raise ValueError("Span exporter not found in the base span processor.")

        self._client = MlflowClient()
        self._trace_manager = InMemoryTraceManager.get_instance()
        self._mlflow_span_processor = MlflowSpanProcessor(self.span_exporter, self._client)

    def on_start(self, span: OTelSpan, parent_context: Optional[Context] = None) -> None:
        self._base_span_processor.on_start(span, parent_context)
        parent_id = encode_span_id(span.context.span_id)
        request_id = self._trace_manager.get_request_id_from_trace_id(span.context.trace_id)
        if parent_id:
            self._client.start_span(
                name=span.name,
                request_id=request_id,
                parent_id=parent_id,
                span_type=_get_span_type(span),
            )
        else:
            # self._client.start_trace(
            #     name=span.name,
            #     span_type=_get_span_type(span),
            # )
            mlflow.start_span(name=span.name)
            

    def on_end(self, span: OTelReadableSpan) -> None:
        request_id = json.loads(span.attributes.get(SpanAttributeKey.REQUEST_ID))
        span_id = encode_span_id(span.context.span_id)
        self._client.end_span(
            request_id=request_id,
            span_id=span_id,
        )
        self._base_span_processor.on_end(span)

    # def _on_start(self, span: OTelSpan, parent_context: Optional[Context] = None) -> None:
    #     print("Wrapper on start")
    #     self._base_span_processor.on_start(span, parent_context)
    #     span_type = _get_span_type(span)

    #     # parent_span = mlflow.get_current_active_span()
    #     # if parent_span is not None:
    #     #     self._client.start_span(
    #     #         parent_span.name,
    #     #         request_id=parent_span._trace_id,
    #     #         parent_id=parent_span.span_id,
    #     #         span_type=span_type,
    #     #     )

    #     request_id = self._trace_manager.get_request_id_from_trace_id(span.context.trace_id)
    #     # parent_id = encode_span_id(span.context.span_id)
    #     # parent_id = _encode_span_id_to_byte(span.context.span_id)
    #     live_span = self._trace_manager.get_span_from_id(request_id=request_id, span_id=encode_span_id(span.context.span_id))
    #     if live_span is not None:
    #         print('live_span', live_span.span_id)
    #         parent_id = live_span.span_id
    #     else:
    #         print("live_span doesn't exist")
    #         parent_id = None
    #     # # parent_id = live_span.span_id

    #     if request_id is not None and parent_id is not None:
    #         print("span has a parent", parent_id)
    #         print('request_id', request_id, span.context.trace_id)
    #         self._client.start_span(span.name, request_id=request_id, parent_id=parent_id, span_type=span_type)
    #     print("on_start end")

    # def _on_end(self, span: OTelReadableSpan) -> None:
    #     print("Wrapper on end")
    #     # if span.parent is not None:
    #         # request_id = self._trace_manager.get_request_id_from_trace_id(span.context.trace_id)
    #     request_id = json.loads(span.attributes.get(SpanAttributeKey.REQUEST_ID))
    #     # print("on_end request_id", request_id, encode_span_id(span.context.span_id))
    #     span_id = _encode_span_id_to_byte(span.context.span_id)
    #     print("on_end request_id", request_id, encode_span_id(span.context.span_id), span_id)
    #     self._client.end_span(
    #         request_id=request_id, span_id=span_id
    #     )
    #     # self._base_span_processor.on_end(span) # TODO: これが必要なのか？


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
    current_span_processor = multi_processor._span_processors[0]
    wrapped = SmolagentsSpanProcessor(current_span_processor)
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
    from mlflow.tracing.provider import _MLFLOW_TRACER_PROVIDER

    SmolagentsInstrumentor().instrument(tracer_provider=_MLFLOW_TRACER_PROVIDER)


def setup_smolagents_tracing():
    _wrap_mlflow_processor_with_smolagents_processor()
    _set_tracer_provider_with_mlflow_tracer_provider()
    _set_smolagents_instrumentor()
