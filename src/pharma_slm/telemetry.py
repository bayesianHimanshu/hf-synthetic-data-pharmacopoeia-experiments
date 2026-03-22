from __future__ import annotations

import json
from pathlib import Path

from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    ConsoleMetricExporter,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.trace.export import SpanExportResult

class _FileSpanExporter:
    """Writes finished spans as JSONL lines to a local file."""

    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def export(self, spans) -> SpanExportResult:
        with open(self._path, "a") as f:
            for span in spans:
                f.write(
                    json.dumps(
                        {
                            "name": span.name,
                            "trace_id": format(span.context.trace_id, "032x"),
                            "span_id": format(span.context.span_id, "016x"),
                            "start_time": span.start_time,
                            "end_time": span.end_time,
                            "status": span.status.status_code.name,
                            "attributes": dict(span.attributes or {}),
                        }
                    )
                    + "\n"
                )
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return True


def setup_telemetry(cfg) -> None:
    """Configure OTel providers from a TelemetryConfig.
    """
    resource = Resource.create({"service.name": cfg.service_name})

    tp = TracerProvider(resource=resource)
    for exp in cfg.exporters:
        if exp.type == "console":
            tp.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        elif exp.type == "otlp":
            tp.add_span_processor(
                BatchSpanProcessor(OTLPSpanExporter(endpoint=exp.endpoint))
            )
        elif exp.type == "file":
            tp.add_span_processor(
                BatchSpanProcessor(_FileSpanExporter(exp.path))
            )
    trace.set_tracer_provider(tp)

    readers = []
    for exp in cfg.exporters:
        if exp.type == "console":
            readers.append(PeriodicExportingMetricReader(ConsoleMetricExporter()))
        elif exp.type == "otlp":
            readers.append(
                PeriodicExportingMetricReader(
                    OTLPMetricExporter(endpoint=exp.endpoint)
                )
            )
    mp = MeterProvider(resource=resource, metric_readers=readers)
    metrics.set_meter_provider(mp)


def get_tracer(name: str) -> trace.Tracer:
    return trace.get_tracer_provider().get_tracer(name)


def get_meter(name: str) -> metrics.Meter:
    return metrics.get_meter_provider().get_meter(name)
