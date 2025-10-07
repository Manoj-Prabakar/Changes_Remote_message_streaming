import logging
import inspect

log = logging.getLogger(__name__)

def _install():
    try:
        from google.adk.flows.llm_flows import functions as _fn
    except Exception as e:
        log.warning("patch_adk_tracing: cannot import google.adk.flows.llm_flows.functions: %s", e)
        return

    trace = getattr(_fn, "trace_tool_call", None)
    if trace is None:
        log.warning("patch_adk_tracing: functions.trace_tool_call not found; nothing to patch")
        return

    if getattr(trace, "__ank_patched__", False):
        log.info("patch_adk_tracing: already patched; skip")
        return

    try:
        sig = inspect.signature(trace)
    except Exception:
        sig = None

    def _call_with_filtered_kwargs(func, *args, **kwargs):

        if sig:
            params = sig.parameters
            if "function_response" in kwargs and "response" in params and "response" not in kwargs:
                kwargs["response"] = kwargs.pop("function_response")
            if "function_call" in kwargs and "call" in params and "call" not in kwargs:
                kwargs["call"] = kwargs.pop("function_call")


            accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
            if not accepts_kwargs:
                kwargs = {k: v for k, v in kwargs.items() if k in params}


        for k in ("response_event_id", "request_event_id", "function_response", "function_call", "tool_invocation_id"):
            kwargs.pop(k, None)

        try:
            return func(*args, **kwargs)
        except TypeError as e:

            log.info("trace_tool_call TypeError after filter: %s — retry without kwargs", e)
            try:
                return func(*args, **{})
            except Exception as e2:
                log.warning("trace_tool_call final fallback failed: %s", e2)
                return None

    _orig = trace

    def _shim(*args, **kwargs):
        log.debug("trace_tool_call shim: args=%s kwargs=%s", args, kwargs)
        return _call_with_filtered_kwargs(_orig, *args, **kwargs)

    _shim.__ank_patched__ = True
    _fn.trace_tool_call = _shim
    log.info("✅ patch_adk_tracing: flexible shim installed on functions.trace_tool_call")

_install()
