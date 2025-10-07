# test/patch_genai_live_ws.py
import logging
import inspect

log = logging.getLogger(__name__)

def _target_supports_param(param_name: str) -> bool:
    """
    Detects whether the installed `websockets.connect` supports a given kwarg.
    We only check the public signature; no calls are made.
    """
    try:
        import websockets
        sig = inspect.signature(websockets.connect)
        return param_name in sig.parameters
    except Exception as e:
        log.warning("Cannot introspect websockets.connect signature: %s", e)
        return False


try:
    from google.genai import live as _genai_live
    _orig_ws_connect = _genai_live.ws_connect

    def _patched_ws_connect(*args, **kwargs):
        """
        google.genai.live sometimes calls:
          - ws_connect(..., additional_headers=hdrs)
          - ws_connect(..., extra_headers=hdrs)
        Different websockets versions accept different kwarg names.
        We convert ONLY on the boundary here.
        """
        has_extra = "extra_headers" in kwargs
        has_additional = "additional_headers" in kwargs


        supports_extra = _target_supports_param("extra_headers")
        supports_additional = _target_supports_param("additional_headers")


        if has_extra and not supports_extra and supports_additional and not has_additional:

            kwargs["additional_headers"] = kwargs.pop("extra_headers")
            log.info("🔁 Converted ws_connect kwarg: extra_headers -> additional_headers")
        elif has_additional and not supports_additional and supports_extra and not has_extra:

            kwargs["extra_headers"] = kwargs.pop("additional_headers")
            log.info("🔁 Converted ws_connect kwarg: additional_headers -> extra_headers")
        elif has_extra and has_additional:

            log.debug("Both header kwargs present; leaving as-is.")
        else:

            pass

        return _orig_ws_connect(*args, **kwargs)  

    _genai_live.ws_connect = _patched_ws_connect
    log.info("✅ Patched google.genai.live.ws_connect successfully")
except Exception as e:
    log.warning("Failed to patch google.genai.live.ws_connect: %s", e)
