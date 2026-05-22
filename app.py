"""
Curve Fitting Flask Application
================================
Exponential and power-law regression via log-linearised least squares.

Endpoints
---------
GET  /            → Single-page application
POST /calculate   → Fit curve, return coefficients + diagnostics
GET  /health      → Liveness check

Author  : Klent Adrian B.
Python  : 3.8+
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import asdict, dataclass
from typing import Any

from flask import Flask, Response, jsonify, render_template, request


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False   # preserve key order in responses



class FitError(ValueError):
    """Raised when curve fitting cannot be completed due to invalid data."""


class ValidationError(ValueError):
    """Raised when the request payload fails schema or domain validation."""



@dataclass
class FitResult:
    """Structured container for a single curve-fit outcome."""

    fit_type:    str
    equation:    str
    A:           float
    b:           float
    r_squared:   float
    y_predicted: list[float]

    # Derived diagnostics populated after construction
    ss_res:      float = 0.0
    ss_tot:      float = 0.0
    rmse:        float = 0.0
    mae:         float = 0.0
    n:           int   = 0

    def to_response(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict, including a success flag."""
        d = asdict(self)
        d["success"] = True
        return d



def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _linear_least_squares(x: list[float], y: list[float]) -> tuple[float, float]:
    """
    Fit the line  Y = m·X + c  using the two-variable normal equations.

    Parameters
    ----------
    x, y : equal-length lists of floats (already transformed if needed)

    Returns
    -------
    (slope m, intercept c)

    Raises
    ------
    FitError if all X values are identical (degenerate system).
    """
    n     = len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    sum_xx = sum(xi * xi for xi in x)
    sum_xy = sum(xi * yi for xi, yi in zip(x, y))

    denom = n * sum_xx - sum_x ** 2
    if denom == 0:
        raise FitError(
            "All transformed X values are identical — "
            "the normal equations have no unique solution."
        )

    m = (n * sum_xy - sum_x * sum_y) / denom
    c = (sum_y - m * sum_x) / n
    return m, c


def _r_squared(y_actual: list[float], y_predicted: list[float]) -> float:
    """Coefficient of determination R² in the original (untransformed) space."""
    y_bar  = _mean(y_actual)
    ss_tot = sum((yi - y_bar) ** 2 for yi in y_actual)
    ss_res = sum((yi - yp) ** 2 for yi, yp in zip(y_actual, y_predicted))
    return 1.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot


def _diagnostics(
    y_actual: list[float],
    y_predicted: list[float],
) -> dict[str, float]:
    """
    Compute supplementary fit diagnostics in the original space.

    Returns a dict with keys: ss_res, ss_tot, rmse, mae.
    """
    n      = len(y_actual)
    y_bar  = _mean(y_actual)
    ss_tot = sum((yi - y_bar) ** 2 for yi in y_actual)
    ss_res = sum((yi - yp) ** 2 for yi, yp in zip(y_actual, y_predicted))
    rmse   = math.sqrt(ss_res / n)
    mae    = sum(abs(yi - yp) for yi, yp in zip(y_actual, y_predicted)) / n

    return {
        "ss_res": round(ss_res, 8),
        "ss_tot": round(ss_tot, 8),
        "rmse":   round(rmse,   8),
        "mae":    round(mae,    8),
    }



def fit_exponential(x: list[float], y: list[float]) -> FitResult:
    """
    Fit  y = A · e^(b·x)  via linearisation  ln(y) = ln(A) + b·x.

    The model is fitted by applying ordinary least squares to the
    transformed pairs  (xᵢ, ln yᵢ).  The prefactor A = e^(intercept)
    is recovered by back-transformation.

    Constraints
    -----------
    All yᵢ must be strictly positive (ln is undefined otherwise).
    """
    if any(yi <= 0 for yi in y):
        raise FitError(
            "Exponential fitting requires all Y values to be strictly positive "
            "(ln y is undefined for y ≤ 0)."
        )

    ln_y      = [math.log(yi) for yi in y]
    b, ln_A   = _linear_least_squares(x, ln_y)
    A         = math.exp(ln_A)
    y_pred    = [A * math.exp(b * xi) for xi in x]
    r2        = _r_squared(y, y_pred)
    diag      = _diagnostics(y, y_pred)

    sign  = "+" if b >= 0 else ""
    equation = f"y = {A:.4f} · e^({sign}{b:.4f}x)"

    result = FitResult(
        fit_type    = "exponential",
        equation    = equation,
        A           = round(A,  6),
        b           = round(b,  6),
        r_squared   = round(r2, 6),
        y_predicted = [round(v, 6) for v in y_pred],
        n           = len(x),
        **diag,
    )
    return result


def fit_power(x: list[float], y: list[float]) -> FitResult:
    """
    Fit  y = A · x^b  via linearisation  ln(y) = ln(A) + b · ln(x).

    The model is fitted by applying ordinary least squares to the
    transformed pairs  (ln xᵢ, ln yᵢ).  The prefactor A = e^(intercept)
    is recovered by back-transformation.

    Constraints
    -----------
    All xᵢ and yᵢ must be strictly positive.
    """
    bad_x = [xi for xi in x if xi <= 0]
    bad_y = [yi for yi in y if yi <= 0]

    if bad_x:
        raise FitError(
            f"Power-law fitting requires all X values to be strictly positive. "
            f"Non-positive values found: {bad_x}."
        )
    if bad_y:
        raise FitError(
            f"Power-law fitting requires all Y values to be strictly positive. "
            f"Non-positive values found: {bad_y}."
        )

    ln_x      = [math.log(xi) for xi in x]
    ln_y      = [math.log(yi) for yi in y]
    b, ln_A   = _linear_least_squares(ln_x, ln_y)
    A         = math.exp(ln_A)
    y_pred    = [A * (xi ** b) for xi in x]
    r2        = _r_squared(y, y_pred)
    diag      = _diagnostics(y, y_pred)

    equation = f"y = {A:.4f} · x^{b:.4f}"

    result = FitResult(
        fit_type    = "power",
        equation    = equation,
        A           = round(A,  6),
        b           = round(b,  6),
        r_squared   = round(r2, 6),
        y_predicted = [round(v, 6) for v in y_pred],
        n           = len(x),
        **diag,
    )
    return result


# Strategy registry — extend here when adding new model types.
_FIT_REGISTRY: dict[str, Any] = {
    "exponential": fit_exponential,
    "power":       fit_power,
}



_MAX_POINTS = 500   # reasonable upper bound to prevent abuse

def _validate_payload(data: dict) -> tuple[list[float], list[float], str]:
    """
    Parse and validate the incoming JSON payload.

    Returns
    -------
    (x, y, fit_type)  ready for the chosen fitting strategy.

    Raises
    ------
    ValidationError with a human-readable message on any failure.
    """
    if not isinstance(data, dict):
        raise ValidationError("Request body must be a JSON object.")

    fit_type = data.get("type", "exponential")
    if fit_type not in _FIT_REGISTRY:
        valid = ", ".join(f"'{k}'" for k in _FIT_REGISTRY)
        raise ValidationError(
            f"Unknown fit type '{fit_type}'. Valid options are: {valid}."
        )

    x_raw = data.get("x")
    y_raw = data.get("y")

    if x_raw is None or y_raw is None:
        raise ValidationError("Both 'x' and 'y' arrays are required.")

    if not isinstance(x_raw, list) or not isinstance(y_raw, list):
        raise ValidationError("'x' and 'y' must be JSON arrays.")

    if len(x_raw) != len(y_raw):
        raise ValidationError(
            f"'x' has {len(x_raw)} values but 'y' has {len(y_raw)}. "
            "Both arrays must be the same length."
        )

    if len(x_raw) < 3:
        raise ValidationError(
            f"At least 3 data points are required; received {len(x_raw)}."
        )

    if len(x_raw) > _MAX_POINTS:
        raise ValidationError(
            f"Maximum {_MAX_POINTS} data points allowed; received {len(x_raw)}."
        )

    try:
        x = [float(v) for v in x_raw]
        y = [float(v) for v in y_raw]
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"Non-numeric value in data arrays: {exc}") from exc

    if any(not math.isfinite(v) for v in x + y):
        raise ValidationError(
            "Data arrays must not contain NaN or Infinity values."
        )

    return x, y, fit_type



@app.errorhandler(404)
def not_found(_: Any) -> Response:
    return jsonify({"error": "Endpoint not found."}), 404


@app.errorhandler(405)
def method_not_allowed(_: Any) -> Response:
    return jsonify({"error": "HTTP method not allowed on this endpoint."}), 405



@app.route("/")
def index() -> str:
    """Render the single-page application."""
    return render_template("index.html")


@app.route("/health")
def health() -> Response:
    """
    GET /health — lightweight liveness probe.

    Returns 200 with uptime and app metadata; useful for monitoring.
    """
    return jsonify({
        "status":  "ok",
        "version": "1.0.0",
        "models":  list(_FIT_REGISTRY.keys()),
    })


@app.route("/calculate", methods=["POST"])
def calculate() -> Response:
    """
    POST /calculate
    ---------------
    Accepts a JSON body and returns a fitted curve with diagnostics.

    Request body
    ~~~~~~~~~~~~
    {
        "x":    [float, ...],       // independent variable values
        "y":    [float, ...],       // observed dependent values
        "type": "exponential"|"power"
    }

    Success response  (HTTP 200)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    {
        "success":     true,
        "fit_type":    "exponential",
        "equation":    "y = 1.2345 · e^(+0.5755x)",
        "A":           1.2345,
        "b":           0.5755,
        "r_squared":   0.9999,
        "y_predicted": [2.1, 5.8, ...],
        "ss_res":      0.00049,
        "ss_tot":      562157.0,
        "rmse":        0.00314,
        "mae":         0.00271,
        "n":           5
    }

    Error response  (HTTP 400 / 500)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    { "error": "Human-readable message." }
    """
    t_start = time.perf_counter()

    try:
        payload = request.get_json(force=True, silent=True)
        if payload is None:
            return jsonify({"error": "Request body must be valid JSON."}), 400

        x, y, fit_type = _validate_payload(payload)

        fit_fn = _FIT_REGISTRY[fit_type]
        result = fit_fn(x, y)

        elapsed_ms = round((time.perf_counter() - t_start) * 1000, 2)
        logger.info(
            "Fit OK  type=%-12s  n=%d  R²=%.6f  elapsed=%sms",
            fit_type, len(x), result.r_squared, elapsed_ms,
        )

        response_body = result.to_response()
        response_body["elapsed_ms"] = elapsed_ms
        return jsonify(response_body)

    except ValidationError as exc:
        logger.warning("Validation error — %s", exc)
        return jsonify({"error": str(exc)}), 400

    except FitError as exc:
        logger.warning("Fit error — %s", exc)
        return jsonify({"error": str(exc)}), 422

    except Exception:
        logger.exception("Unexpected error in /calculate")
        return jsonify({"error": "An unexpected server error occurred."}), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)