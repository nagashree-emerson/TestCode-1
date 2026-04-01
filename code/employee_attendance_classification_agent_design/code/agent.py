try:
    from observability.observability_wrapper import (
        trace_agent, trace_step, trace_step_sync, trace_model_call, trace_tool_call,
    )
except ImportError:  # observability module not available (e.g. isolated test env)
    from contextlib import contextmanager as _obs_cm, asynccontextmanager as _obs_acm
    def trace_agent(*_a, **_kw):  # type: ignore[misc]
        def _deco(fn): return fn
        return _deco
    class _ObsHandle:
        output_summary = None
        def capture(self, *a, **kw): pass
    @_obs_acm
    async def trace_step(*_a, **_kw):  # type: ignore[misc]
        yield _ObsHandle()
    @_obs_cm
    def trace_step_sync(*_a, **_kw):  # type: ignore[misc]
        yield _ObsHandle()
    def trace_model_call(*_a, **_kw): pass  # type: ignore[misc]
    def trace_tool_call(*_a, **_kw): pass  # type: ignore[misc]

from modules.guardrails.content_safety_decorator import with_content_safety

GUARDRAILS_CONFIG = {'check_credentials_output': True,
 'check_jailbreak': True,
 'check_output': True,
 'check_pii_input': False,
 'check_toxic_code_output': True,
 'check_toxicity': True,
 'content_safety_enabled': True,
 'content_safety_severity_threshold': 3,
 'runtime_enabled': True,
 'sanitize_pii': False}


import os
import sys
import asyncio
import logging
import time as _time
from typing import Optional, Dict, Any, List, Union
from functools import lru_cache

from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator, ValidationError, Field, model_validator
from dotenv import load_dotenv
import httpx
from cachetools import TTLCache
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Observability wrappers are injected by runtime (do not import trace_agent, trace_step, etc.)

# Load environment variables from .env if present
load_dotenv()

# --- Configuration Management ---

class Config:
    """
    Configuration loader for API keys, endpoints, and LLM settings.
    """
    # KB API base path
    BASE_URL = "/api"

    @staticmethod
    def get_hris_api_base_url() -> str:
        """
        Returns the base path for all HRIS KB APIs.
        """
        return Config.BASE_URL

    @staticmethod
    def get_authorization_header() -> Optional[str]:
        """
        Returns the OAuth2 Bearer token for HRIS API calls.
        """
        return os.getenv("HRIS_API_BEARER_TOKEN")

    @staticmethod
    def get_content_type() -> str:
        return "application/json"

    @staticmethod
    def get_azure_openai_key() -> Optional[str]:
        return os.getenv("AZURE_OPENAI_API_KEY")

    @staticmethod
    def get_azure_openai_endpoint() -> Optional[str]:
        return os.getenv("AZURE_OPENAI_ENDPOINT")

    @staticmethod
    def get_azure_openai_deployment() -> Optional[str]:
        return os.getenv("AZURE_OPENAI_DEPLOYMENT")

    @staticmethod
    @trace_agent(agent_name='Employee Attendance Classification Agent')
    @with_content_safety(config=GUARDRAILS_CONFIG)
    def validate_llm_config() -> None:
        """
        Validates that all required LLM config is present.
        """
        missing = []
        if not Config.get_azure_openai_key():
            missing.append("AZURE_OPENAI_API_KEY")
        if not Config.get_azure_openai_endpoint():
            missing.append("AZURE_OPENAI_ENDPOINT")
        if not Config.get_azure_openai_deployment():
            missing.append("AZURE_OPENAI_DEPLOYMENT")
        if missing:
            raise RuntimeError(f"Missing required LLM config: {', '.join(missing)}")

    @staticmethod
    def validate_hris_config() -> None:
        if not Config.get_authorization_header():
            raise RuntimeError("Missing HRIS_API_BEARER_TOKEN for HRIS API authentication.")

# --- Models ---

class AttendanceRequest(BaseModel):
    employee_id: Optional[str] = Field(None, description="Employee ID to classify attendance for")
    date: str = Field(..., description="Date in YYYY-MM-DD format")
    department: Optional[str] = Field(None, description="Department (optional)")

    @field_validator("date")
    @classmethod
    def validate_date(cls, v):
        import re
        if not v or not isinstance(v, str):
            raise ValueError("Date must be a non-empty string in YYYY-MM-DD format.")
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", v):
            raise ValueError("Date must be in YYYY-MM-DD format.")
        return v

    @field_validator("employee_id")
    @classmethod
    def validate_employee_id(cls, v):
        if v is not None and not isinstance(v, str):
            raise ValueError("Employee ID must be a string.")
        return v

    @field_validator("department")
    @classmethod
    def validate_department(cls, v):
        if v is not None and not isinstance(v, str):
            raise ValueError("Department must be a string.")
        return v

    @model_validator(mode="after")
    def check_at_least_one_identifier(self):
        if not self.employee_id and not self.department:
            raise ValueError("At least one of employee_id or department must be provided.")
        return self

class AttendanceStatusResult(BaseModel):
    employee_id: str
    date: str
    status: str
    reason: Optional[str] = None

class ErrorResponse(BaseModel):
    success: bool = False
    error_type: str
    error_message: str
    tips: Optional[str] = None

class AuditLogEntry(BaseModel):
    event: str
    employee_id: str
    date: str
    status: str
    log_entry_id: Optional[str] = None

# --- Utility: Error Handler ---

class ErrorHandler:
    """
    Centralized error handling, retry logic, and fallback.
    """
    def __init__(self):
        self.retry_attempts = 3

    def handle_error(self, error_code: str, context: Dict[str, Any]) -> None:
        """
        Handles errors, applies retry logic, and triggers fallback.
        """
        logger.error(f"Error occurred: {error_code} | Context: {context}")
        # Fallback: Could notify HR, mark as pending, etc.
        # For now, just log and pass.
        pass

# --- Utility: Cache Manager ---

class CacheManager:
    """
    In-memory cache for shift rules and holiday calendars.
    """
    def __init__(self):
        # TTL: 1 day (86400s) for shift rules and holiday calendars
        self.shift_rules_cache = TTLCache(maxsize=1000, ttl=86400)
        self.holiday_calendar_cache = TTLCache(maxsize=100, ttl=86400)

    def get_cached_shift_rules(self, shift_id: str) -> Optional[Dict[str, Any]]:
        return self.shift_rules_cache.get(shift_id)

    def set_cached_shift_rules(self, shift_id: str, value: Dict[str, Any]):
        self.shift_rules_cache[shift_id] = value

    def get_cached_holiday_calendar(self, location: str, year: int) -> Optional[Dict[str, Any]]:
        key = f"{location}:{year}"
        return self.holiday_calendar_cache.get(key)

    def set_cached_holiday_calendar(self, location: str, year: int, value: Dict[str, Any]):
        key = f"{location}:{year}"
        self.holiday_calendar_cache[key] = value

    def set_cache(self, key: str, value: Any):
        # Generic cache setter (not used for shift/holiday, but for extensibility)
        self.shift_rules_cache[key] = value

# --- Integration Layer ---

class IntegrationLayer:
    """
    Handles all API integrations for check-in logs, leave data, shift rules, and holiday calendars.
    """
    def __init__(self, cache_manager: CacheManager, error_handler: ErrorHandler):
        self.cache_manager = cache_manager
        self.error_handler = error_handler

    def _get_headers(self) -> Dict[str, str]:
        token = Config.get_authorization_header()
        if not token:
            raise RuntimeError("HRIS_API_BEARER_TOKEN not configured.")
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": Config.get_content_type()
        }

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=1, max=4),
        retry=retry_if_exception_type(httpx.HTTPError)
    )
    async def get_checkin_logs(self, employee_id: str, date: str) -> Dict[str, Any]:
        """
        Fetches check-in logs for an employee and date.
        """
        url = f"{Config.get_hris_api_base_url()}/checkin-logs"
        params = {"employee_id": employee_id, "date": date}
        headers = self._get_headers()
        _t0 = _time.time()
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url, headers=headers, params=params, timeout=8)
                response.raise_for_status()
                data = response.json()
                try:
                    trace_tool_call(
                        tool_name="get_checkin_logs",
                        latency_ms=int((_time.time() - _t0) * 1000),
                        output=str(data)[:200], status="success"
                    )
                except Exception:
                    pass
                return data
            except Exception as e:
                self.error_handler.handle_error("ATTENDANCE_DATA_MISSING", {"api": "checkin-logs", "error": str(e)})
                raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=1, max=4),
        retry=retry_if_exception_type(httpx.HTTPError)
    )
    async def get_leave_data(self, employee_id: str, date: str) -> Dict[str, Any]:
        """
        Fetches leave data for an employee and date.
        """
        url = f"{Config.get_hris_api_base_url()}/leave-data"
        params = {"employee_id": employee_id, "date": date}
        headers = self._get_headers()
        _t0 = _time.time()
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url, headers=headers, params=params, timeout=8)
                response.raise_for_status()
                data = response.json()
                try:
                    trace_tool_call(
                        tool_name="get_leave_data",
                        latency_ms=int((_time.time() - _t0) * 1000),
                        output=str(data)[:200], status="success"
                    )
                except Exception:
                    pass
                return data
            except Exception as e:
                self.error_handler.handle_error("ATTENDANCE_DATA_MISSING", {"api": "leave-data", "error": str(e)})
                raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=1, max=4),
        retry=retry_if_exception_type(httpx.HTTPError)
    )
    async def get_shift_rules(self, shift_id: str) -> Dict[str, Any]:
        """
        Fetches shift rules for a given shift_id, with caching.
        """
        cached = self.cache_manager.get_cached_shift_rules(shift_id)
        if cached:
            return cached
        url = f"{Config.get_hris_api_base_url()}/shift-rules"
        params = {"shift_id": shift_id}
        headers = self._get_headers()
        _t0 = _time.time()
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url, headers=headers, params=params, timeout=8)
                response.raise_for_status()
                data = response.json()
                self.cache_manager.set_cached_shift_rules(shift_id, data)
                try:
                    trace_tool_call(
                        tool_name="get_shift_rules",
                        latency_ms=int((_time.time() - _t0) * 1000),
                        output=str(data)[:200], status="success"
                    )
                except Exception:
                    pass
                return data
            except Exception as e:
                self.error_handler.handle_error("INVALID_SHIFT_RULES", {"api": "shift-rules", "error": str(e)})
                raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=1, max=4),
        retry=retry_if_exception_type(httpx.HTTPError)
    )
    async def get_holiday_calendar(self, location: str, year: int) -> Dict[str, Any]:
        """
        Fetches holiday calendar for a location and year, with caching.
        """
        cached = self.cache_manager.get_cached_holiday_calendar(location, year)
        if cached:
            return cached
        url = f"{Config.get_hris_api_base_url()}/holiday-calendar"
        params = {"location": location, "year": year}
        headers = self._get_headers()
        _t0 = _time.time()
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url, headers=headers, params=params, timeout=8)
                response.raise_for_status()
                data = response.json()
                self.cache_manager.set_cached_holiday_calendar(location, year, data)
                try:
                    trace_tool_call(
                        tool_name="get_holiday_calendar",
                        latency_ms=int((_time.time() - _t0) * 1000),
                        output=str(data)[:200], status="success"
                    )
                except Exception:
                    pass
                return data
            except Exception as e:
                self.error_handler.handle_error("ATTENDANCE_DATA_MISSING", {"api": "holiday-calendar", "error": str(e)})
                raise

# --- Data Validator ---

class DataValidator:
    """
    Validates completeness and integrity of all required data sources before processing.
    """
    def __init__(self, integration_layer: IntegrationLayer):
        self.integration_layer = integration_layer

    async def validate_sources(self, employee_id: str, date: str) -> bool:
        """
        Validates that all required data sources are available and complete.
        """
        async with trace_step(
            "validate_sources", step_type="parse",
            decision_summary="Validate all required data sources for attendance classification",
            output_fn=lambda r: f"valid={r}"
        ) as step:
            try:
                checkin = await self.integration_layer.get_checkin_logs(employee_id, date)
                leave = await self.integration_layer.get_leave_data(employee_id, date)
                # For demo, assume shift_id and location are available from checkin or leave data
                shift_id = checkin.get("shift_id") or leave.get("shift_id") or "default"
                location = checkin.get("location") or leave.get("location") or "default"
                year = int(date[:4])
                shift_rules = await self.integration_layer.get_shift_rules(shift_id)
                holiday_calendar = await self.integration_layer.get_holiday_calendar(location, year)
                valid = all([checkin, leave, shift_rules, holiday_calendar])
                step.capture(valid)
                return valid
            except Exception as e:
                logger.error(f"Data validation failed: {e}")
                step.capture(False)
                return False

# --- Policy Engine ---

class PolicyEngine:
    """
    Enforces strict policy order and executes rule sets and decision tables.
    """
    def __init__(self):
        pass

    def apply_policy(self, data: Dict[str, Any]) -> str:
        """
        Applies the attendance classification policy rules in strict order.
        Returns status string.
        """
        async def _inner():
            async with trace_step(
                "apply_policy", step_type="process",
                decision_summary="Apply attendance policy rules in strict order",
                output_fn=lambda r: f"status={r}"
            ) as step:
                status = self._apply_policy_sync(data)
                step.capture(status)
                return status
        # For compatibility, run the sync logic in a coroutine
        return self._apply_policy_sync(data)

    def _apply_policy_sync(self, data: Dict[str, Any]) -> str:
        # Extract required fields
        is_holiday = data.get("is_holiday", False)
        is_on_leave = data.get("is_on_leave", False)
        check_in_time = data.get("check_in_time")
        shift_rules = data.get("shift_rules", {})
        # Policy order: Holiday > Leave > Present > Late Present > Half Day > Absent
        if is_holiday:
            return "Holiday"
        if is_on_leave:
            return "Leave"
        if check_in_time is None:
            return "Absent"
        # Parse shift rules
        shift_start = shift_rules.get("start_time")  # e.g., "09:00"
        grace_period = shift_rules.get("grace_period_minutes", 0)
        half_day_limit = shift_rules.get("half_day_limit_minutes", 0)
        # Parse check-in time
        try:
            from datetime import datetime, timedelta
            fmt = "%H:%M"
            check_in_dt = datetime.strptime(check_in_time, fmt)
            shift_start_dt = datetime.strptime(shift_start, fmt)
            delta_minutes = int((check_in_dt - shift_start_dt).total_seconds() / 60)
            if delta_minutes <= 0:
                return "Present"
            elif 0 < delta_minutes <= grace_period:
                return "Late Present"
            elif grace_period < delta_minutes <= half_day_limit:
                return "Half Day"
            else:
                return "Absent"
        except Exception as e:
            logger.error(f"PolicyEngine: Error parsing times: {e}")
            return "Absent"

# --- Audit Logger ---

class AuditLogger:
    """
    Logs all classification decisions for compliance and traceability.
    """
    def __init__(self):
        self.audit_log: List[AuditLogEntry] = []

    def log_decision(self, event: str, employee_id: str, date: str, status: str) -> str:
        """
        Logs the classification decision for audit purposes.
        """
        log_entry_id = f"log_{len(self.audit_log)+1}_{int(_time.time())}"
        entry = AuditLogEntry(
            event=event,
            employee_id=employee_id,
            date=date,
            status=status,
            log_entry_id=log_entry_id
        )
        self.audit_log.append(entry)
        try:
            logger.info(f"Audit log: {entry.model_dump_json()}")
        except Exception:
            logger.info(f"Audit log: {entry}")
        return log_entry_id

# --- Attendance Classifier ---

class AttendanceClassifier:
    """
    Implements policy-based attendance classification logic.
    """
    def __init__(
        self,
        data_validator: DataValidator,
        policy_engine: PolicyEngine,
        integration_layer: IntegrationLayer,
        audit_logger: AuditLogger,
        error_handler: ErrorHandler
    ):
        self.data_validator = data_validator
        self.policy_engine = policy_engine
        self.integration_layer = integration_layer
        self.audit_logger = audit_logger
        self.error_handler = error_handler

    async def classify_attendance(self, employee_id: str, date: str) -> AttendanceStatusResult:
        """
        Classifies attendance for a given employee and date.
        """
        async with trace_step(
            "classify_attendance", step_type="plan",
            decision_summary="Classify attendance for employee and date",
            output_fn=lambda r: f"status={r.status if r else '?'}"
        ) as step:
            try:
                # Validate data sources
                valid = await self.data_validator.validate_sources(employee_id, date)
                if not valid:
                    self.error_handler.handle_error("ATTENDANCE_DATA_MISSING", {"employee_id": employee_id, "date": date})
                    return AttendanceStatusResult(
                        employee_id=employee_id,
                        date=date,
                        status="Pending",
                        reason="Required data missing. Marked for manual review."
                    )
                # Fetch data
                checkin = await self.integration_layer.get_checkin_logs(employee_id, date)
                leave = await self.integration_layer.get_leave_data(employee_id, date)
                shift_id = checkin.get("shift_id") or leave.get("shift_id") or "default"
                location = checkin.get("location") or leave.get("location") or "default"
                year = int(date[:4])
                shift_rules = await self.integration_layer.get_shift_rules(shift_id)
                holiday_calendar = await self.integration_layer.get_holiday_calendar(location, year)
                # Prepare policy input
                is_holiday = False
                for h in holiday_calendar.get("holidays", []):
                    if h.get("date") == date:
                        is_holiday = True
                        break
                is_on_leave = leave.get("is_on_leave", False)
                check_in_time = checkin.get("check_in_time")
                policy_data = {
                    "is_holiday": is_holiday,
                    "is_on_leave": is_on_leave,
                    "check_in_time": check_in_time,
                    "shift_rules": shift_rules
                }
                # Apply policy
                status = self.policy_engine.apply_policy(policy_data)
                # Audit log
                event = f"{status} classified"
                log_id = self.audit_logger.log_decision(event, employee_id, date, status)
                result = AttendanceStatusResult(
                    employee_id=employee_id,
                    date=date,
                    status=status
                )
                step.capture(result)
                return result
            except Exception as e:
                logger.error(f"Attendance classification failed: {e}")
                self.error_handler.handle_error("ATTENDANCE_CLASSIFICATION_ERROR", {"employee_id": employee_id, "date": date, "error": str(e)})
                return AttendanceStatusResult(
                    employee_id=employee_id,
                    date=date,
                    status="Pending",
                    reason=f"Error during classification: {e}"
                )

# --- LLM Integration (Azure OpenAI) ---

class LLMInteractionManager:
    """
    Handles LLM calls for policy explanation, edge-case reasoning, and response generation.
    """
    def __init__(self):
        self.model = "gpt-4.1"
        self.temperature = 0.7
        self.max_tokens = 2000
        self.system_prompt = (
            "You are an Employee Attendance Classification Agent. Your role is to classify each employee's daily attendance status as Present, Late Present, Half Day, Leave, Absent, or Holiday. "
            "Use check-in logs, leave data, shift rules, and holiday calendars. Always follow strict policy order: Holiday > Leave > Present > Late Present > Half Day > Absent. "
            "Validate all data sources before making a decision. Log every classification for audit purposes. Communicate clearly and professionally."
        )
        self.user_prompt_template = (
            "Please provide the date and, if applicable, the employee ID or department for attendance classification. "
            "The agent will process and return the attendance status for each employee."
        )

    def _get_client(self):
        import openai
        api_key = Config.get_azure_openai_key()
        endpoint = Config.get_azure_openai_endpoint()
        deployment = Config.get_azure_openai_deployment()
        if not api_key or not endpoint or not deployment:
            raise ValueError("Azure OpenAI configuration missing. Please set AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, and AZURE_OPENAI_DEPLOYMENT.")
        return openai.AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            azure_deployment=deployment
        )

    async def explain_policy_decision(self, attendance_result: AttendanceStatusResult, context: Dict[str, Any]) -> str:
        """
        Uses LLM to explain the policy decision for the given attendance result.
        """
        async with trace_step(
            "llm_explain_policy", step_type="llm_call",
            decision_summary="Call LLM to explain attendance policy decision",
            output_fn=lambda r: f"length={len(r) if r else 0}"
        ) as step:
            client = self._get_client()
            user_message = (
                f"Employee {attendance_result.employee_id} on {attendance_result.date} was classified as '{attendance_result.status}'. "
                f"Context: {context}. Please explain the policy reasoning for this classification."
            )
            _t0 = _time.time()
            try:
                response = await client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                content = response.choices[0].message.content
                try:
                    trace_model_call(
                        provider="azure",
                        model_name=self.model,
                        prompt_tokens=response.usage.prompt_tokens if hasattr(response, "usage") else None,
                        completion_tokens=response.usage.completion_tokens if hasattr(response, "usage") else None,
                        latency_ms=int((_time.time() - _t0) * 1000),
                        response_summary=content[:200] if content else ""
                    )
                except Exception:
                    pass
                step.capture(content)
                return content
            except Exception as e:
                logger.error(f"LLM explanation failed: {e}")
                step.capture("")
                return "Unable to generate policy explanation at this time."

# --- Main Agent ---

class AttendanceClassificationAgent:
    """
    Main agent class aggregating all supporting components.
    """
    def __init__(self):
        self.error_handler = ErrorHandler()
        self.cache_manager = CacheManager()
        self.integration_layer = IntegrationLayer(self.cache_manager, self.error_handler)
        self.data_validator = DataValidator(self.integration_layer)
        self.policy_engine = PolicyEngine()
        self.audit_logger = AuditLogger()
        self.attendance_classifier = AttendanceClassifier(
            self.data_validator,
            self.policy_engine,
            self.integration_layer,
            self.audit_logger,
            self.error_handler
        )
        self.llm_manager = LLMInteractionManager()

    async def classify(self, employee_id: str, date: str) -> AttendanceStatusResult:
        """
        Classifies attendance for a given employee and date.
        """
        return await self.attendance_classifier.classify_attendance(employee_id, date)

    async def explain(self, attendance_result: AttendanceStatusResult, context: Dict[str, Any]) -> str:
        """
        Uses LLM to explain the policy decision.
        """
        return await self.llm_manager.explain_policy_decision(attendance_result, context)

# --- FastAPI App ---

app = FastAPI(
    title="Employee Attendance Classification Agent",
    description="API for classifying employee attendance status according to strict HR policy.",
    version="1.0.0"
)

# CORS (allow all origins for demo; restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = AttendanceClassificationAgent()

# --- Exception Handlers ---

@app.exception_handler(ValidationError)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def validation_exception_handler(request: Request, exc: ValidationError):
    logger.warning(f"Validation error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            error_type="ValidationError",
            error_message=str(exc),
            tips="Check your input fields, ensure correct types and formats (e.g., date as YYYY-MM-DD)."
        ).model_dump()
    )

@app.exception_handler(HTTPException)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.warning(f"HTTPException: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error_type="HTTPException",
            error_message=exc.detail,
            tips="Check your request and try again."
        ).model_dump()
    )

@app.exception_handler(Exception)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error_type="InternalServerError",
            error_message="An unexpected error occurred.",
            tips="If you submitted JSON, check for missing commas, quotes, or brackets. Contact support if the issue persists."
        ).model_dump()
    )

# --- API Endpoints ---

@app.post("/classify", response_model=AttendanceStatusResult)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def classify_attendance_endpoint(request: Request):
    """
    Classify attendance for a given employee and date.
    """
    try:
        body = await request.json()
    except Exception as e:
        logger.warning(f"Malformed JSON: {e}")
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                error_type="MalformedJSON",
                error_message="Malformed JSON in request body.",
                tips="Check for missing commas, quotes, or brackets in your JSON."
            ).model_dump()
        )
    try:
        req = AttendanceRequest(**body)
    except ValidationError as ve:
        logger.warning(f"Validation error: {ve}")
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(
                error_type="ValidationError",
                error_message=str(ve),
                tips="Check your input fields, ensure correct types and formats (e.g., date as YYYY-MM-DD)."
            ).model_dump()
        )
    # Only support single employee_id for now
    if not req.employee_id:
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                error_type="MissingEmployeeID",
                error_message="employee_id is required for classification.",
                tips="Provide a valid employee_id in your request."
            ).model_dump()
        )
    async with trace_step(
        "api_classify_attendance", step_type="final",
        decision_summary="API endpoint for attendance classification",
        output_fn=lambda r: f"status={r.status if r else '?'}"
    ) as step:
        _obs_t0 = _time.time()
        result = await agent.classify(req.employee_id, req.date)
        try:
            trace_tool_call(
                tool_name='agent.classify',
                latency_ms=int((_time.time() - _obs_t0) * 1000),
                output=str(result)[:200] if result is not None else None,
                status="success",
            )
        except Exception:
            pass
        step.capture(result)
        return result

@app.post("/explain", response_model=Dict[str, str])
@with_content_safety(config=GUARDRAILS_CONFIG)
async def explain_policy_endpoint(request: Request):
    """
    Explain the policy decision for a given attendance result.
    """
    try:
        body = await request.json()
    except Exception as e:
        logger.warning(f"Malformed JSON: {e}")
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                error_type="MalformedJSON",
                error_message="Malformed JSON in request body.",
                tips="Check for missing commas, quotes, or brackets in your JSON."
            ).model_dump()
        )
    try:
        attendance_result = AttendanceStatusResult(**body.get("attendance_result", {}))
        context = body.get("context", {})
    except Exception as ve:
        logger.warning(f"Validation error: {ve}")
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(
                error_type="ValidationError",
                error_message=str(ve),
                tips="Check your input fields and structure."
            ).model_dump()
        )
    async with trace_step(
        "api_explain_policy", step_type="final",
        decision_summary="API endpoint for LLM policy explanation",
        output_fn=lambda r: f"length={len(r) if r else 0}"
    ) as step:
        explanation = await agent.explain(attendance_result, context)
        step.capture(explanation)
        return {"success": True, "explanation": explanation}

@app.get("/health")
async def health_check():
    return {"success": True, "status": "ok"}

# --- Logging Configuration ---

logger.remove()
logger.add(sys.stderr, level="INFO", format="{time} | {level} | {message}")

# --- Main Entrypoint ---



async def _run_with_eval_service():
    """Entrypoint: initialises observability then runs the agent."""
    import logging as _obs_log
    _obs_logger = _obs_log.getLogger(__name__)
    # ── 1. Observability DB schema ─────────────────────────────────────
    try:
        from observability.database.engine import create_obs_database_engine
        from observability.database.base import ObsBase
        import observability.database.models  # noqa: F401 – register ORM models
        _obs_engine = create_obs_database_engine()
        ObsBase.metadata.create_all(bind=_obs_engine, checkfirst=True)
    except Exception as _e:
        _obs_logger.warning('Observability DB init skipped: %s', _e)
    # ── 2. OpenTelemetry tracer ────────────────────────────────────────
    try:
        from observability.instrumentation import initialize_tracer
        initialize_tracer()
    except Exception as _e:
        _obs_logger.warning('Tracer init skipped: %s', _e)
    # ── 3. Evaluation background worker ───────────────────────────────
    _stop_eval = None
    try:
        from observability.evaluation_background_service import (
            start_evaluation_worker as _start_eval,
            stop_evaluation_worker as _stop_eval_fn,
        )
        await _start_eval()
        _stop_eval = _stop_eval_fn
    except Exception as _e:
        _obs_logger.warning('Evaluation worker start skipped: %s', _e)
    # ── 4. Run the agent ───────────────────────────────────────────────
    try:
        import uvicorn
        logger.info("Starting Employee Attendance Classification Agent...")
        uvicorn.run("agent:app", host="0.0.0.0", port=int(os.getenv("PORT", 8080)), reload=True)
        pass  # TODO: run your agent here
    finally:
        if _stop_eval is not None:
            try:
                await _stop_eval()
            except Exception:
                pass


if __name__ == "__main__":
    import asyncio as _asyncio
    _asyncio.run(_run_with_eval_service())