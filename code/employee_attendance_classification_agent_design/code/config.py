
import os
from dotenv import load_dotenv
from typing import Optional, Dict, Any

# Load environment variables from .env file if present
load_dotenv()

class ConfigError(Exception):
    """Custom exception for configuration errors."""
    pass

class Config:
    # --- API Base URL ---
    BASE_URL = "/api"

    # --- API Endpoints ---
    ENDPOINTS = {
        "checkin_logs": "/api/checkin-logs",
        "leave_data": "/api/leave-data",
        "shift_rules": "/api/shift-rules",
        "holiday_calendar": "/api/holiday-calendar"
    }

    # --- API Key Management ---
    @staticmethod
    def get_hris_api_token() -> str:
        token = os.getenv("HRIS_API_BEARER_TOKEN")
        if not token:
            raise ConfigError("Missing HRIS_API_BEARER_TOKEN environment variable for HRIS API authentication.")
        return token

    @staticmethod
    def get_azure_openai_key() -> str:
        key = os.getenv("AZURE_OPENAI_API_KEY")
        if not key:
            raise ConfigError("Missing AZURE_OPENAI_API_KEY environment variable for Azure OpenAI.")
        return key

    @staticmethod
    def get_azure_openai_endpoint() -> str:
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        if not endpoint:
            raise ConfigError("Missing AZURE_OPENAI_ENDPOINT environment variable for Azure OpenAI.")
        return endpoint

    @staticmethod
    def get_azure_openai_deployment() -> str:
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        if not deployment:
            raise ConfigError("Missing AZURE_OPENAI_DEPLOYMENT environment variable for Azure OpenAI.")
        return deployment

    # --- LLM Configuration ---
    LLM_CONFIG = {
        "provider": "azure",
        "model": "gpt-4.1",
        "temperature": 0.7,
        "max_tokens": 2000,
        "system_prompt": (
            "You are an Employee Attendance Classification Agent. Your role is to classify each employee's daily attendance status as Present, Late Present, Half Day, Leave, Absent, or Holiday. "
            "Use check-in logs, leave data, shift rules, and holiday calendars. Always follow strict policy order: Holiday > Leave > Present > Late Present > Half Day > Absent. "
            "Validate all data sources before making a decision. Log every classification for audit purposes. Communicate clearly and professionally."
        ),
        "user_prompt_template": (
            "Please provide the date and, if applicable, the employee ID or department for attendance classification. The agent will process and return the attendance status for each employee."
        ),
        "few_shot_examples": [
            "Classify attendance for employee 12345 on 2024-06-10.",
            "What is the attendance status for all employees on 2024-06-12?"
        ]
    }

    # --- Domain-specific Settings ---
    POLICY_ORDER = [
        "Holiday",
        "Leave",
        "Present",
        "Late Present",
        "Half Day",
        "Absent"
    ]
    REQUIRED_CONFIG_KEYS = [
        "shift_rules",
        "holiday_calendar"
    ]
    ERROR_CODES = [
        "ATTENDANCE_DATA_MISSING",
        "INVALID_SHIFT_RULES"
    ]

    # --- Validation and Error Handling ---
    @staticmethod
    def validate_all():
        errors = []
        try:
            Config.get_hris_api_token()
        except ConfigError as e:
            errors.append(str(e))
        try:
            Config.get_azure_openai_key()
        except ConfigError as e:
            errors.append(str(e))
        try:
            Config.get_azure_openai_endpoint()
        except ConfigError as e:
            errors.append(str(e))
        try:
            Config.get_azure_openai_deployment()
        except ConfigError as e:
            errors.append(str(e))
        if errors:
            raise ConfigError("Configuration validation failed: " + "; ".join(errors))

    # --- Default Values and Fallbacks ---
    @staticmethod
    def get_default_shift_id() -> str:
        return "default"

    @staticmethod
    def get_default_location() -> str:
        return "default"

    @staticmethod
    def get_default_year() -> int:
        from datetime import datetime
        return datetime.now().year

    @staticmethod
    def get_headers() -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {Config.get_hris_api_token()}",
            "Content-Type": "application/json"
        }

    @staticmethod
    def get_llm_settings() -> Dict[str, Any]:
        return Config.LLM_CONFIG

# Validate config at import (fail fast)
try:
    Config.validate_all()
except ConfigError as e:
    # Comment out the next line if you want to allow import without all keys present
    # raise
    print(f"WARNING: {e}")

# Usage example (in other modules):
# from config import Config
# token = Config.get_hris_api_token()
# endpoint = Config.ENDPOINTS["checkin_logs"]
# headers = Config.get_headers()
# llm_settings = Config.get_llm_settings()
