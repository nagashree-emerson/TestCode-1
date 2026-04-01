
import logging
import asyncio
from typing import Dict, Any, List, Optional

# Dynamic import for agent modules
import importlib

logger = logging.getLogger(__name__)

# Import agent classes from their respective folders
try:
    from employee_work_assignment_agent_design.agent import EmployeeWorkAssignmentAgent
except ImportError as e:
    logger.error("Failed to import EmployeeWorkAssignmentAgent: %s", e)
    raise

try:
    from employee_attendance_classification_agent_design.agent import AttendanceClassificationAgent
except ImportError as e:
    logger.error("Failed to import AttendanceClassificationAgent: %s", e)
    raise

class OrchestrationEngine:
    """
    Orchestrates the workflow:
      1. Calls EmployeeWorkAssignmentAgent.assign_tasks with input_data['tasks'] and input_data['employee_roster'].
      2. For each assignment in the result, calls AttendanceClassificationAgent.classify for the assigned employee and date.
      3. Returns a dict with assignment results and attendance classifications.
    """

    def __init__(self):
        self.assignment_agent = EmployeeWorkAssignmentAgent()
        self.attendance_agent = AttendanceClassificationAgent()

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Orchestration entrypoint.
        Args:
            input_data: dict with keys 'tasks' (list of task dicts) and 'employee_roster' (list of employee dicts)
        Returns:
            dict with assignment results and attendance classifications
        """
        orchestration_result = {
            "assignment_result": None,
            "attendance_classifications": [],
            "errors": []
        }

        # Step 1: Call EmployeeWorkAssignmentAgent.assign_tasks
        try:
            tasks = input_data.get("tasks")
            employee_roster = input_data.get("employee_roster")
            if not isinstance(tasks, list) or not isinstance(employee_roster, list):
                raise ValueError("Input must contain 'tasks' (list) and 'employee_roster' (list).")
            logger.info("Calling EmployeeWorkAssignmentAgent.assign_tasks with %d tasks and %d employees", len(tasks), len(employee_roster))
            assignment_result = await self.assignment_agent.assign_tasks(tasks, employee_roster)
            orchestration_result["assignment_result"] = assignment_result
        except Exception as e:
            logger.error("Error in EmployeeWorkAssignmentAgent.assign_tasks: %s", e, exc_info=True)
            orchestration_result["errors"].append({
                "step": "assign_tasks",
                "error": str(e)
            })
            # If assignment fails, we cannot proceed to attendance classification
            return orchestration_result

        # Step 2: For each assignment, call AttendanceClassificationAgent.classify
        attendance_classifications = []
        assignments = assignment_result.get("assignments", [])
        # Try to get a date for attendance classification
        # Use the first task's date if available, else require input_data['date']
        date = None
        if tasks and isinstance(tasks, list):
            # Try to find a 'date' field in any task
            for task in tasks:
                if isinstance(task, dict) and "date" in task:
                    date = task["date"]
                    break
        if not date:
            # Try to get from input_data
            date = input_data.get("date")
        if not date:
            # Fallback: use today's date
            from datetime import datetime
            date = datetime.now().strftime("%Y-%m-%d")
            logger.warning("No date found in tasks or input_data; using today's date: %s", date)

        for assignment in assignments:
            try:
                employee_id = assignment.get("employee_id")
                if not employee_id:
                    raise ValueError("Assignment missing 'employee_id'")
                logger.info("Classifying attendance for employee_id=%s, date=%s", employee_id, date)
                attendance_result = await self.attendance_agent.classify(employee_id, date)
                attendance_classifications.append({
                    "employee_id": employee_id,
                    "date": date,
                    "attendance_status": attendance_result.get("status"),
                    "attendance_reason": attendance_result.get("reason", None)
                })
            except Exception as e:
                logger.error("Error in AttendanceClassificationAgent.classify for employee_id=%s: %s", assignment.get("employee_id"), e, exc_info=True)
                orchestration_result["errors"].append({
                    "step": "attendance_classification",
                    "employee_id": assignment.get("employee_id"),
                    "error": str(e)
                })
                # Continue to next assignment

        orchestration_result["attendance_classifications"] = attendance_classifications
        return orchestration_result

# Convenience function for direct use
async def run_orchestration(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience async function to run the orchestration.
    """
    engine = OrchestrationEngine()
    return await engine.execute(input_data)

# Synchronous wrapper for environments that require sync call
def run_orchestration_sync(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Synchronous wrapper for orchestration (runs event loop internally).
    """
    return asyncio.run(run_orchestration(input_data))
