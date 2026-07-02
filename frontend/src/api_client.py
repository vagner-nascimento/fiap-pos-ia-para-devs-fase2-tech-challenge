"""Cliente HTTP para comunicação com a API backend."""

from __future__ import annotations

import os
import time
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv()

DEFAULT_BACKEND_URL = "http://localhost:8000"
POLL_INTERVAL_SEC = 2.0


class ApiError(Exception):
    """Erro retornado pela API backend."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class TuningClient:
    def __init__(self, base_url: str | None = None, timeout: float = 3600.0):
        self.base_url = (base_url or os.getenv("BACKEND_URL", DEFAULT_BACKEND_URL)).rstrip("/")
        self.timeout = timeout

    def _handle_response(self, response: httpx.Response) -> Any:
        if response.is_success:
            return response.json()
        detail = response.text
        try:
            detail = response.json().get("detail", detail)
        except Exception:
            pass
        raise ApiError(str(detail), response.status_code)

    def health_check(self) -> dict:
        with httpx.Client(base_url=self.base_url, timeout=10.0) as client:
            return self._handle_response(client.get("/health"))

    def list_datasets(self) -> list[str]:
        with httpx.Client(base_url=self.base_url, timeout=10.0) as client:
            data = self._handle_response(client.get("/tuning/datasets"))
            return data.get("datasets", [])

    def run_tuning(self, async_mode: bool = False, **params) -> dict:
        payload = {**params, "async_mode": async_mode}
        with httpx.Client(base_url=self.base_url, timeout=self.timeout) as client:
            data = self._handle_response(client.post("/tuning/run", json=payload))

        if async_mode:
            return self._poll_job(client_base=self.base_url, job_id=data["job_id"])
        return data

    def start_tuning_async(self, **params) -> str:
        """
        Inicia o tuning em modo assíncrono e retorna o job_id imediatamente.

        Diferente de `run_tuning(async_mode=True)`, não bloqueia aguardando o resultado.
        Use este método quando o frontend implementa seu próprio loop de polling.

        Returns:
            job_id: Identificador único do job iniciado.
        """
        payload = {**params, "async_mode": True}
        with httpx.Client(base_url=self.base_url, timeout=30.0) as client:
            data = self._handle_response(client.post("/tuning/run", json=payload))
        return data["job_id"]

    def _poll_job(self, client_base: str, job_id: str) -> dict:
        with httpx.Client(base_url=client_base, timeout=self.timeout) as client:
            while True:
                status_data = self._handle_response(client.get(f"/tuning/jobs/{job_id}"))
                status = status_data["status"]
                if status == "completed":
                    return status_data["result"]
                if status == "failed":
                    raise ApiError(status_data.get("error") or "Job falhou")
                time.sleep(POLL_INTERVAL_SEC)

    def get_generation_snapshots(self, job_id: str, since: int = 0) -> dict:
        """
        Busca snapshots de gerações do AG posteriores a `since`.

        Args:
            job_id: Identificador do job de tuning.
            since: Cursor — apenas gerações com número > since são retornadas.

        Returns:
            Dict com chaves: snapshots (list), last_generation (int), job_status (str).
        """
        with httpx.Client(base_url=self.base_url, timeout=10.0) as client:
            return self._handle_response(
                client.get(f"/tuning/jobs/{job_id}/generations", params={"since": since})
            )

    def get_latest_logs(self) -> dict:
        with httpx.Client(base_url=self.base_url, timeout=30.0) as client:
            return self._handle_response(client.get("/tuning/logs/latest"))


class PipelineClient:
    def __init__(self, base_url: str | None = None, timeout: float = 3600.0):
        self.base_url = (base_url or os.getenv("BACKEND_URL", DEFAULT_BACKEND_URL)).rstrip("/")
        self.timeout = timeout

    def _handle_response(self, response: httpx.Response) -> Any:
        if response.is_success:
            return response.json()
        detail = response.text
        try:
            detail = response.json().get("detail", detail)
        except Exception:
            pass
        raise ApiError(str(detail), response.status_code)

    def run_preprocessing(self) -> dict:
        with httpx.Client(base_url=self.base_url, timeout=self.timeout) as client:
            data = self._handle_response(client.post("/pipeline/preprocess"))
        return self._poll_job(self.base_url, data["job_id"])

    def run_tuning(self, **params) -> dict:
        with httpx.Client(base_url=self.base_url, timeout=self.timeout) as client:
            data = self._handle_response(client.post("/pipeline/tune", json=params))
        return self._poll_job(self.base_url, data["job_id"])

    def start_pipeline_tuning_async(self, **params) -> str:
        """
        Inicia o tuning via /pipeline/tune e retorna o job_id imediatamente.

        Usa o dataset processado pelo preprocessing (path fixo no backend).
        Não bloqueia aguardando o resultado.

        Returns:
            job_id: Identificador único do job iniciado.
        """
        with httpx.Client(base_url=self.base_url, timeout=30.0) as client:
            data = self._handle_response(client.post("/pipeline/tune", json=params))
        return data["job_id"]

    def get_generation_snapshots(self, job_id: str, since: int = 0) -> dict:
        """
        Busca snapshots de gerações do AG posteriores a `since`.

        Chama /tuning/jobs/{job_id}/generations (endpoint dedicado ao dashboard).

        Args:
            job_id: Identificador do job de tuning.
            since: Cursor — apenas gerações com número > since são retornadas.

        Returns:
            Dict com chaves: snapshots (list), last_generation (int), job_status (str).
        """
        with httpx.Client(base_url=self.base_url, timeout=10.0) as client:
            return self._handle_response(
                client.get(f"/tuning/jobs/{job_id}/generations", params={"since": since})
            )

    def run_predictions(self) -> dict:
        with httpx.Client(base_url=self.base_url, timeout=self.timeout) as client:
            data = self._handle_response(client.post("/pipeline/predict"))
        return self._poll_job(self.base_url, data["job_id"])

    def get_pipeline_status(self) -> dict:
        with httpx.Client(base_url=self.base_url, timeout=10.0) as client:
            return self._handle_response(client.get("/pipeline/status"))

    def get_job_status(self, job_id: str) -> dict:
        with httpx.Client(base_url=self.base_url, timeout=10.0) as client:
            return self._handle_response(client.get(f"/pipeline/jobs/{job_id}"))

    def _poll_job(self, client_base: str, job_id: str) -> dict:
        with httpx.Client(base_url=client_base, timeout=self.timeout) as client:
            while True:
                status_data = self._handle_response(client.get(f"/pipeline/jobs/{job_id}"))
                status = status_data["status"]
                if status == "completed":
                    return status_data
                if status == "failed":
                    raise ApiError(status_data.get("error") or "Job falhou")
                time.sleep(POLL_INTERVAL_SEC)


class LLMClient:
    def __init__(self, base_url: str | None = None, timeout: float = 120.0):
        self.base_url = (base_url or os.getenv("BACKEND_URL", DEFAULT_BACKEND_URL)).rstrip("/")
        self.timeout = timeout

    def _handle_response(self, response: httpx.Response) -> Any:
        if response.is_success:
            return response.json()
        detail = response.text
        try:
            detail = response.json().get("detail", detail)
        except Exception:
            pass
        raise ApiError(str(detail), response.status_code)

    def create_session(self, csv_bytes: bytes, filename: str, mappings: dict | None = None) -> dict:
        files = {"file": (filename, csv_bytes, "text/csv")}
        data = {"mappings": __import__("json").dumps(mappings or {})}
        with httpx.Client(base_url=self.base_url, timeout=self.timeout) as client:
            return self._handle_response(
                client.post("/llm/session", files=files, data=data)
            )

    def chat(self, session_id: str, question: str) -> str:
        with httpx.Client(base_url=self.base_url, timeout=self.timeout) as client:
            data = self._handle_response(
                client.post("/llm/chat", json={"session_id": session_id, "question": question})
            )
            return data["answer"]

    def close_session(self, session_id: str) -> None:
        with httpx.Client(base_url=self.base_url, timeout=10.0) as client:
            self._handle_response(client.delete(f"/llm/session/{session_id}"))

    def create_session_from_files(self) -> dict:
        with httpx.Client(base_url=self.base_url, timeout=self.timeout) as client:
            return self._handle_response(client.post("/llm/session/from-files"))
