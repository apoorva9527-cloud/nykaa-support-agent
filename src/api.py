import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

from agent import answer_query


PROJECT_FOLDER = Path(__file__).resolve().parents[1]
LOG_FOLDER = PROJECT_FOLDER / "logs"
LOG_FILE = LOG_FOLDER / "requests.jsonl"

LOG_FOLDER.mkdir(exist_ok=True)


app = FastAPI(
    title="Nykaa Support Agent API",
    description="LangGraph-based Nykaa E-commerce Support Agent",
    version="1.0.0",
)


class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        description="Customer support question",
    )


class QueryResponse(BaseModel):
    trace_id: str
    query: str
    answer: str
    intent: str
    grounded: bool
    escalation_score: int
    escalation_level: str
    sources: list[str]
    blocked: bool
    latency_ms: float


def write_log(
    trace_id: str,
    query: str,
    response: dict,
    latency_ms: float,
):
    log_entry = {
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
        "trace_id": trace_id,
        "query": response.get(
            "query",
            query,
        ),
        "intent": response.get(
            "intent",
            "",
        ),
        "grounded": response.get(
            "grounded",
            False,
        ),
        "blocked": response.get(
            "blocked",
            False,
        ),
        "escalation_score": response.get(
            "escalation_score",
            0,
        ),
        "escalation_level": response.get(
            "escalation_level",
            "",
        ),
        "sources": response.get(
            "sources",
            [],
        ),
        "latency_ms": latency_ms,
    }

    with LOG_FILE.open(
        "a",
        encoding="utf-8",
    ) as file:
        file.write(
            json.dumps(
                log_entry,
                ensure_ascii=False,
            )
            + "\n"
        )


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "nykaa-support-agent",
    }


@app.post(
    "/query",
    response_model=QueryResponse,
)
def query_agent(request: QueryRequest):

    trace_id = str(uuid.uuid4())

    start_time = time.perf_counter()

    response = answer_query(
        request.query
    )

    latency_ms = round(
        (time.perf_counter() - start_time)
        * 1000,
        2,
    )

    response["trace_id"] = trace_id
    response["latency_ms"] = latency_ms

    write_log(
        trace_id=trace_id,
        query=request.query,
        response=response,
        latency_ms=latency_ms,
    )

    return response


if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "api:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )