# Agent Hub Decision Tree API (`agent-hub-api`)

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.121+-009688.svg)](https://fastapi.tiangolo.com)
[![GraphQL](https://img.shields.io/badge/GraphQL-Ariadne-e10098.svg)](https://ariadnegraphql.org)
[![SQLAlchemy 2.0](https://img.shields.io/badge/SQLAlchemy-2.0+-red.svg)](https://www.sqlalchemy.org)

**`agent-hub-api`** is the central orchestration and schema management service of the **Agent Hub** document automation platform. It acts as the single source of truth for customers, projects, decision tree DAGs (Directed Acyclic Graphs), extraction schemas, prompts, and machine learning model catalogs.

Whether you run it as an independent microservice or as the backbone of the Agent Hub system, this service provides both comprehensive **RESTful endpoints** and an **Ariadne GraphQL interface** to manage complex document extraction logic.

---

## Key Features

- **Decision Tree Directed Acyclic Graph (DAG)**:
  - Model complex multi-step document intake workflows.
  - Nodes support document classification, section extraction, business logic validation, and verifier loops.
  - Dynamic branching with conditional edges based on model outputs, document types, or confidence scores.
- **Dual API Design**:
  - **RESTful API**: Fast, ergonomic endpoints for CRUD operations on organizations, projects, trees, nodes, fields, and models.
  - **GraphQL Endpoint**: High-efficiency query interface (`/graphql`) allowing frontends to query deeply nested tree structures in a single round-trip.
- **Multi-Tenant Hierarchy**:
  - Strict isolation between **Customers/Organizations** and **Projects**.
  - Flexible tenant-scoped access controls and authorization.
- **Pluggable Authentication & Sessions**:
  - Built-in session token management signed with HMAC SHA-256 (`SESSION_SECRET`).
  - Bcrypt password hashing for local user accounts.
  - API Key authentication (`X-API-Key`) for automated bot workers and internal microservices.
- **Zero-Config Local Dev & High-Throughput Production**:
  - Instant local startup with SQLite (`sqlite:///./data/decision_tree.db`).
  - Production-ready with PostgreSQL 16 (`psycopg` v3 driver) and PgBouncer transaction pooling.
  - Built-in support for lag-tolerant read replicas (`DB_READ_URL`).
- **Automated Bootstrapping & Seeding**:
  - Automatically seeds default demo projects (e.g., Mortgage Change, Loss Run) and templates from `./seed`.
  - Populates the catalog with available LLM models (Azure OpenAI, OpenAI GPT-4o, and local Ollama models like `qwen3:8b`).

---

## Architecture Overview

```
                        +----------------------+
                        |   Agent Hub Web UI   |
                        +----------+-----------+
                                   |
                     REST / GraphQL (Port 8001)
                                   v
             +---------------------------------------------+
             |         agent-hub-api (FastAPI)             |
             |                                             |
             |  +----------------+    +-----------------+  |
             |  |  Auth / Users  |    |  Decision Trees |  |
             |  +----------------+    +-----------------+  |
             |  |  Model Catalog |    |  GraphQL Schema |  |
             |  +----------------+    +-----------------+  |
             +---------------------+-----------------------+
                                   |
                 +-----------------+-----------------+
                 |                                   |
                 v (PgBouncer: 6432)                 v (Read Replica)
     +-----------------------+           +-----------------------+
     |  PostgreSQL (Primary) |           | PostgreSQL (Replica)  |
     +-----------------------+           +-----------------------+
```

---

## API Endpoints Summary

### Authentication & Users
| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/auth/login` | Authenticate with username & password; returns session cookie |
| `POST` | `/auth/logout` | Terminate active user session |
| `GET` | `/auth/me` | Fetch authenticated user profile & tenant roles |

### Customers & Projects
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/customers` | List all organizations/customers |
| `POST` | `/customers` | Create a new customer organization |
| `GET` | `/projects` | List projects (optionally filtered by `customer_id`) |
| `POST` | `/projects` | Create a new project under an organization |
| `GET` | `/projects/{id}` | Retrieve specific project details |

### Decision Trees & Nodes
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/trees/{project_id}` | Fetch active decision tree and node graph for a project |
| `POST` | `/trees` | Create or update a project's decision tree |
| `POST` | `/nodes` | Create a tree node (classification, extraction, verification) |
| `PUT` | `/nodes/{id}` | Update node prompt, schema fields, or model assignments |
| `DELETE` | `/nodes/{id}` | Remove a node and clean up edge links |

### Model Catalog
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/models` | List supported ML models (Ollama, Azure OpenAI, OpenAI) |
| `POST` | `/models` | Register a new model configuration |

### Ariadne GraphQL
| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/graphql` | Execute GraphQL queries/mutations for nested tree graphs |

### System & Health
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness & database readiness probe |
| `GET` | `/docs` | Interactive Swagger / OpenAPI documentation |

---

## Getting Started

### Prerequisites
- **Python 3.11+**
- **pip** or **uv**
- (Optional) Docker & PostgreSQL 16

### 1. Local Setup (Zero-Config SQLite)

```bash
# Clone the repository
git clone https://github.com/madalavt/agent-hub-api.git
cd agent-hub-api

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install package and dependencies
pip install -e ".[dev]"

# Copy environment configuration
cp .env.example .env

# Run database setup & start the development server
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

Visit the interactive OpenAPI docs at [http://localhost:8001/docs](http://localhost:8001/docs).

### 2. Running with Docker

```bash
# Build the Docker image
docker build -t agent-hub-api .

# Run container with default SQLite volume
docker run -d \
  --name agent-hub-api \
  -p 8001:8001 \
  -e SESSION_SECRET="your-secure-secret-key" \
  -e SEED_ADMIN_PASSWORD="admin" \
  agent-hub-api
```

### 3. Production Deployment with PostgreSQL

When running with PostgreSQL and PgBouncer:

```bash
export DATABASE_URL="postgresql+psycopg://agent_hub:password@pgbouncer:6432/decision_tree"
export DIRECT_DATABASE_URL="postgresql+psycopg://agent_hub:password@postgres:5432/decision_tree"
export API_KEY="your-production-service-api-key"
export SESSION_SECRET="production-random-secret-at-least-32-chars"
export WEB_CONCURRENCY=4

uvicorn app.main:app --host 0.0.0.0 --port 8001 --workers 4
```

---

## Environment Variables

| Variable | Type | Default | Description |
|---|---|---|---|
| `DATABASE_URL` | String | `sqlite:///./data/decision_tree.db` | Primary database connection string |
| `DIRECT_DATABASE_URL`| String | `None` | Direct PostgreSQL connection for startup locks/migrations (bypassing PgBouncer) |
| `DB_READ_URL` | String | `None` | Optional read-replica connection for lag-tolerant queries |
| `API_KEY` | String | `None` | Shared API key required for internal microservice calls (`X-API-Key`) |
| `SESSION_SECRET` | String | `None` | Secret key used to sign user session cookies (required in production) |
| `SESSION_TTL_S` | Integer| `28800` (8 hours) | Session lifetime in seconds |
| `SEED_ADMIN_PASSWORD`| String | `None` | Initial password for the seeded `admin` account (auto-generated if unset) |
| `SEED_ON_START` | Boolean| `true` | Whether to automatically seed demo projects from `./seed` |
| `SEED_LOCAL_MODELS` | Boolean| `true` | Add local Ollama models to catalog |
| `WEB_CONCURRENCY` | Integer| `4` | Number of Uvicorn worker processes |
| `CORS_ORIGINS` | JSON | `["http://localhost:3000"]` | Allowed CORS origins for browser calls |

---

## Running Tests

Unit and integration tests run against an in-memory or temporary database using `pytest`:

```bash
pytest
```

To run lint checks:
```bash
ruff check .
```

---

## License

This project is licensed under the Apache 2.0 / MIT License.
