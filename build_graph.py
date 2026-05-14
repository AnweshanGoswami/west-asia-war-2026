import os
import asyncio
from pathlib import Path
from datetime import datetime

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from graphiti_core.llm_client.gemini_client import GeminiClient

# =========================
# ENV VARIABLES
# =========================

neo4j_uri = os.environ["NEO4J_URI"]
neo4j_user = os.environ["NEO4J_USER"]
neo4j_password = os.environ["NEO4J_PASSWORD"]

# Gemini client
llm_client = GeminiClient()

# Graphiti instance
graphiti = Graphiti(
    neo4j_uri,
    neo4j_user,
    neo4j_password,
    llm_client=llm_client
)

# =========================
# PROJECT CONFIG
# =========================

project_root = Path(".")

# Start SMALLER first
# Add more extensions later if needed
extensions = [
    ".py"
]

# Ignore noisy/unnecessary folders
ignore_dirs = {
    "venv",
    ".venv",
    "__pycache__",
    "node_modules",
    ".git",
    "results",
    "snapshots",
    "dist",
    "build"
}

files = []

# =========================
# COLLECT FILES
# =========================

for ext in extensions:
    for file in project_root.rglob(f"*{ext}"):

        # Skip ignored directories
        if any(part in ignore_dirs for part in file.parts):
            continue

        files.append(file)

print(f"Found {len(files)} files to index.\n")

# =========================
# MAIN INDEXING FUNCTION
# =========================

async def main():

    print("Building Neo4j indices and constraints...\n")

    await graphiti.build_indices_and_constraints()

    print("Starting project indexing...\n")

    for file in files:

        try:
            content = file.read_text(
                encoding="utf-8",
                errors="ignore"
            )

            # Skip empty files
            if not content.strip():
                print(f"Skipped empty file: {file}")
                continue

            await graphiti.add_episode(
                name=str(file),
                episode_body=content,
                source=EpisodeType.text,
                source_description="Project source code file",
                reference_time=datetime.utcnow()
            )

            print(f"Indexed: {file}")

            # IMPORTANT:
            # Prevent Gemini rate limits
            await asyncio.sleep(5)

        except Exception as e:

            print(f"Skipped {file}: {e}")

    print("\nGraph indexing complete.")


# =========================
# RUN
# =========================

asyncio.run(main())