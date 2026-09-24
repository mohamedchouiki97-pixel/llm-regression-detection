# The evaluator as a portable image. Configure it entirely with environment variables:
#   docker run --rm -e OPENAI_API_KEY -v "$PWD/out:/out" mrd run
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.18 /uv /uvx /bin/

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

# Dependencies first, so code changes do not reinstall them.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# pyproject.toml names README.md as the package readme, so the build needs it.
COPY README.md ./
COPY src ./src
COPY templates ./templates
COPY prompts ./prompts
COPY data ./data
RUN uv sync --frozen --no-dev

# State and reports go to /out so a mounted volume keeps them. There is no .git in the image,
# so pass GIT_SHA and GIT_BRANCH to label runs.
ENV PATH="/app/.venv/bin:$PATH" \
    MRD_DB=/out/runs.db \
    MRD_REPORT_DIR=/out/reports
RUN mkdir -p /out

ENTRYPOINT ["mrd"]
CMD ["--help"]
