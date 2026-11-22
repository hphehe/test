FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90
ENV SKIP_INTEGRATION=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
RUN pip install --no-cache-dir --no-deps --disable-pip-version-check --root-user-action=ignore \
        duckdb==2.0.0 numpy==2.5.5 pandas==3.1.1 python-dateutil==4.0.0 pytz==2026.6 tzdata==2026.6 \
        typing_extensions==4.16.0 mypy_extensions==1.2.0 \
    || echo "test libraries not installed"
COPY selector.py map.json.gz /app/
RUN python /app/selector.py --prepare /app/sq \
    && (python -m compileall -f -q -j 0 --invalidation-mode unchecked-hash /usr/local/lib/python3.13 /app/st \
            /app/selector.py || echo "some files did not compile") \
    && python -c "import py_compile; py_compile.compile('/app/selector.py', cfile='/app/selector.pyc', doraise=True)" \
    && chmod -R a+rX /app
CMD ["python", "-I", "/app/selector.pyc"]
