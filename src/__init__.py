"""Project source root -- makes `python -m src.rag.<module>` resolvable.

Step 1 (src/acquisition) and Step 2 (src/normalization) are unaffected:
their own scripts still import them as bare top-level packages via a
sys.path insertion (see scripts/build_data.py, scripts/build_table.py),
which keeps working regardless of this file's presence. Step 3
(src/rag) is addressed as src.rag.* instead, and only ever reads Step
2's frozen output files (comparison_table.csv/.parquet, evidence.json)
rather than importing src.normalization's internal classes -- so the two
import styles never need to interoperate.
"""
