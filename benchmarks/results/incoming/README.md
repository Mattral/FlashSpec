# Incoming GPU bundles

Drop the `flashspec_gpu_bundle_<commit>.zip` downloaded from
`notebooks/04_gpu_measurement.ipynb` here, then run:

```bash
python scripts/ingest_gpu_bundle.py --dry-run
python scripts/ingest_gpu_bundle.py
```

The zips are git-ignored. An accepted bundle is unpacked to
`benchmarks/results/gpu/<commit>/` and the README GPU block is rebuilt from
its JSONs.
