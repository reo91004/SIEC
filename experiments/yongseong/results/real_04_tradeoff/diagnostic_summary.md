# real_04_tradeoff Diagnostic Summary

This summary is diagnostic, not paper-facing.

## Immediate findings

- Fresh 2K rows use these unique per-sample NFE values: [110]
- If the list above contains only `110`, the compute-quality axis has collapsed.
- Best fresh S-IEC FID = 22.6162 (S-IEC p70) vs No correction = 22.5610.
- Always-on FID = 22.6420; best fresh S-IEC delta = -0.0258.
- Best matched-random FID = 22.6092; best fresh S-IEC delta = +0.0070.
- Best matched-uniform FID = 22.5184; best fresh S-IEC delta = +0.0978.

## Seed mismatch

- IEC 50K seed FID = 7.2048
- S-IEC p80 50K seed FID = 8.7873
- These seed rows should not be mixed with fresh 2K rows when making compute-matched claims.
