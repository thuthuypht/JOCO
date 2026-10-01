# JOCO Step 4B — deterministic repair quality (corrected reporting note)

The experiment completed 60/60 selected instances with reference status `Optimal`.

Three Q1 images contained zero detector objects, so their reference objective is
undefined/NaN and objective-loss percentage is not meaningful. Therefore:

- constraint-compliance / repair-time analysis uses all 60 instances;
- objective-loss analysis uses the 57 non-empty instances.

## Corrected overall results
- Objective-loss cases: 57
- Mean objective loss: -1.555e-15 %
- Median objective loss: 0.000e+00 %
- Maximum absolute numerical deviation: 4.484e-14 %
- Mean repair time: 0.000135 s
- Median repair time: 0.000039 s
- Maximum repair time: 0.001947 s
- Violations after repair: 0
- Mean constraint compliance: 100%

The non-zero loss magnitudes are floating-point noise and should be reported as
approximately 0.000% objective loss, not as negative improvements.

The full Step-3B evaluation had no natural fallback (0/500, 0/400, 0/300).
Accordingly, Table 5 natural-fallback fields for objective loss, repair time and
total fallback-case time should be N/A rather than zero.

Step 4B is a controlled repair-quality stress test, not an observed natural
fallback frequency estimate. These empirical results do not constitute an
approximation guarantee.
