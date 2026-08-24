# ADR 0001: Develop against synthetic terrain first

- Status: Accepted
- Date: 2026-08-23

## Context

Lunar Pathfinder will eventually process real lunar elevation data.
Real planetary datasets introduce coordinate systems, missing values,
large files, projection details, and dataset-specific artifacts.

Developing terrain and routing algorithms while simultaneously solving
those ingestion problems would make failures harder to isolate.

## Decision

Begin with deterministic synthetic terrain containing regional relief,
crater bowls, raised rims, and low-amplitude surface roughness.

Use a fixed random seed for reproducible analysis.

## Consequences

### Benefits

- Algorithms can be tested against known conditions.
- Results remain reproducible.
- Visualization and routing can be developed before data ingestion.
- Failures are easier to isolate.
- Tests run quickly without external datasets.

### Limitations

- Synthetic terrain is not evidence about a real lunar site.
- The current noise model does not reproduce lunar regolith physics.
- Real elevation resolution and uncertainty are not yet represented.
- All algorithms must later be validated using authoritative lunar data.

## Follow-up

Introduce real lunar elevation data only after traversability and route
optimization work correctly on controlled terrain.