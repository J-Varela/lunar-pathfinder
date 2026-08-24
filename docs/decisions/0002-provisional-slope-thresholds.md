# ADR 0002: Use provisional slope-based traversability classes

- Status: Accepted for algorithm development
- Date: 2026-08-23

## Context

The route planner requires discrete operational classes or numerical
costs rather than raw terrain slope values.

No specific rover design has been selected. Therefore, the current
project cannot claim vehicle-qualified mobility limits.

## Decision

Use four provisional slope classes:

| Class | Interval | Initial interpretation |
|---|---:|---|
| Preferred | 0° ≤ slope < 8° | Normal travel |
| Caution | 8° ≤ slope < 15° | Increased cost |
| Hazardous | 15° ≤ slope ≤ 25° | Strongly discouraged |
| Blocked | slope > 25° | Excluded from routing |

Represent the classes with unsigned integer identifiers from zero
through three.

## Consequences

The classes provide a deterministic input to route optimization and
make terrain decisions visually interpretable.

The thresholds are engineering assumptions, not specifications for a
real rover. Future versions must incorporate vehicle geometry,
traction, travel direction, surface properties, uncertainty, and
mission safety margins.

## Follow-up

Replace or parameterize these assumptions when Lunar Pathfinder adopts
a specific rover mobility model.