# ADR 0003: Use A* for initial rover route planning

- Status: Accepted
- Date: 2026-08-24

## Context

Lunar Pathfinder requires a route planner that considers both physical
distance and terrain risk. Blocked terrain must remain impassable, while
caution and hazardous terrain should carry increasing movement costs.

## Decision

Use eight-direction A* search with the following initial multipliers:

| Terrain class | Cost multiplier |
|---|---:|
| Preferred | 1.0 |
| Caution | 2.0 |
| Hazardous | 8.0 |
| Blocked | Impassable |

Orthogonal movement uses the grid-cell width. Diagonal movement uses
the cell width multiplied by the square root of two.

Diagonal corner-cutting between blocked cells is prohibited.

## Rationale

A* provides an interpretable and computationally efficient baseline.
Its output can be tested on controlled grids, and its objective function
can later incorporate energy, illumination, communications, uncertainty,
and vehicle-specific constraints.

## Consequences

The route minimizes weighted operational cost rather than distance
alone. A longer route may therefore be selected when it substantially
reduces terrain risk.

The current costs are development assumptions and are not validated
against a specific lunar rover.

## Follow-up

Compare the current route against distance-only and energy-aware routes.