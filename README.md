# trassenplanung_oeffentlich
first mvp for algorithm based Trassenplanung

## Project Overview

This project aims to enable automated route planning for high-voltage cable networks within Berlin, including downstream cost estimation and permit generation.

## Target Architecture

The solution consists of three sequential steps:

1. **Route Optimization**
   * Input: start and end points
   * Output: optimized cable route
   * Technology: optimization algorithms considering constraints and cost factors

2. **Cost Calculation**
   * Input: optimized route
   * Output: total project cost
   * Technology: rule-based computation (pure deterministic logic)

3. **Permit Generation**
   * Input: route and cost data
   * Output: required approval and permit documents
   * Technology: integration with generative large language models (LLM) for automated document creation

## Current Implementation Status

Only Step 1 is partially implemented:

* **Routing**:
  * Basic automated route generation relying only on publicly available data
  * Limited optimization logic:
    * Preference for proximity to main roads
    * Avoidance of small side streets
    * Water crossings increase cost
    * Primary objective: shortest path

* **Missing Features**:
  * Advanced optimization criteria and internal data (won't be seen on github though)
  * Full cost calculation (Step 2)
  * Automated permit generation using LM (Step 3)

## Goal

Achieve a fully automated pipeline from route planning to cost estimation and permit generation, combining optimization algorithms, rule-based systems, and generative AI.




