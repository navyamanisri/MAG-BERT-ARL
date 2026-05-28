"""
MAG-BERT-ARL Backend Package

This is the root package of the MAG-BERT-ARL backend application.
It contains the core modules, API endpoints, services, pipelines, and utility functions
powering the Fair Automated Video Interview Assessment framework.

Module Structure:
- api/: REST API routes, schemas, and endpoint controllers (FastAPI).
- core/: Foundation configurations, centralized logger, and global constants.
- services/: Isolated business logic, third-party adapters, and AI inference helpers.
- utils/: Helper scripts, data formatters, and small common utilities.
- pipelines/: End-to-end multi-modal ingestion, processing, and adversarial training flows.
"""

__version__ = "0.1.0"
