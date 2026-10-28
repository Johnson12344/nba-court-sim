from __future__ import annotations

import pytest

from nba_sim.config import load_settings
from nba_sim.data.demo_provider import DemoProvider
from nba_sim.features.profile_builder import build_team_profile


@pytest.fixture
def settings():
    return load_settings()


@pytest.fixture
def demo_bundle():
    return DemoProvider().load("2025-26")


@pytest.fixture
def demo_profiles(settings, demo_bundle):
    return (
        build_team_profile(demo_bundle.teams, demo_bundle.players, "OKC", settings),
        build_team_profile(demo_bundle.teams, demo_bundle.players, "SAS", settings),
    )

