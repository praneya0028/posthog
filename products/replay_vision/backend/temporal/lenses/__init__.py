"""Concrete lens implementations + factory keyed off `ReplayLens.lens_type`."""

from typing import Annotated

from pydantic import Field, TypeAdapter

from products.replay_vision.backend.models.replay_lens import ReplayLens
from products.replay_vision.backend.models.replay_observation import ReplayObservation
from products.replay_vision.backend.temporal.lenses.base import BaseLens, BaseLensOutput
from products.replay_vision.backend.temporal.lenses.classifier import ClassifierLens, ClassifierOutput
from products.replay_vision.backend.temporal.lenses.indexer import IndexerLens, IndexerOutput
from products.replay_vision.backend.temporal.lenses.monitor import MonitorLens, MonitorOutput
from products.replay_vision.backend.temporal.lenses.scorer import ScorerLens, ScorerOutput, ScoreScale
from products.replay_vision.backend.temporal.lenses.summarizer import SummarizerLens, SummarizerOutput

AnyLens = Annotated[
    ClassifierLens | IndexerLens | MonitorLens | ScorerLens | SummarizerLens,
    Field(discriminator="lens_type"),
]
_LENS_ADAPTER: TypeAdapter[AnyLens] = TypeAdapter(AnyLens)


def lens_from_db(replay_lens: ReplayLens) -> AnyLens:
    """Build the concrete `BaseLens` subclass for `replay_lens`, validating `lens_config` against its per-type schema."""
    # Spread `lens_config` first so the trusted `ReplayLens` columns override anything that may have leaked into the JSON blob.
    return _LENS_ADAPTER.validate_python(
        {**replay_lens.lens_config, "lens_type": replay_lens.lens_type, "emits_signals": replay_lens.emits_signals}
    )


def lens_from_observation(observation: ReplayObservation) -> AnyLens:
    """Build the lens from `observation.lens_config_snapshot` so a mid-flight lens edit can't switch the prompt/schema under us."""
    # TODO: also snapshot `lens_type` + `emits_signals` on `ReplayObservation` for full snapshot fidelity.
    lens = observation.lens
    return _LENS_ADAPTER.validate_python(
        {**observation.lens_config_snapshot, "lens_type": lens.lens_type, "emits_signals": lens.emits_signals}
    )


__all__ = [
    "AnyLens",
    "BaseLens",
    "BaseLensOutput",
    "ClassifierLens",
    "ClassifierOutput",
    "IndexerLens",
    "IndexerOutput",
    "MonitorLens",
    "MonitorOutput",
    "ScoreScale",
    "ScorerLens",
    "ScorerOutput",
    "SummarizerLens",
    "SummarizerOutput",
    "lens_from_db",
    "lens_from_observation",
]
