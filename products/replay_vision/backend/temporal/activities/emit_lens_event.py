"""Emit the `$replay_lens` event with the lens output to the customer's events table."""

from datetime import UTC, datetime
from uuid import UUID

import structlog
from temporalio import activity
from temporalio.exceptions import ApplicationError

from posthog.api.capture import capture_internal
from posthog.models.team import Team
from posthog.sync import database_sync_to_async

from products.replay_vision.backend.models.replay_observation import ObservationTrigger, ReplayObservation
from products.replay_vision.backend.temporal.types import EmitLensEventInputs

logger = structlog.get_logger(__name__)

_EVENT_NAME = "$replay_lens"
_EVENT_SOURCE = "replay_vision"


@activity.defn
async def emit_lens_event_activity(inputs: EmitLensEventInputs) -> None:
    """Capture the `$replay_lens` event into the customer's events table; dedup-keyed by observation_id."""
    await database_sync_to_async(_emit_event, thread_sensitive=False)(inputs)


def _emit_event(inputs: EmitLensEventInputs) -> None:
    observation = ReplayObservation.objects.select_related("lens", "team").filter(pk=inputs.observation_id).first()
    if observation is None:
        raise ApplicationError(f"ReplayObservation {inputs.observation_id} not found", non_retryable=True)

    try:
        team: Team = observation.team
    except Team.DoesNotExist:
        raise ApplicationError(f"Team for observation {inputs.observation_id} not found", non_retryable=True)

    properties: dict = {
        # Deterministic id so a worker crash mid-flush doesn't produce a duplicate `$replay_lens` row.
        "$insert_id": str(observation.id),
        "lens_id": str(observation.lens_id),
        "lens_name": observation.lens.name,
        "lens_type": str(observation.lens.lens_type),
        "lens_version": observation.lens_version,
        "session_id": observation.session_id,
        "triggered_by": str(observation.triggered_by),
        "triggered_by_user_id": observation.triggered_by_user_id,
        "model_used": inputs.model_used,
        "provider_used": inputs.provider_used,
        # Flatten so HogQL can query individual output fields without a JSON extract.
        **{f"lens_output_{k}": v for k, v in inputs.model_output.items()},
    }
    distinct_id = (
        str(observation.triggered_by_user_id)
        if observation.triggered_by_user_id is not None and observation.triggered_by == ObservationTrigger.ON_DEMAND
        else f"replay-vision:{observation.team_id}"
    )

    response = capture_internal(
        token=team.api_token,
        event_name=_EVENT_NAME,
        event_source=_EVENT_SOURCE,
        distinct_id=distinct_id,
        timestamp=datetime.now(UTC),
        properties=properties,
        process_person_profile=False,
    )
    response.raise_for_status()


# Helper kept around for tests that want to inspect the loaded objects.
def _load_observation(observation_id: UUID) -> ReplayObservation:
    return ReplayObservation.objects.select_related("lens", "team").get(pk=observation_id)
