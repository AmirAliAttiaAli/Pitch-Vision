import math
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

from src.config import POSSESSION_DIST_THRESHOLD
from src.detector import FrameState, DetectedObject
from src.team_clusterer import TeamClusterer


@dataclass
class PossessionInfo:
    player_id: int
    distance: float
    team_id: int
    team_name: str
    confidence: float
    team_color_bgr: Tuple[int, int, int]
    player_obj: DetectedObject


class PossessionTracker:
    def __init__(
        self,
        dist_threshold: float = POSSESSION_DIST_THRESHOLD,
        confirm_frames: int = 5,
        switch_frames: int = 4,
        release_frames: int = 5,
        keep_ratio: float = 0.80
    ):
        self.dist_threshold = dist_threshold

        # Consecutive frames required before assigning a new possessor.
        self.confirm_frames = confirm_frames

        # Consecutive frames required before switching from the
        # current possessor to another player.
        self.switch_frames = switch_frames

        # Consecutive frames without a valid possessor before releasing.
        self.release_frames = release_frames

        # If the new player is only slightly closer, keep the current
        # possessor to avoid A-B-A-B flickering.
        self.keep_ratio = keep_ratio

        self.current_player_id: Optional[int] = None
        self.current_team_id: Optional[int] = None

        self.candidate_player_id: Optional[int] = None
        self.candidate_team_id: Optional[int] = None
        self.candidate_frames = 0

        self.lost_frames = 0

    def reset(self):
        """Reset possession state."""
        self.current_player_id = None
        self.current_team_id = None

        self.candidate_player_id = None
        self.candidate_team_id = None
        self.candidate_frames = 0

        self.lost_frames = 0

    def _get_team_info(
        self,
        frame: np.ndarray,
        player: DetectedObject,
        clusterer: TeamClusterer
    ):
        """Classify the player's team using the existing TeamClusterer."""
        torso_crop = clusterer.crop_torso(frame, player.bbox)

        return clusterer.predict_team(
            torso_crop,
            track_id=player.id
        )

    def _create_possession_info(
        self,
        player: DetectedObject,
        distance: float,
        team_id: int,
        confidence: float,
        team_color: Tuple[int, int, int],
        clusterer: TeamClusterer
    ) -> PossessionInfo:

        if (
            hasattr(clusterer, "team_names")
            and 0 <= team_id < len(clusterer.team_names)
        ):
            team_name = clusterer.team_names[team_id]
        else:
            team_name = f"Team {team_id + 1}"

        return PossessionInfo(
            player_id=player.id,
            distance=distance,
            team_id=team_id,
            team_name=team_name,
            confidence=confidence,
            team_color_bgr=team_color,
            player_obj=player
        )

    def _clear_candidate(self):
        self.candidate_player_id = None
        self.candidate_team_id = None
        self.candidate_frames = 0

    def get_possession(
        self,
        frame: np.ndarray,
        frame_state: FrameState,
        clusterer: TeamClusterer
    ) -> Optional[PossessionInfo]:

        """
        Stable possession tracking.

        The closest valid player is only a candidate first.
        Possession is confirmed after several consecutive frames.

        When possession already exists:
        - keep it if the current player remains a valid candidate;
        - do not switch because of a tiny distance difference;
        - switch only after the new player remains the best candidate
          for switch_frames consecutive processed frames.

        Referees (team_id == -1) are always excluded.
        """

        # ---------------------------------------------------------
        # 1. No ball / no players
        # ---------------------------------------------------------
        if (
            frame_state.ball is None
            or len(frame_state.players) == 0
        ):
            self.lost_frames += 1

            if self.lost_frames >= self.release_frames:
                self.reset()

            return None

        # ---------------------------------------------------------
        # 2. Build valid possession candidates
        # ---------------------------------------------------------
        bx, by = frame_state.ball.bbox.center
        candidates = []

        for player_id, player in frame_state.players.items():

            # Use feet/bottom-center for field proximity.
            px, py = player.bbox.bottom_center

            distance = math.hypot(
                bx - px,
                by - py
            )

            if distance > self.dist_threshold:
                continue

            team_id, confidence, team_color = self._get_team_info(
                frame,
                player,
                clusterer
            )

            # Referee / rejected classification.
            if team_id == -1:
                continue

            candidates.append({
                "player": player,
                "distance": distance,
                "team_id": team_id,
                "confidence": confidence,
                "team_color": team_color
            })

        # ---------------------------------------------------------
        # 3. No valid player close enough
        # ---------------------------------------------------------
        if not candidates:
            self.lost_frames += 1

            if self.lost_frames >= self.release_frames:
                self.reset()

            return None

        self.lost_frames = 0

        # Nearest valid player first.
        candidates.sort(key=lambda x: x["distance"])
        best = candidates[0]

        best_player = best["player"]
        best_player_id = best_player.id
        best_team_id = best["team_id"]

        # ---------------------------------------------------------
        # 4. Existing possession
        # ---------------------------------------------------------
        if self.current_player_id is not None:

            current_candidate = next(
                (
                    c for c in candidates
                    if c["player"].id == self.current_player_id
                ),
                None
            )

            # Current possessor is still close enough.
            if current_candidate is not None:

                # If another player is only slightly closer,
                # retain current possession.
                if (
                    best_player_id != self.current_player_id
                    and best["distance"]
                    >= current_candidate["distance"] * self.keep_ratio
                ):
                    self._clear_candidate()

                    return self._create_possession_info(
                        current_candidate["player"],
                        current_candidate["distance"],
                        current_candidate["team_id"],
                        current_candidate["confidence"],
                        current_candidate["team_color"],
                        clusterer
                    )

                # Current player remains the best candidate.
                if best_player_id == self.current_player_id:

                    self._clear_candidate()

                    return self._create_possession_info(
                        best_player,
                        best["distance"],
                        best_team_id,
                        best["confidence"],
                        best["team_color"],
                        clusterer
                    )

            # -----------------------------------------------------
            # Current possessor is no longer the valid best holder.
            # Start / continue switch confirmation.
            # -----------------------------------------------------
            if self.candidate_player_id == best_player_id:
                self.candidate_frames += 1
            else:
                self.candidate_player_id = best_player_id
                self.candidate_team_id = best_team_id
                self.candidate_frames = 1

            # Confirm the new possessor.
            if self.candidate_frames >= self.switch_frames:

                self.current_player_id = best_player_id
                self.current_team_id = best_team_id
                self._clear_candidate()

                return self._create_possession_info(
                    best_player,
                    best["distance"],
                    best_team_id,
                    best["confidence"],
                    best["team_color"],
                    clusterer
                )

            # During the transition, keep the old possessor if
            # they are still inside the valid distance.
            if current_candidate is not None:

                return self._create_possession_info(
                    current_candidate["player"],
                    current_candidate["distance"],
                    current_candidate["team_id"],
                    current_candidate["confidence"],
                    current_candidate["team_color"],
                    clusterer
                )

            # Old player is gone and the new player is not confirmed yet.
            return None

        # ---------------------------------------------------------
        # 5. No current possession
        # ---------------------------------------------------------
        if self.candidate_player_id == best_player_id:
            self.candidate_frames += 1
        else:
            self.candidate_player_id = best_player_id
            self.candidate_team_id = best_team_id
            self.candidate_frames = 1

        # Confirm first possession.
        if self.candidate_frames >= self.confirm_frames:

            self.current_player_id = best_player_id
            self.current_team_id = best_team_id
            self._clear_candidate()

            return self._create_possession_info(
                best_player,
                best["distance"],
                best_team_id,
                best["confidence"],
                best["team_color"],
                clusterer
            )

        return None
