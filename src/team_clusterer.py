import cv2
import numpy as np
from typing import List, Tuple, Optional
from sklearn.mixture import GaussianMixture
from src.detector import BBox


class TeamClusterer:
    def __init__(self, n_teams: int = 2):
        self.n_teams = n_teams
        self.gmm: Optional[GaussianMixture] = None

        # Team visualization colors
        self.team_colors_bgr: List[Tuple[int, int, int]] = [
            (255, 0, 0),   # Team A
            (0, 0, 255)    # Team B
        ]

        # Team names
        self.team_names: List[str] = [
            "Team A",
            "Team B"
        ]

        self.is_fitted: bool = False

        # =========================================================
        # REFEREE DETECTION
        # =========================================================
        # These settings are kept from the working version.
        # Referee detection does NOT assume a specific referee color.

        self.referee_min_distance = 42.0
        self.referee_distance_gap = 18.0
        self.referee_high_distance = 50.0
        self.referee_low_confidence = 0.35

        # Frames required to confirm referee
        self.referee_confirm_frames = 4

        # Frames required to remove referee status
        self.referee_release_frames = 10

        # Track-based referee history
        self.referee_votes = {}

        # Confirmed referee tracks
        self.referee_tracks = set()

        # Count frames where confirmed referee
        # looks like a normal player
        self.referee_release_counter = {}

    # =========================================================
    # TORSO CROP
    # =========================================================

    def crop_torso(
        self,
        frame: np.ndarray,
        bbox: BBox
    ) -> np.ndarray:
        """
        Crop upper torso region from player bounding box.

        This is used for:
        - Team clustering
        - Team identification
        - Possession team identification
        """

        h = bbox.height
        w = bbox.width

        # Vertical range: 15% -> 65%
        y1 = max(
            0,
            bbox.y1 + int(h * 0.15)
        )

        y2 = min(
            frame.shape[0],
            bbox.y1 + int(h * 0.65)
        )

        # Horizontal range: 20% -> 80%
        x1 = max(
            0,
            bbox.x1 + int(w * 0.20)
        )

        x2 = min(
            frame.shape[1],
            bbox.x1 + int(w * 0.80)
        )

        crop = frame[y1:y2, x1:x2]

        # Fallback to full bbox
        if crop.size == 0:
            crop = frame[
                bbox.y1:bbox.y2,
                bbox.x1:bbox.x2
            ]

        return crop

    # =========================================================
    # REMOVE GRASS
    # =========================================================

    def mask_grass(
        self,
        torso_bgr: np.ndarray
    ) -> np.ndarray:
        """
        Remove green football pitch pixels
        from the torso crop.
        """

        if (
            torso_bgr is None
            or torso_bgr.size == 0
        ):
            return torso_bgr

        hsv = cv2.cvtColor(
            torso_bgr,
            cv2.COLOR_BGR2HSV
        )

        # Football grass HSV range
        lower_green = np.array([
            30,
            35,
            35
        ])

        upper_green = np.array([
            85,
            255,
            255
        ])

        grass_mask = cv2.inRange(
            hsv,
            lower_green,
            upper_green
        )

        non_grass_mask = cv2.bitwise_not(
            grass_mask
        )

        non_grass_pixels = torso_bgr[
            non_grass_mask > 0
        ]

        # If too few pixels remain,
        # use original torso
        if len(non_grass_pixels) < 15:
            return torso_bgr.reshape(-1, 3)

        return non_grass_pixels

    # =========================================================
    # EXTRACT DOMINANT LAB COLOR
    # =========================================================

    def extract_dominant_color_lab(
        self,
        torso_bgr: np.ndarray
    ) -> np.ndarray:
        """
        Extract dominant jersey color in CIELAB space.
        """

        pixels = self.mask_grass(
            torso_bgr
        )

        if len(pixels) == 0:
            return np.array([
                128.0,
                128.0,
                128.0
            ])

        # BGR -> LAB
        pixels_bgr_2d = pixels.reshape(
            -1,
            1,
            3
        ).astype(np.uint8)

        pixels_lab_2d = cv2.cvtColor(
            pixels_bgr_2d,
            cv2.COLOR_BGR2LAB
        )

        pixels_lab = pixels_lab_2d.reshape(
            -1,
            3
        ).astype(np.float32)

        # Median LAB color
        dominant_lab = np.median(
            pixels_lab,
            axis=0
        )

        return dominant_lab

    # =========================================================
    # FIT TEAM GMM
    # =========================================================

    def fit_gmm(
        self,
        player_crops: List[np.ndarray]
    ) -> bool:
        """
        Fit GMM using player torso LAB features.

        The clustering remains exactly the same
        as the working version.
        """

        lab_features = []

        for crop in player_crops:

            lab_vec = self.extract_dominant_color_lab(
                crop
            )

            lab_features.append(
                lab_vec
            )

        if len(lab_features) < self.n_teams * 2:

            print(
                "⚠️ Warning: Not enough player crops "
                "to fit team clustering GMM."
            )

            return False

        X = np.array(
            lab_features,
            dtype=np.float32
        )

        # GMM
        self.gmm = GaussianMixture(
            n_components=self.n_teams,
            covariance_type="full",
            random_state=42
        )

        self.gmm.fit(X)

        self.is_fitted = True

        # Convert GMM centroids LAB -> BGR
        self.team_colors_bgr = []

        for mean_lab in self.gmm.means_:

            lab_pixel = np.clip(
                mean_lab,
                0,
                255
            ).astype(np.uint8).reshape(
                1,
                1,
                3
            )

            bgr_pixel = cv2.cvtColor(
                lab_pixel,
                cv2.COLOR_LAB2BGR
            )[0, 0]

            bgr_color = (
                int(bgr_pixel[0]),
                int(bgr_pixel[1]),
                int(bgr_pixel[2])
            )

            self.team_colors_bgr.append(
                bgr_color
            )

        print(
            "✅ GMM Team Clustering Successfully Fitted!"
        )

        for idx, col in enumerate(
            self.team_colors_bgr
        ):
            print(
                f"   Team {idx + 1} Centroid "
                f"BGR Color: {col}"
            )

        return True

    # =========================================================
    # PREDICT TEAM / REFEREE
    # =========================================================

    def predict_team(
        self,
        torso_bgr: np.ndarray,
        track_id: Optional[int] = None
    ) -> Tuple[
        int,
        float,
        Tuple[int, int, int]
    ]:
        """
        Predict team membership and optionally identify referee.

        Returns:

            0  -> Team A
            1  -> Team B
           -1  -> Referee

        Also returns:

            confidence
            visualization color

        track_id is used to stabilize referee classification
        across frames.

        This function is also used by possession_tracker.py
        to determine the team of the player possessing the ball.
        """

        # ---------------------------------------------------------
        # GMM not ready
        # ---------------------------------------------------------

        if (
            not self.is_fitted
            or self.gmm is None
        ):
            return (
                0,
                0.5,
                (255, 255, 255)
            )

        # ---------------------------------------------------------
        # Extract LAB color
        # ---------------------------------------------------------

        lab_vec = self.extract_dominant_color_lab(
            torso_bgr
        ).reshape(1, -1)

        # ---------------------------------------------------------
        # GMM probabilities
        # ---------------------------------------------------------

        probs = self.gmm.predict_proba(
            lab_vec
        )[0]

        team_id = int(
            np.argmax(probs)
        )

        confidence = float(
            probs[team_id]
        )

        # Team visualization color
        color_bgr = self.team_colors_bgr[
            team_id
        ]

        # ---------------------------------------------------------
        # Distance from both team centroids
        # ---------------------------------------------------------

        distances = np.linalg.norm(
            self.gmm.means_ - lab_vec,
            axis=1
        )

        min_distance = float(
            np.min(distances)
        )

        sorted_distances = np.sort(
            distances
        )

        distance_gap = float(
            sorted_distances[1]
            - sorted_distances[0]
        )

        # ---------------------------------------------------------
        # REFEREE CANDIDATE
        # ---------------------------------------------------------

        referee_candidate = (
            (
                min_distance
                > self.referee_min_distance
                and
                distance_gap
                < self.referee_distance_gap
            )
            or
            (
                min_distance
                > self.referee_high_distance
                and
                confidence
                < self.referee_low_confidence
            )
        )

        # ---------------------------------------------------------
        # TRACK-BASED REFEREE STABILIZATION
        # ---------------------------------------------------------

        if track_id is not None:

            if referee_candidate:

                self.referee_votes[
                    track_id
                ] = (
                    self.referee_votes.get(
                        track_id,
                        0
                    ) + 1
                )

                self.referee_release_counter[
                    track_id
                ] = 0

                # Confirm referee
                if (
                    self.referee_votes[
                        track_id
                    ]
                    >= self.referee_confirm_frames
                ):

                    self.referee_tracks.add(
                        track_id
                    )

            else:

                # Already confirmed referee
                if track_id in self.referee_tracks:

                    self.referee_release_counter[
                        track_id
                    ] = (
                        self.referee_release_counter.get(
                            track_id,
                            0
                        ) + 1
                    )

                    # Release referee status
                    if (
                        self.referee_release_counter[
                            track_id
                        ]
                        >= self.referee_release_frames
                    ):

                        self.referee_tracks.discard(
                            track_id
                        )

                        self.referee_votes[
                            track_id
                        ] = 0

                        self.referee_release_counter[
                            track_id
                        ] = 0

                else:

                    # Slowly reduce referee votes
                    self.referee_votes[
                        track_id
                    ] = max(
                        0,
                        self.referee_votes.get(
                            track_id,
                            0
                        ) - 1
                    )

            # -----------------------------------------------------
            # CONFIRMED REFEREE
            # -----------------------------------------------------

            if track_id in self.referee_tracks:

                # Visualization only:
                # BGR (255, 0, 255) = Magenta
                return (
                    -1,
                    confidence,
                    (255, 0, 255)
                )

        # ---------------------------------------------------------
        # NORMAL TEAM
        # ---------------------------------------------------------

        return (
            team_id,
            confidence,
            color_bgr
        )