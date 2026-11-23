import cv2
from src.detector import FootballDetector
from src.team_clusterer import TeamClusterer
from src.config import INPUT_VIDEO_PATH


def main():
    detector = FootballDetector()
    clusterer = TeamClusterer()

    cap = cv2.VideoCapture(str(INPUT_VIDEO_PATH))

    if not cap.isOpened():
        print("ERROR: Could not open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = "Test Data/team_detection_output.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)
    )

    # =========================
    # Collect player crops
    # =========================

    player_crops = []
    frame_index = 0
    max_samples = 100

    while len(player_crops) < max_samples:

        ret, frame = cap.read()

        if not ret:
            break

        state = detector.process_frame(
            frame,
            frame_index=frame_index,
            use_tracking=True
        )

        for obj in state.players.values():

            crop = clusterer.crop_torso(
                frame,
                obj.bbox
            )

            if crop.size > 0:

                player_crops.append(crop)

                if len(player_crops) >= max_samples:
                    break

        frame_index += 1

    # =========================
    # Fit Team Clustering
    # =========================

    print(
        f"Collected {len(player_crops)} player crops."
    )

    if not clusterer.fit_gmm(player_crops):

        print(
            "ERROR: Could not fit team clustering."
        )

        cap.release()
        return

    # =========================
    # Restart video
    # =========================

    cap.release()

    cap = cv2.VideoCapture(
        str(INPUT_VIDEO_PATH)
    )

    frame_index = 0

    # =========================
    # Process video
    # =========================

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        state = detector.process_frame(
            frame,
            frame_index=frame_index,
            use_tracking=True
        )

        # =========================
        # Draw players + teams
        # =========================

        for obj in state.players.values():

            b = obj.bbox

            crop = clusterer.crop_torso(
                frame,
                b
            )

            # IMPORTANT:
            # Pass the YOLO track ID so that
            # referee classification can be
            # stabilized over time.
            team_id, team_conf, team_color = (
                clusterer.predict_team(
                    crop,
                    track_id=obj.id
                )
            )

            # =========================
            # Referee
            # =========================

            if team_id == -1:

                team_name = "Referee"

                team_color = (
                    255,
                    0,
                    0
                )

            # =========================
            # Team
            # =========================

            else:

                team_name = (
                    clusterer.team_names[
                        team_id
                    ]
                )

            # =========================
            # Draw bounding box
            # =========================

            cv2.rectangle(
                frame,
                (b.x1, b.y1),
                (b.x2, b.y2),
                team_color,
                2
            )

            # =========================
            # Draw label
            # =========================

            label = (
                f"{team_name} "
                f"ID:{obj.id} "
                f"{team_conf:.2f}"
            )

            cv2.putText(
                frame,
                label,
                (
                    b.x1,
                    max(
                        20,
                        b.y1 - 10
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                team_color,
                2
            )

        # =========================
        # Draw YOLO referees
        # =========================

        for obj in state.referees.values():

            b = obj.bbox

            cv2.rectangle(
                frame,
                (b.x1, b.y1),
                (b.x2, b.y2),
                (255, 0, 0),
                2
            )

            label = (
                f"Referee "
                f"ID:{obj.id} "
                f"{obj.confidence:.2f}"
            )

            cv2.putText(
                frame,
                label,
                (
                    b.x1,
                    max(
                        20,
                        b.y1 - 10
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2
            )

        # =========================
        # Draw ball
        # =========================

        if state.ball is not None:

            b = state.ball.bbox

            cv2.rectangle(
                frame,
                (b.x1, b.y1),
                (b.x2, b.y2),
                (0, 0, 255),
                2
            )

            label = (
                f"Ball "
                f"{state.ball.confidence:.2f}"
            )

            cv2.putText(
                frame,
                label,
                (
                    b.x1,
                    max(
                        20,
                        b.y1 - 10
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2
            )

        # =========================
        # Write frame
        # =========================

        out.write(frame)

        frame_index += 1

    # =========================
    # Release
    # =========================

    cap.release()
    out.release()

    print(
        f"Done! Output saved to: {output_path}"
    )


if __name__ == "__main__":
    main()