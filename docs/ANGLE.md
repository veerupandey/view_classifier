# Angle convention

`pred_ang` is the **camera azimuth** around the vehicle (degrees).

- **0°** = looking at the **front**
- Increases **clockwise**
- **90°** = **right** side (passenger side on left-hand-drive vehicles)
- **180°** = **rear**
- **270°** = **left** side

Eight pose bins are 45° wide, centered on those compass points:

| Approx. angle | Pose label |
|---|---|
| 0° | `front` |
| 45° | `front_right` |
| 90° | `right` |
| 135° | `rear_right` |
| 180° | `rear` |
| 225° | `rear_left` |
| 270° | `left` |
| 315° | `front_left` |

The model outputs both the discrete pose (softmax over 8 classes) and a continuous angle via a `(sin θ, cos θ)` head. Training may use MSE on unit sincos (`--angle-loss mse`) or normalized circular degrees (`--angle-loss circular`).
