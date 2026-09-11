def flip_ggarray(ggarray):
    # Extract rotation matrices from ggarray: shape (N, 3, 3)
    """Flip local Y/Z axes in place when local Y opposes world Y; return the array and flip flags."""
    rotations = ggarray[:, 4:13].reshape((-1, 3, 3))

    # Take the y-component of the local y-axis in the world frame.
    # rotations[:, 1, 1] is R[1,1], i.e. the projection of local y onto world y.
    local_y_on_world_y = rotations[:, 1, 1]

    flipped_flags = [False for _ in range(len(ggarray))]
    for idx, y_component in enumerate(local_y_on_world_y):
        if y_component < 0:
            # Flip local y and z axes to keep a consistent orientation
            rotations[idx, :, 1:3] = -rotations[idx, :, 1:3]
            flipped_flags[idx] = True

    # Write back the updated rotations
    ggarray[:, 4:13] = rotations.reshape(-1, 9)
    return ggarray, flipped_flags
