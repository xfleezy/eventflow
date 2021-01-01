import numpy as np
import cv2


def flow_error_dense(flow_gt, flow_pred, event_img, is_car=False):
    max_row = flow_gt.shape[1]

    if is_car:
        max_row = 190

    event_img_cropped = np.squeeze(event_img)[:max_row, :]
    flow_gt_cropped = flow_gt[:max_row, :, :]

    flow_pred_cropped = flow_pred[:max_row, :, :]

    event_mask = event_img_cropped > 0

    flow_mask = np.logical_and(
        np.logical_and(~np.isinf(flow_gt_cropped[:, :, 0]), ~np.isinf(flow_gt_cropped[:, :, 1])),
        np.linalg.norm(flow_gt_cropped, axis=2) > 0)
    total_mask = np.squeeze(np.logical_and(event_mask, flow_mask))

    gt_masked = flow_gt_cropped[total_mask, :]
    pred_masked = flow_pred_cropped[total_mask, :]

    EE = np.linalg.norm(gt_masked - pred_masked, axis=-1)
    n_points = EE.shape[0]
    AEE = np.mean(EE)

    # Percentage of points with EE < 3 pixels.
    thresh = 3.
    percent_AEE = float((EE < thresh).sum()) / float(EE.shape[0] + 1e-5)

    return AEE, percent_AEE, n_points



def prop_flow(x_flow, y_flow, x_indices, y_indices, x_mask, y_mask, scale_factor=1.0):
    flow_x_interp = cv2.remap(x_flow,
                              x_indices,
                              y_indices,
                              cv2.INTER_NEAREST)

    flow_y_interp = cv2.remap(y_flow,
                              x_indices,
                              y_indices,
                              cv2.INTER_NEAREST)

    x_mask[flow_x_interp == 0] = False
    y_mask[flow_y_interp == 0] = False

    x_indices += flow_x_interp * scale_factor
    y_indices += flow_y_interp * scale_factor

    return


def estimate_corresponding_gt_flow(x_flow_in,
                                   y_flow_in,
                                   gt_timestamps,
                                   start_time,
                                   end_time):
   
    gt_iter = np.searchsorted(gt_timestamps, start_time, side='right') - 1
    gt_dt = gt_timestamps[gt_iter + 1] - gt_timestamps[gt_iter]
    x_flow = np.squeeze(x_flow_in[gt_iter, ...])
    y_flow = np.squeeze(y_flow_in[gt_iter, ...])

    dt = end_time - start_time

    # No need to propagate if the desired dt is shorter than the time between gt timestamps.
    if gt_dt > dt:
        return x_flow * dt / gt_dt, y_flow * dt / gt_dt

    x_indices, y_indices = np.meshgrid(np.arange(x_flow.shape[1]),
                                       np.arange(x_flow.shape[0]))
    x_indices = x_indices.astype(np.float32)
    y_indices = y_indices.astype(np.float32)

    orig_x_indices = np.copy(x_indices)
    orig_y_indices = np.copy(y_indices)

    # Mask keeps track of the points that leave the image, and zeros out the flow afterwards.
    x_mask = np.ones(x_indices.shape, dtype=bool)
    y_mask = np.ones(y_indices.shape, dtype=bool)

    scale_factor = (gt_timestamps[gt_iter + 1] - start_time) / gt_dt
    total_dt = gt_timestamps[gt_iter + 1] - start_time

    prop_flow(x_flow, y_flow,
              x_indices, y_indices,
              x_mask, y_mask,
              scale_factor=scale_factor)

    gt_iter += 1

    while gt_timestamps[gt_iter + 1] < end_time:
        x_flow = np.squeeze(x_flow_in[gt_iter, ...])
        y_flow = np.squeeze(y_flow_in[gt_iter, ...])

        prop_flow(x_flow, y_flow,
                  x_indices, y_indices,
                  x_mask, y_mask)
        total_dt += gt_timestamps[gt_iter + 1] - gt_timestamps[gt_iter]

        gt_iter += 1

    final_dt = end_time - gt_timestamps[gt_iter]
    total_dt += final_dt

    final_gt_dt = gt_timestamps[gt_iter + 1] - gt_timestamps[gt_iter]

    x_flow = np.squeeze(x_flow_in[gt_iter, ...])
    y_flow = np.squeeze(y_flow_in[gt_iter, ...])

    scale_factor = final_dt / final_gt_dt

    prop_flow(x_flow, y_flow,
              x_indices, y_indices,
              x_mask, y_mask,
              scale_factor)

    x_shift = x_indices - orig_x_indices
    y_shift = y_indices - orig_y_indices
    x_shift[~x_mask] = 0
    y_shift[~y_mask] = 0

    return x_shift, y_shift







