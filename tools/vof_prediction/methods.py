import importlib
import numpy as np
import tensorflow as tf
import sklearn.metrics as skm

from skimage.metrics import structural_similarity

class VOFPrediction():
    def set_data(self, data):
        """
        Sets the current sample for prediction.
        """
        self.data = data

    def predict(self, path_architecture, path_model):
        """
        Predicts future volume fraction fields using a trained model and computes 
        geometric and error metrics comparing predicted and reference data.

        :param path_architecture: path to the module containing model-related methods
        :param path_model: path to the trained model file

        :return: predicted volume fraction sequence (output_pred) and a dictionary 
                 containing geometric metrics and error metrics
        """
        # Imports architecture related methods
        self.methods = importlib.import_module(path_architecture)

        # Collects configurations from data
        output_test = self.data["sample"]
        res = output_test.shape[1]
        n_steps = output_test.shape[0]
        xmin, xmax = self.data["x_range"]
        ymin, ymax = self.data["y_range"]
        tmin, tmax = self.data["t_range"]
        dx = (xmax - xmin) / (res - 1)
        dy = (ymax - ymin) / (res - 1)
        dt = (tmax - tmin) / (n_steps)
        vof_threshold = 0.5

        # Loads the selected model
        model = tf.keras.models.load_model(path_model)
        n_steps_slice = model.input_shape[1]
        n_steps_output = model.output_shape[1] if len(model.output_shape) == 5 else 1

        # Predicts the subsequent timesteps given an initial slice
        output_pred = []
        slice = np.expand_dims(output_test[:n_steps_slice], axis=-1)

        for _ in range(n_steps_slice, output_test.shape[0], n_steps_output):
            pred = model.predict(np.expand_dims(slice, 0), verbose=0)[0]
            pred = pred if n_steps_output > 1 else np.array(pred)[np.newaxis, ...]
            output_pred.extend(pred)
            slice = np.concatenate([slice[len(pred):], pred], axis=0)
                
        # Temporarily adds the test n_steps_slice to the network data
        output_pred = np.array(output_pred)[:, :, :, 0]
        output_pred = np.concatenate([output_test[:n_steps_slice, :, :], output_pred], axis=0)

        # Computes the horizontal and vertical diameters, the center of mass, and the contact time        
        # For test data
        dh, dv, cy = self.calculate_metrics(output_test, dx, dy, vof_threshold)
        d_horizontal_test = dh
        d_vertical_test = dv
        c_mass_test = cy
        t_contact_test = self.contact_time(output_test, dt, vof_threshold)
        # For predicted data
        dh, dv, cy = self.calculate_metrics(output_pred, dx, dy, vof_threshold)
        d_horizontal_pred = dh
        d_vertical_pred = dv
        c_mass_pred = cy
        t_contact_pred = self.contact_time(output_pred, dt, vof_threshold)

        # Removes the already known n_steps_slice, keeping only the predicted values
        output_test = output_test[n_steps_slice:]
        output_pred = output_pred[n_steps_slice:]

        # R2-score
        r2s = skm.r2_score(output_test.flatten(), output_pred.flatten())

        # Root Mean Square Error
        rmse = skm.root_mean_squared_error(output_test.flatten(), output_pred.flatten())

        # Structural Similarity Index
        M, H, W = output_test.shape
        ssim_values = np.zeros(M)
        for m in range(M):
            ssim_values[m] = structural_similarity(output_test[m], output_pred[m], data_range=1.0)
        ssim = ssim_values.mean()

        # Dictionary to store calculated metrics
        metrics = {
            "d_horizontal_test": d_horizontal_test, 
            "d_horizontal_pred": d_horizontal_pred, 
            "d_vertical_test": d_vertical_test, 
            "d_vertical_pred": d_vertical_pred,
            "c_mass_test": c_mass_test,
            "c_mass_pred": c_mass_pred,
            "t_contact_test": t_contact_test,
            "t_contact_pred": t_contact_pred,
            "r2s": r2s,
            "rmse": rmse,
            "ssim": ssim
        }

        return output_pred, metrics

    def calculate_metrics(self, sample, dx, dy, threshold):
        """
        Calculates geometric metrics from a sequence of frames.

        :param sample: 3D array (T, H, W) representing the temporal sequence
        :param dx: spatial resolution in x direction
        :param dy: spatial resolution in y direction
        :param threshold: threshold value to define the region of interest

        :return: arrays of horizontal diameter (dh), vertical diameter (dv), 
                 and center of mass in y (cy) over time
        """
        T, H, W = sample.shape
        dh, dv, cy = [], [], []
        for t in range(T):
            phi = sample[t]
            mask = phi >= threshold
            if np.any(mask):
                y_idx, x_idx = np.where(mask)
                dh.append((x_idx.max() - x_idx.min()) * dx)
                dv.append((y_idx.max() - y_idx.min()) * dy)
                cy.append(np.sum(y_idx * phi[mask]) / np.sum(phi[mask]) * dy)
            else:
                dh.append(0); dv.append(0); cy.append(np.nan)
        return np.array(dh), np.array(dv), np.array(cy)

    def contact_time(self, sample, dt, threshold):
        """
        Computes the contact time based on threshold interaction with the wall.

        :param sample: 3D array (T, H, W) representing the temporal sequence
        :param dt: time step between frames
        :param threshold: threshold value to detect contact

        :return: total contact time
        """
        y_wall, idx_wall = None, 0
        for i in range(sample.shape[0]):
            mask = sample[i] > threshold

            ys = np.where(mask.any(axis=1))[0]
            y_t = ys[-1]
            if y_wall is None or y_t > y_wall:
                idx_wall = i
                y_wall = y_t

        count_contact_steps = idx_wall
        for i in range(sample.shape[0]):
            if np.any(sample[i, y_wall, :] > threshold):
                count_contact_steps += 1

        return count_contact_steps * dt