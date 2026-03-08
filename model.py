# =========================
# Python Standard Libraries
# =========================
import os
import time
import random
import threading

# =========================
# Data Processing
# =========================
import numpy as np
import pandas as pd

# Fix cảnh báo np.bool (NumPy >= 1.20)
np.bool = np.bool_
from statsmodels.tsa.vector_ar.var_model import VAR

# =========================
# Model Evaluation
# =========================
from sklearn.metrics import mean_squared_error, mean_absolute_error

# =========================
# Deep Learning (TensorFlow / Keras)
# =========================
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import *
from tensorflow.keras.regularizers import l2

# =========================
# Visualization
# =========================
import matplotlib.pyplot as plt

# =========================
# Utilities
# =========================
from tqdm import tqdm

# =========================
# Streamlit UI
# =========================
import streamlit as st

stop_training = threading.Event()

class StopTrainingCallback:
    def __init__(self):
        self.stop_training = False
    
    def check_stop(self):
        return stop_training.is_set()

# Find best lag based on AIC and plot on Streamlit
def find_bestlag(train_data, range_lag):
    results = []
    AIC = {}
    best_aic, best_lag = np.inf, 0

    # Grid search over lag values
    for i in tqdm(range(range_lag)):
        model = VAR(endog=train_data.values)
        model_result = model.fit(maxlags=i)
        AIC[i] = model_result.aic

        results.append([i, AIC[i]])
        if AIC[i] < best_aic:
            best_aic = AIC[i]
            best_lag = i

    # Prepare result DataFrame
    result_df = pd.DataFrame(results, columns=["p", "AIC"])
    result_df = result_df.sort_values(by="p", ascending=True).reset_index(drop=True)

    # Plot AIC vs lag
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(range(len(AIC)), list(AIC.values()), label="AIC")
    ax.plot([best_lag], [best_aic], marker="o", markersize=8, color="red", label="Best lag")

    ticks = list(range(0, len(AIC), 1))
    labels = [str(i) for i in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=90)

    ax.set_xlabel("lags")
    ax.set_ylabel("AIC")
    ax.set_title("AIC by lag")
    ax.legend()

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("AIC values by lag")
    st.dataframe(result_df)

    # TRẢ VỀ CẢ 2 GIÁ TRỊ
    return result_df, best_lag


# Tạo dự đoán từ mô hình VAR
def create_var_predictions(data, model, lag_order, features):
    lagged_data = []
    for i in range(lag_order, len(data)):        
        pred = model.forecast(data[features].values[i-lag_order:i], steps=1)
        lagged_data.append(pred[0])
    
    return np.array(lagged_data)

# Tạo cửa sổ trượt
def create_windows(data, window_shape, step=1, start_id=None, end_id=None):
    data = np.asarray(data)
    data = data.reshape(-1, 1) if np.prod(data.shape) == np.max(data.shape) else data

    start_id = 0 if start_id is None else start_id
    end_id = data.shape[0] if end_id is None else end_id

    data = data[int(start_id):int(end_id), :]
    window_shape = (int(window_shape), data.shape[-1])
    step = (int(step),) * data.ndim
    slices = tuple(slice(None, None, st) for st in step)
    indexing_strides = data[slices].strides
    win_indices_shape = ((np.array(data.shape) - window_shape) // step) + 1

    new_shape = tuple(list(win_indices_shape) + list(window_shape))
    strides = tuple(list(indexing_strides) + list(data.strides))

    window_data = np.lib.stride_tricks.as_strided(data, shape=new_shape, strides=strides)
    
    return np.squeeze(window_data, 1)

# Gieo số ngẫu nhiên
def set_seed(seed):
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)

def make_callbacks_for_epochs(epochs):
    patience_es = max(8, int(epochs * 0.15))  # 15% tổng số epoch
    patience_rlr = max(3, int(epochs * 0.05)) # 5% tổng số epoch
    es = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience_es,
        min_delta=1e-4,
        restore_best_weights=True,
        verbose=0
    )
    rlr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=patience_rlr,
        min_lr=1e-6,
        verbose=0
    )
    return [es, rlr]

# Chọn 3 giá trị quanh coarse best cho learning_rate và units_lstm
def get_fine_values(val, grid):
    grid_sorted = sorted(grid)
    n = len(grid_sorted)

    # Tìm vị trí của val trong grid
    idx = grid_sorted.index(val)

    # Danh sách giá trị sẽ chọn (luôn cố gắng = 3 giá trị)
    fine_values = [val]

    # 1) Lấy trước nếu có
    if idx - 1 >= 0:
        fine_values.append(grid_sorted[idx - 1])

    # 2) Lấy sau nếu có
    if idx + 1 < n:
        fine_values.append(grid_sorted[idx + 1])

    # Nếu chưa đủ 3 giá trị → mở rộng sang phải trước
    while len(fine_values) < 3:
        if idx + 2 < n:
            fine_values.append(grid_sorted[idx + 2])
        else:
            break

    # Nếu vẫn chưa đủ → mở rộng sang trái
    while len(fine_values) < 3:
        if idx - 2 >= 0:
            fine_values.append(grid_sorted[idx - 2])
        else:
            break

    return sorted(list(set(fine_values)))


def build_lstm (input_dim, output_dim, look_back, look_ahead, lr, units_lstm,
               dropout, L2_reg):
    """
    input_dim: số đặc trưng (n_features)
    output_dim: số đầu ra tại mỗi bước (n_outputs)
        - 'lr': learning rate
        - 'units_lstm': số units của LSTM
        - 'dropout': tắt ngẫu nhiên nơ ron ngăn mô hình nhớ quá kỹ dữ liệu huấn luyện
        - 'L2_reg': giới hạn độ lớn trọng số
        - 'look_back': số bước nhìn lại (sequence length đầu vào)
        - 'look_ahead': số bước dự đoán (sequence length đầu ra)
    """
    set_seed(33)
    input_shape = (look_back, input_dim)
    inp = Input(shape=input_shape)  # Dữ liệu đầu vào: một chuỗi có kích thước (look_back, input_dim)

    # L2 regularization
    regularizer = l2(L2_reg)

    # Encoder: “nhìn lại” quá khứ, nén thông tin vào một vector
    x = LSTM(units_lstm, activation='tanh',
             kernel_regularizer=regularizer,  # Áp dụng L2 regularization lên trọng số đầu vào
             recurrent_regularizer=regularizer, # Áp dụng L2 regularization lên trọng số hồi quy
             dropout=dropout)(inp) 
    
    # Decoder: học cách “giải mã” vector này để dự báo từng bước trong tương lai
    x = RepeatVector(look_ahead)(x)
    x = LSTM(units_lstm, activation='tanh', return_sequences=True,
             kernel_regularizer=regularizer,
             recurrent_regularizer=regularizer,
             dropout=dropout)(x)
    
    out = TimeDistributed(Dense(output_dim))(x)
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse', metrics=['mae'])
    return model

# Tìm siêu tham số tốt nhất bằng grid search
def grid_search (input_dim, output_dim, train_var_pred, train_targets,
                val_var_pred, val_targets, param_grid, look_backk, look_aheadd):

    best_params = None
    best_mse = float("inf")
    start_time = time.time()

    if train_var_pred.shape[0] != train_targets.shape[0]:
        raise ValueError("Mismatch between training features and targets dimensions.")
    if val_var_pred.shape[0] != val_targets.shape[0]:
        raise ValueError("Mismatch between validation features and targets dimensions.")
    
    required_keys = ['learning_rate', 'batch_size', 'units_lstm', 'epoch', 'dropout', 'L2_reg']
    if not all(key in param_grid for key in required_keys):
        raise ValueError(f"Parameter grid must contain {required_keys}.")
    
    es = tf.keras.callbacks.EarlyStopping(patience=30, verbose=0, min_delta=0.001,
                                          monitor='val_loss', mode='auto',
                                          restore_best_weights=True)

    for lr in param_grid['learning_rate']:
        for batch_size in param_grid['batch_size']:
            for units_lstm in param_grid['units_lstm']:
                for epoch in param_grid['epoch']:
                    for dropout in param_grid['dropout']:
                        for L2_reg in param_grid['L2_reg']:

                            model = build_lstm(
                                input_dim=input_dim,
                                output_dim=output_dim,
                                look_back=look_backk,
                                look_ahead=look_aheadd,
                                lr=lr,
                                units_lstm=units_lstm,
                                dropout=dropout,
                                L2_reg=L2_reg
                            )

                            callbacks = make_callbacks_for_epochs(epoch)
                            history = model.fit(
                                train_var_pred, train_targets,
                                epochs=epoch,
                                batch_size=batch_size,
                                validation_data=(val_var_pred, val_targets),
                                verbose=0,
                                callbacks=callbacks
                            )
                            
                            # Epoch thực tế train
                            epochs_trained = len(history.history['loss'])
                            # Learning rate cuối cùng
                            final_lr = float(model.optimizer.learning_rate)

                            # Dự đoán trên validation
                            val_pred = model.predict(val_var_pred)
                            
                            # Để tính MSE trên toàn bộ dữ liệu cần flatten (làm phẳng) toàn bộ mảng
                            mse = mean_squared_error(val_targets.flatten(), val_pred.flatten())

                            if mse < best_mse:
                                best_mse = mse
                                best_params = {
                                    'learning_rate': lr,
                                    'batch_size': batch_size,
                                    'units_lstm': units_lstm,
                                    'epoch': epoch,
                                    'dropout': dropout,
                                    'L2_reg': L2_reg,
                                    'final_lr': final_lr,
                                    'epoch_trained': epochs_trained
                                }

    execution_time = time.time() - start_time
    return best_params, best_mse, execution_time

# Kết quả đánh giá cho từng biến
# Tự động xử lý cả dữ liệu 2D (samples, features) và 3D (samples, timesteps, features)
def evaluate_multivariate_forecast(y_test, predictions, column_names):
    # Nếu dữ liệu là DataFrame thì chuyển sang numpy để tiện xử lý
    if isinstance(y_test, pd.DataFrame):
        y_test_values = y_test.values
    else:
        y_test_values = y_test

    if isinstance(predictions, pd.DataFrame):
        pred_values = predictions.values
    else:
        pred_values = predictions

    # Nếu dữ liệu là 3D, chuyển về 2D
    if y_test_values.ndim == 3 and pred_values.ndim == 3:
        y_test_values = y_test_values.reshape(-1, y_test_values.shape[-1])
        pred_values = pred_values.reshape(-1, pred_values.shape[-1])

    # Kiểm tra kích thước
    assert y_test_values.shape == pred_values.shape, "The shape of y_test and predictions must be the same."

    mse_dict, rmse_dict, mae_dict, cv_rmse_dict = {}, {}, {}, {}

    for i, col in enumerate(column_names):
        y_true_col = y_test_values[:, i]
        y_pred_col = pred_values[:, i]

        mse = mean_squared_error(y_true_col, y_pred_col)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true_col, y_pred_col)
        y_mean = np.mean(y_true_col)
        cv_rmse = rmse / y_mean if y_mean != 0 else np.nan

        mse_dict[col] = mse
        rmse_dict[col] = rmse
        mae_dict[col] = mae
        cv_rmse_dict[col] = cv_rmse

    evaluation_df = pd.DataFrame({
        'Variable': column_names,
        'MSE': pd.Series(mse_dict),
        'RMSE': pd.Series(rmse_dict),
        'MAE': pd.Series(mae_dict),
        'CV_RMSE': pd.Series(cv_rmse_dict)
    })

    return evaluation_df


def evaluate_overall_forecast(y_test, predictions, execution_time=None):
    test_mse = mean_squared_error(y_test.flatten(), predictions.flatten())
    test_mae = mean_absolute_error(y_test.flatten(), predictions.flatten())
    test_rmse = np.sqrt(test_mse)
    y_mean = np.mean(y_test.flatten())
    cv_rmse = test_rmse / y_mean if y_mean != 0 else np.nan

    metric_names = [
        "Test Time (seconds)",
        "Test MSE",
        "Test MAE",
        "Test RMSE",
        "Test CV RMSE"
    ]
    metric_values = [
        f"{execution_time:.2f}" if execution_time is not None else "N/A",
        f"{test_mse:.4f}",
        f"{test_mae:.4f}",
        f"{test_rmse:.4f}",
        f"{cv_rmse:.4f}",
    ]

    evaluation_df_overall = pd.DataFrame({
        "Metric": metric_names,
        "Value": metric_values
    })

    return evaluation_df_overall

def evaluate_overall_forecast_restore(y_test, predictions, execution_time=None):
    test_mse = mean_squared_error(y_test, predictions)
    test_mae = mean_absolute_error(y_test, predictions)
    test_rmse = np.sqrt(test_mse)
    y_mean = np.mean(y_test)
    cv_rmse = test_rmse / y_mean if y_mean != 0 else np.nan

    metric_names = [
        "Test Time (seconds)",
        "Test MSE",
        "Test MAE",
        "Test RMSE",
        "Test CV RMSE"
    ]
    metric_values = [
        f"{execution_time:.2f}" if execution_time is not None else "N/A",
        f"{test_mse:.4f}",
        f"{test_mae:.4f}",
        f"{test_rmse:.4f}",
        f"{cv_rmse:.4f}",
    ]

    evaluation_df_overall = pd.DataFrame({
        "Metric": metric_names,
        "Value": metric_values
    })

    return evaluation_df_overall