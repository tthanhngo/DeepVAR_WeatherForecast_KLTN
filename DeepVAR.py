# =========================
# Standard libraries
# =========================
# =========================
# Python Standard Libraries
# =========================
import os
import time
import json
import pickle
import threading
import base64

# =========================
# Data Processing
# =========================
import numpy as np
import pandas as pd

# Fix cảnh báo np.bool (NumPy >= 1.20)
np.bool = np.bool_

# =========================
# Statistics & Time Series
# =========================
from statsmodels.tsa.vector_ar.var_model import VAR

# =========================
# Deep Learning (TensorFlow / Keras)
# =========================
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import *

# =========================
# Visualization
# =========================
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# Utilities
# =========================
from tqdm import tqdm

# =========================
# Streamlit UI
# =========================
import streamlit as st

from model import (
    find_bestlag, create_var_predictions,
    create_windows, make_callbacks_for_epochs,
    grid_search, evaluate_multivariate_forecast,
    evaluate_overall_forecast, evaluate_overall_forecast_restore,
    build_lstm, get_fine_values
)

# Import custom modules
from preprocessing import (
    preprocess_data, preprocess_data_predict,
    augment_with_gaussian, augment_timeseries_data, 
    check_stationarity, make_stationary, 
    min_max_normalize, z_score_normalize, 
    inverse_transformation, preprocess_data_restore,
)
from visualization import (
    compare_original_augmented, 
    plot_smoothed_time_series, plot_dataset_split,
    plot_loss_curve, plot_normalized_data,
    plot_actual_vs_predicted_new,
    visualize_data, plot_correlation_matrix, 
    _model_display_name, show_last_saved_training_results_box, 
    plot_weather_forecast_single_variable,
)

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/{{"png"}};base64,{encoded_string.decode()});
            background-size: cover
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

def main():
    st.set_page_config(page_title="Deep Vector Autoregression")
    st.title("Weather Indicators Forecasting Tool")
    add_bg_from_local("Background/Home.jpg")

    uploaded_file = st.sidebar.file_uploader("Upload CSV File", type="csv")

    if uploaded_file:

        # 1. Load data
        file_name = uploaded_file.name[:-4]

        dataset_original = pd.read_csv(uploaded_file, sep=';', encoding='utf-8')

        st.write("### Raw Data")
        st.dataframe(dataset_original)
        st.write(dataset_original.shape)
        
        # 2. Tiền xử lý.
        dataset = preprocess_data (dataset_original)
        st.write("### Preprocess Data")
        st.dataframe(dataset)
        st.write(dataset.shape)

        # 3. Trực quan hóa dữ liệu.

        st.markdown(
            "<h1 style='font-size: 30px; color: black;'>Visualization</h1>",
            unsafe_allow_html=True,
        )
        with st.expander("Click to view chart"):
            visualize_data(dataset)

        # 4. Vẽ ma trận tương quan.
        st.markdown(
            "<h1 style='font-size: 30px; color: black;'>Correlation Matrix</h1>",
            unsafe_allow_html=True,
        )
        with st.expander("Click to view chart"):
            plot_correlation_matrix(dataset)

        ## CHỌN CỘT

        st.sidebar.write(
            "<h2 style='font-size: 30px; color: black;'>Select target columns</h2>",
            unsafe_allow_html=True,
        )

        # 1) Lấy danh sách cột numeric
        numeric_cols = dataset.select_dtypes(include=["number"]).columns.tolist()

        # 1.1) Kiểm tra tồn tại của các cặp sin/cos
        has_wsin = "winddir_sin" in numeric_cols
        has_wcos = "winddir_cos" in numeric_cols

        # 1.2) Tạo danh sách cột hiển thị (UI)
        def make_ui_cols(numeric_cols):
            hidden_cols = [
                "winddir_sin", "winddir_cos",
            ]

            # Bỏ cột sin/cos ra khỏi UI
            ui = [c for c in numeric_cols if c not in hidden_cols]

            # Thêm winddir vào đúng vị trí (dựa theo vị trí sin/cos ban đầu)
            if has_wsin and has_wcos:
                pos_w = min(
                    numeric_cols.index("winddir_sin"),
                    numeric_cols.index("winddir_cos"),
                )
                ui.insert(pos_w, "winddir")

            return ui

        ui_cols = make_ui_cols(numeric_cols)

        # 2) Khởi tạo session_state (mặc định CHỌN TẤT CẢ các cột UI)
        if ("targets_cols_keys" not in st.session_state) or (
            st.session_state["targets_cols_keys"] != tuple(ui_cols)
        ):
            st.session_state["targets_cols_keys"] = tuple(ui_cols)
            st.session_state["select_all_targets"] = True
            for c in ui_cols:
                st.session_state[f"target_{c}"] = True

        # 3) Expander chọn cột
        with st.sidebar.expander("Target column selector", expanded=False):

            # --- Recommendation box (dark) ---
            st.markdown(
                """
                <div style="
                    background-color:#DADADA;
                    border-left:5px solid #FF4B4B;
                    padding:10px 12px;
                    border-radius:8px;
                    color:#263238;
                    margin:6px 0 12px 0;
                    font-size:14px;
                    line-height:1.5;
                ">
                    <span style="font-size:16px; margin-right:0.5px;">💡</span>
                    <b>Recommendation:</b> Select all 11 weather variables to ensure sufficient information for model learning and to generate a comprehensive weather forecast.
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Khởi tạo giá trị mặc định trong session_state (chỉ làm 1 lần)
            if "select_all_targets" not in st.session_state:
                st.session_state["select_all_targets"] = True

            # Dùng toggle thay cho checkbox, KHÔNG truyền value= nữa
            select_all = st.toggle(
                "Use all numeric columns as targets",
                key="select_all_targets",
                help="Enable this to use all numeric columns as targets.",
            )

            st.markdown("---")

            # Danh sách checkbox / toggle cho từng cột
            for c in ui_cols:
                st.checkbox(
                    c,
                    key=f"target_{c}",
                    value=True if select_all else st.session_state.get(f"target_{c}", True),
                    disabled=select_all,  # nếu select_all = True thì khóa các ô riêng lẻ
                )

        # 4) Tính selected_ui_cols (không cần Apply — cập nhật tức thời)
        if st.session_state["select_all_targets"]:
            selected_ui_cols = ui_cols.copy()
        else:
            selected_ui_cols = [
                c for c in ui_cols if st.session_state.get(f"target_{c}", False)
            ]

        # 5) Chuyển UI → cột thực tế
        target_columns = []
        for c in selected_ui_cols:
            if c == "winddir":
                if not (has_wsin and has_wcos):
                    st.error(
                        "To use 'winddir', the dataset must contain both 'winddir_sin' and 'winddir_cos'."
                    )
                    st.stop()
                # GIỮ THỨ TỰ SIN/COS
                target_columns.extend(["winddir_sin", "winddir_cos"])
            else:
                # Các cột numeric bình thường (temp, dew, humidity, ...)
                target_columns.append(c)

        if not target_columns:
            st.error("Please select at least one target column.")
            st.stop()

        # 6) Lưu vào session (để các bước sau dùng thống nhất)
        st.session_state["last_applied_ui"] = selected_ui_cols
        st.session_state["last_applied_targets"] = target_columns
        st.session_state["target_columns"] = target_columns

        # 7) Subset dataset
        dataset = dataset[target_columns].copy()

        # 8) Hiển thị kết quả
        st.markdown(
            "<h1 style='font-size: 30px; color: black;'>Selected target column(s)</h1>",
            unsafe_allow_html=True,
        )
        st.success(f"Using {len(target_columns)} target column(s).")
        st.write("Target columns:", target_columns)

        ## KIỂM TRA TÍNH DỪNG (dataset_diff, hiển thị trong expander)

        # State mặc định
        if "flag_diff" not in st.session_state:
            st.session_state.flag_diff = False
        if "lag" not in st.session_state:
            st.session_state.lag = 0
        if "dataset_diff" not in st.session_state:
            st.session_state.dataset_diff = None

        # --- thêm state cho sai phân bậc 2 ---
        if "flag_diff2" not in st.session_state:
            st.session_state.flag_diff2 = False
        if "dataset_diff2" not in st.session_state:
            st.session_state.dataset_diff2 = None

        # --- Tiêu đề ---
        st.markdown(
            "<h1 style='font-size: 30px; color: black;'>Stationarity test results</h1>",
            unsafe_allow_html=True,
        )

        # --- Giao diện ---
        st.sidebar.markdown(
            "<h1 style='font-size: 30px; color: black;'>Check Stationarity</h1>",
            unsafe_allow_html=True,
        )

        # Chọn mode sai phân
        mode = st.sidebar.radio(
            "Differencing mode",
            options=["all", "non-stationary"],
            index=0,
            help="all: Differentiate all columns; non-stationary: only non-stationary columns (ADF test).",
            key="diff_mode_radio",
        )

        # Ngưỡng alpha cho ADF
        alpha = st.sidebar.number_input(
            "Significant level (α) for ADF",
            min_value=0.00, max_value=0.10, value=0.05, step=0.01,
            key="diff_alpha_num",
        )

        # Lưu lại thông tin vào session
        st.session_state.diff_mode = mode
        st.session_state.diff_alpha = alpha

        # Checkbox để chạy kiểm định và sai phân
        check_stationarity_cb = st.sidebar.checkbox("Check Stationarity", key="check_stationarity_cb")
        apply_diff_cb = st.sidebar.checkbox(
            "Make Data Stationary (lag=1)",
            value=st.session_state.flag_diff,
            key="make_stationary_cb",
            help="Apply first-order differencing once. Disabled after first application."
        )

        # --- HIỂN THỊ KẾT QUẢ ---
        with st.expander("View stationarity test results before/after differencing", expanded=False):

            # 1. Trước khi sai phân
            if check_stationarity_cb:
                stationarity_before = check_stationarity(dataset, alpha=alpha)
                st.subheader("Before differencing")
                st.dataframe(stationarity_before)

            # 2. Thực hiện sai phân 1 lần duy nhất (bậc 1)
            if apply_diff_cb and not st.session_state.flag_diff:
                dataset_diff = make_stationary(dataset, mode=mode, lag=1, alpha=alpha)
                st.session_state.dataset_diff = dataset_diff
                st.session_state.flag_diff = True
                st.session_state.lag = 1

            # 3. Nếu đã sai phân bậc 1, dùng lại bản đã sai phân
            if st.session_state.flag_diff and st.session_state.dataset_diff is not None:
                dataset_after_first = st.session_state.dataset_diff.copy()

                st.subheader("After first-order differencing (lag=1)")
                stationarity_after = check_stationarity(dataset_after_first, alpha=alpha)
                st.dataframe(stationarity_after)

                # --- KIỂM TRA CÒN BIẾN KHÔNG DỪNG KHÔNG ---
                # Cột 'Stationary' kiểu bool: True = dừng, False = chưa dừng
                if "Stationary" in stationarity_after.columns:
                    has_non_stationary = (stationarity_after["Stationary"] != 'Yes').any()

                else:
                    # Dùng p-value, ví dụ cột 'p_value':
                    # has_non_stationary = (stationarity_after["p_value"] > alpha).any()
                    has_non_stationary = False  # TODO: chỉnh lại theo cấu trúc thực tế

                # 4. Nếu còn biến chưa dừng -> hiện checkbox sai phân bậc 2 (mode='all')
                if has_non_stationary:
                    st.info("Some variables are still non-stationary after first-order differencing.")

                    apply_diff2_cb = st.checkbox(
                        "Apply second-order differencing (lag=2, mode='all')",
                        key="make_stationary_2nd_cb",
                        help="Perform an additional differencing step on all columns."
                    )

                    # Chỉ chạy sai phân bậc 2 một lần
                    if apply_diff2_cb and not st.session_state.flag_diff2:
                        # Sai phân thêm 1 lần nữa trên dữ liệu đã sai phân bậc 1
                        dataset_diff2 = make_stationary(
                            dataset_after_first,
                            mode="all",   # luôn mode all như yêu cầu
                            lag=1,        # thêm 1 lần diff nữa => tổng cộng bậc 2
                            alpha=alpha
                        )
                        st.session_state.dataset_diff2 = dataset_diff2
                        st.session_state.flag_diff2 = True
                        st.session_state.lag = 2

                # 5. Quyết định dataset cuối cùng dùng cho phần dưới:
                if st.session_state.flag_diff2 and st.session_state.dataset_diff2 is not None:
                    # Đã sai phân bậc 2
                    dataset = st.session_state.dataset_diff2.copy()
                    st.subheader("After second-order differencing (lag=2)")
                    stationarity_after2 = check_stationarity(dataset, alpha=alpha)
                    st.dataframe(stationarity_after2)
                else:
                    # Chỉ dừng ở bậc 1
                    dataset = dataset_after_first.copy()

        # Hiển thị xem dữ liệu sau khi sai phân.
        with st.expander("View data after stationary transformation"):
            # Chọn cách hiển thị: Table hoặc Chart
            view_mode = st.radio(
                "Display mode",
                options=["Table", "Chart"],
                index=0,
                horizontal=True
            )

            if view_mode == "Table":
                st.subheader("Data table (after stationarity)")
                st.dataframe(dataset, use_container_width=True)

            else:
                st.subheader("Line chart by column")

                # Chỉ cho chọn các cột numeric để vẽ
                numeric_cols = dataset.select_dtypes(include=["number"]).columns.tolist()
                if not numeric_cols:
                    st.warning("No numeric columns available to plot.")
                else:
                    selected_col = st.selectbox(
                        "Select a column to visualize",
                        options=numeric_cols,
                        index=0,
                    )

                    # Cho chọn window để smoothing
                    window = st.slider(
                        "Rolling window size",
                        min_value=3,
                        max_value=60,
                        value=14,
                        step=1,
                        help="Number of time steps used to compute the rolling mean."
                    )

                    plot_smoothed_time_series(
                        dataset=dataset,
                        column=selected_col,
                        window=window,
                        title=f"Smoothed time series for '{selected_col}'"
                    )

        ## CHUẨN HÓA DỮ LIỆU
        # Tiêu đề bên sidebar
        st.sidebar.write(
            "<h2 style='font-size: 30px; color: black;'>Normalization</h2>",
            unsafe_allow_html=True,
        )

        # Radio chọn phương pháp
        selected_method = st.sidebar.radio(
            "**Select Data Normalization Method:**",
            ["No Normalization", "Min-Max Normalization", "Z-Score Normalization"],
        )

        # --- KHỞI TẠO STATE ---
        if "normalization_scaler" not in st.session_state:
            st.session_state.normalization_scaler = None
        if "normalization_method" not in st.session_state:
            # method ĐÃ ÁP DỤNG gần nhất
            st.session_state.normalization_method = "No Normalization"

        # Min–Max mặc định 0.0–1.0 (chỉ là giá trị hiển thị)
        if "norm_min_val" not in st.session_state:
            st.session_state.norm_min_val = 0.0
        if "norm_max_val" not in st.session_state:
            st.session_state.norm_max_val = 1.0

        # Mặc định: không thay đổi dataset cho tới khi người dùng tick chọn
        scaled_data = dataset.copy()
        scaler = None
        applied = False  # cờ đã áp dụng trong lần rerun này

        # ====================== MIN–MAX ======================
        if selected_method == "Min-Max Normalization":
            st.sidebar.markdown("### Min–Max Range")

            min_val = st.sidebar.number_input(
                "Minimum value:",
                min_value=-10.0, max_value=10.0,
                value=st.session_state.norm_min_val,
                step=0.1, format="%.1f",
                key="norm_min_input",
            )
            max_val = st.sidebar.number_input(
                "Maximum value:",
                min_value=-10.0, max_value=10.0,
                value=st.session_state.norm_max_val,
                step=0.1, format="%.1f",
                key="norm_max_input",
            )

            # Checkbox: chỉ khi tick mới chuẩn hóa
            apply_minmax = st.sidebar.checkbox(
                "Apply Min–Max normalization",
                key="apply_minmax_norm",
                help="Tick to normalize data using Min–Max with the range above.",
            )

            if apply_minmax:
                if min_val >= max_val:
                    st.sidebar.error("Minimum value must be less than maximum value.")
                else:
                    scaled_data, scaler = min_max_normalize(
                        min=min_val,
                        max=max_val,
                        dataset=dataset,
                    )
                    st.session_state.norm_min_val = float(min_val)
                    st.session_state.norm_max_val = float(max_val)
                    st.session_state.normalization_scaler = scaler
                    st.session_state.normalization_method = "Min-Max Normalization"
                    applied = True
                    st.sidebar.success(f"Applied Min–Max [{min_val} .. {max_val}]")

        # Z-SCORE
        elif selected_method == "Z-Score Normalization":

            apply_z = st.sidebar.checkbox(
                "Apply Z-Score normalization",
                key="apply_zscore_norm",
                help="Tick to normalize data using Z-Score.",
            )

            if apply_z:
                scaled_data, scaler = z_score_normalize(dataset)
                st.session_state.normalization_scaler = scaler
                st.session_state.normalization_method = "Z-Score Normalization"
                applied = True
                st.sidebar.success("Applied Z-Score normalization")

        # NO NORMALIZATION
        else:
            # Không chuẩn hóa – chỉ copy dataset hiện tại
            # (không cần checkbox vì chính radio đã là lựa chọn “không chuẩn hóa”)
            scaled_data = dataset.copy()
            scaler = None
            st.session_state.normalization_scaler = None
            st.session_state.normalization_method = "No Normalization"

        # CẬP NHẬT dataset khi đã tick checkbox (Min–Max / Z-Score)
        if applied:
            dataset = scaled_data

        # Cho phép người dùng chọn cách hiển thị dữ liệu
        st.markdown(
            "<h1 style='font-size: 30px; color: black;'>View normalized data</h1>",
            unsafe_allow_html=True,
        )

        # Hiển thị trạng thái đang dùng (đã áp dụng gần nhất)
        st.caption(
            f"Normalization in use: **{st.session_state.normalization_method}**"
            + (
                f" (range [{st.session_state.norm_min_val} .. {st.session_state.norm_max_val}])"
                if st.session_state.normalization_method == "Min-Max Normalization"
                else ""
            )
        )

        with st.expander("View data after normalization", expanded=True):
            tab_datatable, tab_chart = st.tabs(["View Data Table", f"Normalized Data Chart ({st.session_state.normalization_method})"])

            # ----- TAB 1: SCALED -----
            with tab_datatable:
                st.subheader(f"Normalized data ({st.session_state.normalization_method})")
                st.dataframe(dataset, use_container_width=True)

            with tab_chart:
                st.subheader(f"Normalized Data Chart ({st.session_state.normalization_method})")

                # Chỉ lấy cột numeric để tránh lỗi
                numeric_cols = dataset.select_dtypes(include=["number"]).columns.tolist()

                if not numeric_cols:
                    st.warning("No numeric columns available to plot.")
                else:
                    # Gọi hàm bạn mới tạo
                    plot_normalized_data(dataset[numeric_cols], st.session_state.normalization_method)

        ## Chia Train / Val / Test
        st.sidebar.write(
            "<h2 style='font-size: 30px; color: black;'>Train - Val - Test Split</h2>",
            unsafe_allow_html=True,
        )

        data_length = len(dataset)

        # Chọn tỉ lệ Train / Test trên toàn bộ dataset
        st.sidebar.subheader("Train / Test ratio")

        train_test_ratio = st.sidebar.slider(
            "Train proportion (relative to full dataset)",
            min_value=0.0,
            max_value=1.0,
            value=0.8,
            step=0.01,
        )

        train_size = int(data_length * train_test_ratio)
        test_size = data_length - train_size

        # Chọn tỉ lệ Validation trong phần Train
        st.sidebar.subheader("Validation within Train")

        val_ratio_in_train = st.sidebar.slider(
            "Validation proportion (within Train)",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.01,
            key="train_val_slider",
        )

        val_size = int(train_size * val_ratio_in_train)
        train_final_size = train_size - val_size

        # Kiểm tra hợp lệ
        if train_final_size <= 0 or test_size <= 0:
            st.error(
                f"Invalid split ratios. "
                f"Got Train={train_final_size}, Val={val_size}, Test={test_size}. "
                f"Please adjust the sliders."
            )
        else:
            # Thực hiện chia dataset theo thời gian (không shuffle)
            train_full = dataset.iloc[:train_size]          # nguyên train (trước tách val)
            test  = dataset.iloc[train_size:]          # test = 20% cuối (hoặc theo slider)

            if val_size > 0:
                train_final = train_full.iloc[:-val_size]   # train dùng để fit
                val = train_full.iloc[-val_size:]  # val = 20% cuối của train_full
            else:
                train_final = train_full
                val = dataset.iloc[0:0]         # DataFrame rỗng

            # Hiển thị kết quả chia tập
            st.markdown(
                "<h1 style='font-size: 30px; color: black;'>Dataset splitting results</h1>",
                unsafe_allow_html=True,
            )

            st.write(f"Total samples: {data_length}")
            st.write(
                f"Train: {len(train_final)} samples | "
                f"Validation: {len(val)} samples | "
                f"Test: {len(test)} samples"
            )

            with st.expander("View data table"):
                st.write("### Train:")
                st.dataframe(train_final, use_container_width=True)
                st.write(train_final.shape)

                st.write("### Validation:")
                st.dataframe(val, use_container_width=True)
                st.write(val.shape)

                st.write("### Test:")
                st.dataframe(test, use_container_width=True)
                st.write(test.shape)

            # Vẽ biểu đồ chia tập cho các cột được chọn
            with st.expander("View split charts by column"):
                # Cho phép chọn cột để vẽ
                cols_to_plot = st.multiselect(
                    "Select columns to plot:",
                    options=target_columns,
                    default=target_columns[:5] if len(target_columns) > 5 else target_columns,
                )

                if not cols_to_plot:
                    st.info("Please select at least one column to plot.")
                else:
                    for col in cols_to_plot:
                        plot_dataset_split(col, train_final, val, test)

        ## Thực hiện Augmentation dữ liệu
        # Checkbox cho Augmentation

        st.sidebar.write(
            "<h2 style='font-size: 30px; color: black;'>Augmentation</h2>",
            unsafe_allow_html=True,
        )

        # Tạo danh sách lựa chọn phương pháp tăng cường
        augment_method = st.sidebar.selectbox(
            "Choose Augmentation Method:",
            ["Gaussian", "Numpy"]
        )

        augment_option = st.sidebar.checkbox("Augment Data")

        if augment_option:

            # Áp dụng phương pháp tương ứng
            if augment_method == "Gaussian":
                stddev = 0.05
                mean = 0.0
                dataset_aug = augment_with_gaussian(train_final, mean, stddev)
            elif augment_method == "Numpy":
                dataset_aug = augment_timeseries_data(train_final, len(train_final))

            st.markdown(
            "<h1 style='font-size: 30px; color: black;'>Augmentation</h1>",
            unsafe_allow_html=True,        )

            # Hiển thị thông tin dữ liệu
            st.write("Train data size:", train_final.shape)
            st.write("Augmented train data size:", dataset_aug.shape)

            # Hiển thị tập dữ liệu tăng cường
            st.dataframe(dataset_aug)
            
            with st.expander("View comparison chart"):

                # Chọn cột dữ liệu để so sánh
                column_to_compare = st.selectbox("Select a data column", train_final.columns)

                # Gọi hàm so sánh với cột đã chọn
                compare_original_augmented(train_final, dataset_aug, column_to_compare)

            # Gộp dữ liệu gốc với dữ liệu tăng cường
            train_final = dataset_aug.sort_index()

        ## Lựa chọn mô hình dự báo

        st.sidebar.write(
            "<h2 style='font-size: 30px; color: black;'>Select Model</h2>",
            unsafe_allow_html=True,
        )
        
        model_type = st.sidebar.selectbox(
            "Select Model:",
            [
                "VAR", "VAR_LAI_DEEPVAR", "VAR-LSTM", "DEEPVAR"
            ],
        )   

        st.sidebar.write(
            "<h2 style='font-size: 30px; color: black;'>Model Training</h2>",
            unsafe_allow_html=True,
        )

        # st.sidebar.subheader("Model Training")
        train_button = st.sidebar.button("Train and Optimize")
        stop_button = st.sidebar.button("Stop Training")
        test_button = st.sidebar.button("Test Model")

        if model_type == "VAR":
            # Tạo đường dẫn folder: model_type / file_name
            var_directory_results = f"results/{model_type}/{file_name}"
            var_directory_models = f"models/{model_type}/{file_name}"
            var_directory_evaluation = f"evaluation/{model_type}/{file_name}"
            var_directory_bestlag = f"best_lag/{model_type}/{file_name}"

            # Tạo folders
            os.makedirs(var_directory_results, exist_ok=True)
            os.makedirs(var_directory_models, exist_ok=True)
            os.makedirs(var_directory_evaluation, exist_ok=True)
            os.makedirs(var_directory_bestlag, exist_ok=True)

            var_session_key = f"VAR_{file_name}"

            # Hiển thị thông báo ở trung tâm màn hình thay vì sidebar
            status_placeholder = st.empty()

            stop_training = threading.Event()

            # Callback to Stop Training
            class StopTrainingCallback(tf.keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    if stop_training.is_set():
                        self.model.stop_training = True
                        st.warning(f"Training stopped at epoch {epoch + 1}")

            if stop_button:
                stop_training.set()
                st.warning("Stopping training process...")

            global flag
            flag = False
            
            if train_button and not stop_training.is_set():
                stop_training.clear()

            # TRAIN
            if train_button and not stop_training.is_set():
                stop_training.clear()
                
                spinner_html = '''
                    <div style="display: flex; align-items: center;">
                        <div class="loader"></div>
                        <p style="margin-left: 10px; font-size: 30px">Optimizing parameters...</p>
                    </div>
                    <style>
                        .loader {
                            border: 4px solid #f3f3f3;
                            border-top: 4px solid #3498db;
                            border-radius: 50%;
                            width: 55px; height: 55px;
                            animation: spin 1s linear infinite;
                        }

                        @keyframes spin {
                            0% { transform: rotate(0deg); }
                            100% { transform: rotate(360deg); }
                        }
                    </style>
                    '''

                status_placeholder.markdown(spinner_html, unsafe_allow_html=True)

                st.write("### Model training results")

                status_placeholder.write("### Optimizing parameters...")

                # Tìm best lag
                result_df_org_data, best_lag = find_bestlag(train_full, 31)

                # Lưu toàn bộ bảng AIC/lag 
                save_path = f"best_lag/{model_type}/{file_name}/best_lag.csv"
                result_df_org_data.to_csv(save_path, index=False)

                st.success(f"**Best lag:** {best_lag}")

                ### FIT FINAL VAR WITH LAG CORRESPONTING TO THE BEST AIC ###
                var = VAR(endog=train_full.values)
                var_result = var.fit(maxlags=best_lag)
                #var_result.aic

                with open(f"results/VAR/{file_name}/var_result.pkl", "wb") as f:
                    pickle.dump(var_result, f)

                trained_columns = list(train_full.columns)
                trained_columns_path = f"results/{model_type}/{file_name}/trained_columns.json"
                with open(trained_columns_path, "w", encoding="utf-8") as f:
                    json.dump(trained_columns, f, ensure_ascii=False)

                # Cập nhật status lớn: hoàn tất toàn bộ pipeline
                local_status = st.empty()
                local_status.markdown(
                    f"""
                    <div style="
                        background-color:#E8F5E9;
                        border-left:5px solid #43A047;
                        padding:10px;
                        border-radius:6px;
                        margin-top:10px;
                    ">
                        <b style="color:#1B5E20;">VAR pipeline completed for dataset: <code>{file_name}</code></b><br>
                        <span style="color:#1B5E20;">
                            Model has been trained, saved to disk, and stored in memory for immediate testing.
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Hiệu ứng toast nhỏ góc phải
                st.toast("VAR model has been trained and saved successfully!")

                # Lưu theo KEY riêng
                st.session_state[f"{var_session_key}_result"] = var_result
                st.session_state[f"{var_session_key}_cols"] = trained_columns
                st.session_state[f"{var_session_key}_best_lag"] = best_lag
                st.session_state["manual_stop_requested"] = False

            # STOP
            if stop_button:
                
                stop_training.set()
                st.subheader(f"Last saved {file_name} model training results")
                st.session_state["manual_stop_requested"] = True  # NEW

                # Path to last saved VAR result
                var_result_path = os.path.join(var_directory_results, "var_result.pkl")

                manual_stop = st.session_state.get("manual_stop_requested", False)

                if manual_stop:
                    show_last_saved_training_results_box(
                        model_type=model_type,
                        file_name=file_name,
                        note_if_missing_cols=(
                            f"Apply for: {file_name} weather dataset with all columns "
                            "after first-order differencing (non-stationary) and min-max normalization. "
                            "No data augmentation techniques are performed."
                        ),
                    )

                # Check if a previous VAR model exists
                if os.path.exists(var_result_path) and os.path.getsize(var_result_path) > 0:
                    try:
                        with open(var_result_path, "rb") as f:
                            var_result = pickle.load(f)

                        best_lag = var_result.k_ar  # lag order used in the last training

                        st.markdown(
                            f"""
                            <div style="
                                background-color:#E8F5E9;
                                border-left:5px solid #43A047;
                                padding:10px;
                                border-radius:6px;
                                margin-top:10px;
                                margin-bottom:10px;
                            ">
                                <b style="color:#1B5E20;">
                                    Best lag from last training (k_ar):
                                    <code>{best_lag}</code>
                                </b>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    except Exception as e:
                        st.error(
                            "An error occurred while loading the last VAR model. "
                            "Please retrain the VAR model."
                        )
                        with st.expander("Show technical details"):
                            st.text(str(e))
                else:
                    st.warning(
                        "No saved VAR model was found. "
                        "Please Train the VAR model before using Stop/Test."
                    )

            # TEST
            if test_button:
                st.write("### VAR Model Testing Results")
                os.makedirs(var_directory_evaluation, exist_ok=True)
                
                var_result_path = os.path.join(var_directory_results, "var_result.pkl")

                with open(var_result_path, "rb") as f:
                    var_result = pickle.load(f)

                best_lag = var_result.k_ar
                k_model  = var_result.neqs

                # Shape check
                if test.shape[1] != k_model:
                    st.error(
                        f"Incompatible test data: VAR model was trained with {k_model} variables, "
                        f"but test data has {test.shape[1]} columns. Please retrain or align feature columns."
                    )
                    st.stop()

                # 1) Dự báo VAR
                start_time = time.time()
                predictions = create_var_predictions(test, var_result, best_lag, test.columns)
                execution_time = time.time() - start_time
                actual = test.iloc[var_result.k_ar:]

                # 3) Lấy scaler / flag_diff / lag từ session
                scaler = st.session_state.get("normalization_scaler", None)
                flag_diff = st.session_state.get("flag_diff", False)
                lag = st.session_state.get("lag", 0)

                # 4) Đánh giá trên dữ liệu hiện tại (scaled / differenced)
                evaluation_df = evaluate_multivariate_forecast(
                    actual.values,
                    predictions,
                    test.columns,
                )
                evaluation_df_overall = evaluate_overall_forecast(
                    actual.values,
                    predictions,
                    execution_time,
                )

                # 1) Thông báo thời gian suy luận
                st.success(f"**Inference completed in {execution_time:.2f} seconds.**")
                st.success(f"**Best lag: {best_lag}**")

                # 2) Chuẩn bị bảng dữ liệu hiện tại (scaled)
                try:
                    current_actual_df = pd.DataFrame(actual.values, columns=list(test.columns))
                except Exception:
                    current_actual_df = pd.DataFrame(actual.values)

                try:
                    current_pred_df = pd.DataFrame(predictions, columns=list(test.columns))
                except Exception:
                    current_pred_df = pd.DataFrame(predictions)

                # CÓ SCALER
                if scaler is not None:
                    # Chuẩn bị dữ liệu gốc để hoàn nguyên
                    dataset_for_restore = preprocess_data(dataset_original)

                    if flag_diff:
                        id = train_full.shape[0] + best_lag
                    else:
                        id = train_full.shape[0] + best_lag - 1

                    original_segment = dataset_for_restore.iloc[(id - 1):(id + 1)]
                    original_segment = original_segment[target_columns].copy()

                    exclude_cols = None

                    # Hoàn nguyên dự đoán
                    pred_restore = inverse_transformation(
                        predictions,
                        scaler,
                        original_segment.columns,
                        original_segment,
                        lag,
                        exclude_cols,
                        flag_diff
                    )

                    # Hoàn nguyên giá trị thực tế
                    actual_restore = inverse_transformation(
                        actual.values,
                        scaler,
                        original_segment.columns,
                        original_segment,
                        lag,
                        exclude_cols,
                        flag_diff
                    )

                    # Chuẩn bị DataFrame original để hiển thị
                    if isinstance(actual_restore, pd.DataFrame):
                        actual_restore_df = actual_restore.copy()
                    else:
                        try:
                            actual_restore_df = pd.DataFrame(
                                actual_restore,
                                columns=list(original_segment.columns)
                            )
                        except Exception:
                            actual_restore_df = pd.DataFrame(actual_restore)

                    if isinstance(pred_restore, pd.DataFrame):
                        pred_restore_df = pred_restore.copy()
                    else:
                        try:
                            pred_restore_df = pd.DataFrame(
                                pred_restore,
                                columns=list(original_segment.columns)
                            )
                        except Exception:
                            pred_restore_df = pd.DataFrame(pred_restore)

                    # Đánh giá trên original scale
                    evaluation_df_org = evaluate_multivariate_forecast(
                        actual_restore,
                        pred_restore,
                        actual_restore_df.columns
                    )
                    evaluation_df_overall_org = evaluate_overall_forecast_restore(
                        actual_restore,
                        pred_restore,
                        execution_time
                    )

                    # 4) Lưu CSV evaluation
                    try:
                        evaluation_df.to_csv(
                            os.path.join(var_directory_evaluation, "VAR_evaluation_scaled.csv"),
                            index=False
                        )
                        evaluation_df_overall.to_csv(
                            os.path.join(var_directory_evaluation, "VAR_evaluation_overall_scaled.csv"),
                            index=False
                        )
                        evaluation_df_org.to_csv(
                            os.path.join(var_directory_evaluation, "VAR_evaluation_org.csv"),
                            index=False
                        )
                        evaluation_df_overall_org.to_csv(
                            os.path.join(var_directory_evaluation, "VAR_evaluation_overall_org.csv"),
                            index=False
                        )
                    except Exception as e:
                        st.warning(f"Could not save evaluation CSVs: {e}")

                    # 5) VIEW: TABS (SCALED / ORIGINAL) + VẼ BIỂU ĐỒ & LƯU PNG
                    # st.subheader("Data View (scaled vs original)")
                    with st.expander("View testing results (transformed data vs inverse transformed data)", expanded=True):
                        tab_scaled, tab_original = st.tabs(["Transformed data", "Inverse transformed data"])

                        # Thư mục lưu biểu đồ
                        plots_dir = os.path.join(var_directory_evaluation, "plots")
                        os.makedirs(plots_dir, exist_ok=True)

                        # ----- TAB 1: SCALED -----
                        with tab_scaled:
                            st.markdown("#### Tables")
                            st.markdown("**Actual**")
                            st.dataframe(current_actual_df)
                            st.markdown("**Predictions**")
                            st.dataframe(current_pred_df)

                            st.markdown("#### Plots")
                            variable_names = list(test.columns)
                            for idx, name in enumerate(variable_names):
                                st.subheader(f"{name}")
                                plot_actual_vs_predicted_new(
                                    current_actual_df,
                                    predictions,
                                    variable_index=idx,
                                    step_index=0,
                                    variable_name=name
                                )

                            st.markdown("#### Evaluation")
                            st.dataframe(evaluation_df)
                            st.dataframe(evaluation_df_overall)

                        # ----- TAB 2: ORIGINAL -----
                        with tab_original:
                            st.markdown("#### Tables")
                            st.markdown("**Actual restore**")
                            st.dataframe(actual_restore_df)
                            st.markdown("**Predictions restore**")
                            st.dataframe(pred_restore_df)

                            st.markdown("#### Plots")
                            variable_names_org = list(actual_restore_df.columns)
                            for idx, name in enumerate(variable_names_org):
                                st.subheader(f"{name}")
                                plot_actual_vs_predicted_new(
                                    actual_restore_df,
                                    pred_restore,
                                    variable_index=idx,
                                    step_index=0,
                                    variable_name=name
                                )

                            st.markdown("#### Evaluation")
                            st.dataframe(evaluation_df_org)
                            st.dataframe(evaluation_df_overall_org)

                # KHÔNG CÓ SCALER
                else:
                    st.warning("No scaler found. Results are shown in the current data (no restoration).")

                    # Lưu CSV evaluation với tên trung tính (không 'scaled'/'original')
                    try:
                        evaluation_df.to_csv(
                            os.path.join(var_directory_evaluation, "VAR_evaluation.csv"),
                            index=False
                        )
                        evaluation_df_overall.to_csv(
                            os.path.join(var_directory_evaluation, "VAR_evaluation_overall.csv"),
                            index=False
                        )
                    except Exception as e:
                        st.warning(f"Could not save evaluation CSVs: {e}")

                    # st.subheader("Data View")
                    with st.expander("View results", expanded=True):
                        plots_dir = os.path.join(var_directory_evaluation, "plots")
                        os.makedirs(plots_dir, exist_ok=True)

                        st.markdown("#### Tables")
                        st.markdown("**Actual**")
                        st.dataframe(current_actual_df)
                        st.markdown("**Predictions**")
                        st.dataframe(current_pred_df)

                        st.markdown("#### Plots")
                        variable_names = list(test.columns)
                        for idx, name in enumerate(variable_names):
                            st.subheader(f"{name}")
                            plot_actual_vs_predicted_new(
                                current_actual_df,
                                predictions,
                                variable_index=idx,
                                step_index=0,
                                variable_name=name
                            )

                        st.markdown("#### Evaluation")
                        st.dataframe(evaluation_df)
                        st.dataframe(evaluation_df_overall)

                
        elif model_type == "DEEPVAR":

            deepvar_session_key = f"DEEPVAR_{file_name}"
            
            # Tạo đường dẫn folder: model_type / file_name
            deepvar2_directory_results   = f"results/{model_type}/{file_name}"     
            deepvar2_directory_models    = f"models/{model_type}/{file_name}"
            deepvar2_directory_evaluation = f"evaluation/{model_type}/{file_name}"
            deepvar2_directory_bestlag   = f"best_lag/{model_type}/{file_name}"

            # Tạo folders
            os.makedirs(deepvar2_directory_results, exist_ok=True)
            os.makedirs(deepvar2_directory_models, exist_ok=True)
            os.makedirs(deepvar2_directory_evaluation, exist_ok=True)
            os.makedirs(deepvar2_directory_bestlag, exist_ok=True)

            # 7. Training

            # Hiển thị thông báo ở trung tâm màn hình thay vì sidebar
            status_placeholder = st.empty()

            stop_training = threading.Event()

            # Callback to Stop Training
            class StopTrainingCallback(tf.keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    if stop_training.is_set():
                        self.model.stop_training = True
                        st.warning(f"Training stopped at epoch {epoch + 1}")

            if stop_button:
                stop_training.set()
                st.warning("Stopping training process...")
            
            # Train model
            
            if train_button and not stop_training.is_set():
                stop_training.clear()
                
                spinner_html = '''
                    <div style="display: flex; align-items: center;">
                        <div class="loader"></div>
                        <p style="margin-left: 10px; font-size: 30px">Optimizing parameters...</p>
                    </div>
                    <style>
                        .loader {
                            border: 4px solid #f3f3f3;
                            border-top: 4px solid #3498db;
                            border-radius: 50%;
                            width: 55px; height: 55px;
                            animation: spin 1s linear infinite;
                        }

                        @keyframes spin {
                            0% { transform: rotate(0deg); }
                            100% { transform: rotate(360deg); }
                        }
                    </style>
                    '''

                status_placeholder.markdown(spinner_html, unsafe_allow_html=True)

                st.write("### Model training results")

                status_placeholder.write("### Optimizing parameters...")

                var_result_path     = os.path.join(deepvar2_directory_results, "var_result.pkl")
                best_lag_path       = os.path.join(deepvar2_directory_bestlag, "best_lag.csv")
                trained_columns_path = os.path.join(deepvar2_directory_results, "trained_columns.json")

                # st.info("Training VAR model to determine best lag (no reuse of previous VAR).")

                # Tìm best lag
                result_df_org_data, best_lag = find_bestlag(train_full, 31)

                # Lưu toàn bộ bảng AIC/lag 
                save_path = f"best_lag/{model_type}/{file_name}/best_lag.csv"
                result_df_org_data.to_csv(save_path, index=False)

                st.success(f"**Best lag:** {best_lag}")

                # Train VAR với best_lag
                var = VAR(endog=train_full.values)
                var_result = var.fit(maxlags=best_lag)

                st.success(f"Trained VAR model with best_lag = {best_lag}.")

                # Cập nhật status ở trên cùng: mô hình đang tìm kiếm tham số tối ưu
                local_status = st.empty()

                local_status.markdown(
                    """
                    <div style="
                        background-color:#FFF8E1;
                        border-left:5px solid #F9A825;
                        padding:10px;
                        border-radius:6px;
                        margin-top:10px;
                    ">
                        <b style="color:#E65100;">Model is searching for optimal parameters...</b><br>
                        <span style="color:#E65100;">
                            Please wait while the model performs hyperparameter optimization.
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Chuyển đổi dữ liệu thành window
                look_back = best_lag
                look_ahead = 1

                X_train = create_windows(train_final, window_shape=look_back, end_id=-look_ahead)
                y_train = create_windows(train_final, window_shape=look_ahead, start_id=look_back)

                X_val = create_windows(val, window_shape=look_back, end_id=-look_ahead)
                y_val = create_windows(val, window_shape=look_ahead, start_id=look_back)

                input_dim = X_train.shape[2]                           
                output_dim = y_train.shape[2]    

                # Tìm tham số tối ưu - Multiphase tunning
                # Phase 1-Coarse tuning: Tìm vùng tham số tốt với 3 giá trị thấp - trung bình - cao
                param_grid_phase1 = {
                    "learning_rate": [3e-4, 7e-4, 3e-3],
                    "batch_size": [64, 128, 256],
                    "units_lstm": [32, 96, 256],  # số nơ ron mạng LSTM
                    "epoch": [300],
                    "dropout": [0.0],
                    "L2_reg": [0.0],
                }

                best_params_phase1, best_mse_phase1, search_time_phase1 = grid_search(
                    input_dim,
                    output_dim,
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    param_grid_phase1,
                    look_back,
                    look_ahead
                )

                result_phase1 = {
                    "best_params": best_params_phase1,
                    "look_back": look_back,
                    "best_lag_VAR": int(best_lag),       # ép kiểu int để tránh lỗi JSON
                    "best_MSE": float(best_mse_phase1),        # lưu MSE
                    "search_time_seconds": float(search_time_phase1)  # lưu thời gian
                }
                with open(f"results/{model_type}/{file_name}/DeepVAR_grid_search_results_phase1.json", "w") as f:
                    json.dump(result_phase1, f)

                # Phase 2-Fine Tuning: Thu hẹp phạm vi tìm tham số tối ưu từ khoảng tốt nhất đã xác định
                param_grid_phase2 = {
                    "learning_rate": get_fine_values(best_params_phase1['learning_rate'], [3e-4, 5e-4, 7e-4, 1e-3, 3e-3]),
                    "batch_size": [best_params_phase1['batch_size']],  # giữ nguyên
                    "units_lstm": get_fine_values(best_params_phase1['units_lstm'], [32, 64, 96, 128, 256]),
                    "epoch": [300],
                    "dropout": [0.0, 0.05, 0.1],
                    "L2_reg": [0.0, 1e-6, 1e-5, 1e-4],
                    
                }

                best_params_phase2, best_mse_phase2, search_time_phase2 = grid_search(
                    input_dim,
                    output_dim,
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    param_grid_phase2,
                    look_back,
                    look_ahead
                )

                best_params = best_params_phase2
                best_mse = best_mse_phase2
                search_time = search_time_phase1 + search_time_phase2

                # Gom tất cả thông tin vào 1 dict
                results_to_save = {
                    "best_params": best_params,
                    "look_back": look_back,
                    "best_lag_VAR": int(best_lag),       # ép kiểu int để tránh lỗi JSON
                    "best_MSE": float(best_mse),        # lưu MSE
                    "search_time_seconds": float(search_time)  # lưu thời gian
                }
                # Lưu ra file JSON
                with open(f"results/{model_type}/{file_name}/DeepVAR_grid_search_results.json", "w") as f:
                    json.dump(results_to_save, f, indent=4)

                # Lưu VAR result
                with open(var_result_path, "wb") as f:
                    pickle.dump(var_result, f)

                # Lưu danh sách cột để dùng khi predict
                with open(trained_columns_path, "w", encoding="utf-8") as f:
                    json.dump(list(train_full.columns), f, ensure_ascii=False)

                # print("Parameter optimization completed.")
                st.write("**Best Parameters:**")
                st.json(best_params)
                st.write(f"**Best MSE:** {best_mse:.4f}")
                st.write(f"**Search Time:** {search_time:.2f} seconds")
                st.write(f"**Best Lags (VAR):** {best_lag}")
                st.write(f"**Look back:** {look_back}")

                local_status.markdown(
                    """
                    <div style="
                        background-color:#E3F2FD;
                        border-left:5px solid #1E88E5;
                        padding:10px;
                        border-radius:6px;
                        margin-top:10px;
                    ">
                        <b style="color:#0D47A1;">Parameter optimization completed.</b><br>
                        <span style="color:#0D47A1;">
                            Best hyperparameters have been found. Proceeding to train the DEEPVAR model...
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Popup nhỏ báo optimize xong
                st.toast("Parameter optimization finished. Starting DEEPVAR training...")

                # Training
                st.subheader("Training DEEPVAR model")

                # Training
                final_model = build_lstm(
                                    input_dim,
                                    output_dim,
                                    look_back = look_back,
                                    look_ahead = look_ahead,
                                    lr = best_params["learning_rate"],
                                    units_lstm = best_params["units_lstm"],
                                    dropout = best_params["dropout"],
                                    L2_reg = best_params["L2_reg"]
                                )

                start_time = time.time()
                callbacks = make_callbacks_for_epochs(best_params["epoch"])
                history = final_model.fit(
                    X_train, y_train,
                    validation_data = (X_val, y_val),
                        epochs=best_params["epoch"],
                        batch_size=best_params["batch_size"],
                        verbose=1,
                        callbacks=callbacks
                        )
                end_time = time.time()

                train_time = end_time - start_time
                st.write(f"Training time: {train_time:.2f} second(s)")
                total_train_time = train_time + search_time
                st.write(f"Total time to find optimal parameters and training time: {total_train_time:.2f} second(s)")

                # Lưu model
                final_model.save(
                    f"models/{model_type}/{file_name}/DeepVAR_final_model.keras"
                )

                # Lưu model DEEPVAR 
                st.session_state[f"{deepvar_session_key}_model"] = final_model

                # Lưu metadata 
                st.session_state[f"{deepvar_session_key}_best_params"] = best_params
                st.session_state[f"{deepvar_session_key}_look_back"] = look_back
                st.session_state[f"{deepvar_session_key}_look_ahead"] = look_ahead
                st.session_state[f"{deepvar_session_key}_var_lag"] = best_lag
                
                # Lưu history
                with open(
                    f"results/{model_type}/{file_name}/DeepVAR_training_history.pkl",
                    "wb",
                ) as f:
                    pickle.dump(history.history, f)

                # Tạo DataFrame chứa loss theo từng epoch
                loss_df = pd.DataFrame(
                    {
                        "Epoch": range(1, len(history.history["loss"]) + 1),
                        "Training Loss": history.history["loss"],
                        "Validation Loss": history.history["val_loss"],
                    }
                )

                st.subheader("Training & Validation Loss")

                # Dùng matplotlib + st.pyplot để hiển thị trên Streamlit
                plot_loss_curve(loss_df)

                # Thông báo train xong
                st.success(
                    f"DEEPVAR training completed successfully!"
                )

                # Cập nhật status lớn: hoàn tất toàn bộ pipeline
                local_status.markdown(
                    f"""
                    <div style="
                        background-color:#E8F5E9;
                        border-left:5px solid #43A047;
                        padding:10px;
                        border-radius:6px;
                        margin-top:10px;
                    ">
                        <b style="color:#1B5E20;">DEEPVAR pipeline completed for dataset: <code>{file_name}</code></b><br>
                        <span style="color:#1B5E20;">
                            Model has been trained, saved to disk, and stored in memory for immediate testing.
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.session_state["manual_stop_requested"] = False
                # Popup nhỏ báo train xong
                st.toast("DEEPVAR model trained and saved successfully!")

            # 8. Stop #

            if stop_training.is_set():

                st.subheader(f"Last saved {file_name} model training results")
                st.session_state["manual_stop_requested"] = True  

                grid_search_path = (
                    f"results/{model_type}/{file_name}/DeepVAR_grid_search_results.json"
                )
                training_history_path = (
                    f"results/{model_type}/{file_name}/DeepVAR_training_history.pkl"
                )
                var_result_path = f"results/{model_type}/{file_name}/var_result.pkl"

                # KIỂM TRA FILE
                if not os.path.exists(grid_search_path) or not os.path.exists(training_history_path):
                    st.error("The complete result files could not be found. Please train the model first.")

                else:

                    manual_stop = st.session_state.get("manual_stop_requested", False)

                    if manual_stop:
                        show_last_saved_training_results_box(
                            model_type=model_type,
                            file_name=file_name,
                            note_if_missing_cols=(
                                f"Apply for: {file_name} weather dataset with all columns "
                                "after first-order differencing (non-stationary) and min-max normalization. "
                                "No data augmentation techniques are performed."
                            ),
                        )

                    with open(grid_search_path, "r") as f:
                        grid_search_data = json.load(f)

                    st.write("#### Hyperparameters")

                    # 1) Best params table
                    best_params = grid_search_data.get("best_params", {})
                    if isinstance(best_params, dict) and best_params:
                        st.markdown("**Best params**")
                        df_params = pd.DataFrame(best_params.items(), columns=["Hyperparameter", "Value"])

                        # ÉP Value thành string
                        df_params["Value"] = df_params["Value"].astype(str)

                        st.dataframe(df_params, use_container_width=True)


                    # 2) Summary table
                    summary = {k: v for k, v in grid_search_data.items() if k != "best_params"}
                    if summary:
                        st.markdown("**Search summary**")
                        df_summary = pd.DataFrame(summary.items(), columns=["Metric", "Value"])

                        # ÉP Value thành string
                        df_summary["Value"] = df_summary["Value"].astype(str)

                        st.dataframe(df_summary, use_container_width=True)

                    # --- 1. Nếu đã có VAR train từ trướctrước, load lên dùng ngay ---
                    if os.path.exists(var_result_path) and os.path.getsize(var_result_path) > 0:
                        try:
                            with open(var_result_path, "rb") as f:
                                var_result = pickle.load(f)
                            best_lag = var_result.k_ar

                            # st.info(f"Best_lag = {best_lag}.")
                        except Exception as e:
                            st.warning("Found VAR model but failed to load it. Recomputing VAR...")
                            var_result = None
                            best_lag = None

                    # LOAD TRAINING HISTORY
                    with open(training_history_path, "rb") as f:
                        loaded_history = pickle.load(f)

                    # FORMAT LOSS DATAFRAME
                    if "val_loss" in loaded_history:
                        loss_df = pd.DataFrame(
                            {
                                "Epoch": range(1, len(loaded_history["loss"]) + 1),
                                "Training Loss": loaded_history["loss"],
                                "Validation Loss": loaded_history["val_loss"],
                            }
                        )
                    else:
                        loss_df = pd.DataFrame(
                            {
                                "Epoch": range(1, len(loaded_history["loss"]) + 1),
                                "Training Loss": loaded_history["loss"],
                            }
                        )

                    # VẼ BIỂU ĐỒ LOSS
                    st.write("#### Training & Validation Loss")
                    st.line_chart(loss_df.set_index("Epoch"))

                    # HIỂN THỊ LOSS CUỐI CÙNG
                    st.write("#### Final Losses")
                    st.write(f"**Training Loss (final)**: {loaded_history['loss'][-1]:.4f}")

                    if "val_loss" in loaded_history:
                        st.write(f"**Validation Loss (final)**: {loaded_history['val_loss'][-1]:.4f}")

            # 9. Test #

            if test_button:

                st.write("### DEEPVAR Model Testing Results")

                grid_search_path      = f"results/{model_type}/{file_name}/DeepVAR_grid_search_results.json"
                model_path            = f"models/{model_type}/{file_name}/DeepVAR_final_model.keras"
                var_result_path       = f"results/{model_type}/{file_name}/var_result.pkl"
                training_history_path = f"results/{model_type}/{file_name}/DeepVAR_training_history.pkl"

                os.makedirs(deepvar2_directory_evaluation, exist_ok=True)

                # Check all required files
                if (
                    not os.path.exists(grid_search_path)
                    or not os.path.exists(model_path)
                    or not os.path.exists(var_result_path)
                    or not os.path.exists(training_history_path)
                ):
                    st.error("The complete result files could not be found. Please train the DEEPVAR model first.")
                else:
                    # 1) Load grid search (display only)
                    with open(grid_search_path, "r") as f:
                        grid_search_data = json.load(f)

                    # Lấy best_lag từ JSON (key: best_lag_VAR)
                    if "best_lag_VAR" not in grid_search_data:
                        st.error("Key 'best_lag_VAR' not found in DeepVAR_grid_search_results.json. Please retrain the model.")
                        st.stop()

                    best_lag = int(grid_search_data["best_lag_VAR"])
                    # 2) Load trained DeepVAR model

                    deep_var_model = st.session_state.get(f"{deepvar_session_key}_model", None)
                    model_source = "memory"

                    if deep_var_model is None:
                        # Không có trong session, load từ file
                        deep_var_model = load_model(model_path)
                        model_source = "disk"
                        # Đồng bộ lại vào session để các lần Test sau dùng bộ nhớ
                        st.session_state[f"{deepvar_session_key}_model"] = deep_var_model

                    # Load best hyperparameters
                    st.subheader("Loaded best hyperparameters (training phase)")
                    # st.json(grid_search_data)

                    # 1) Best params table
                    best_params = grid_search_data.get("best_params", {})
                    if isinstance(best_params, dict) and best_params:
                        st.markdown("**Best params**")
                        df_params = pd.DataFrame(best_params.items(), columns=["Hyperparameter", "Value"])

                        # ÉP Value thành string
                        df_params["Value"] = df_params["Value"].astype(str)

                        st.dataframe(df_params, use_container_width=True)


                    # 2) Summary table
                    summary = {k: v for k, v in grid_search_data.items() if k != "best_params"}
                    if summary:
                        st.markdown("**Search summary**")
                        df_summary = pd.DataFrame(summary.items(), columns=["Metric", "Value"])

                        # ÉP Value thành string
                        df_summary["Value"] = df_summary["Value"].astype(str)

                        st.dataframe(df_summary, use_container_width=True)


                    # 3) Load VAR result to recover best_lag = look_back
                    with open(var_result_path, "rb") as f:
                        var_result = pickle.load(f)

                    look_back = best_lag
                    look_ahead = 1
                    k_model = var_result.neqs

                    # Basic shape check
                    if test.shape[1] != k_model:
                        st.error(
                            f"Incompatible test data: DEEPVAR model was trained with {k_model} variables, "
                            f"but test data has {test.shape[1]} columns.\n"
                            "Please ensure you use the same features as during training or retrain the DEEPVAR model."
                        )
                    else:
                        # PIPELINE (SCALED)
                        start_time = time.time()
                        X_test = create_windows(test, window_shape=look_back, end_id=-look_ahead)
                        y_test = create_windows(test, window_shape=look_ahead, start_id=look_back)
                        predictions = deep_var_model.predict(X_test)
                        execution_time = time.time() - start_time

                        # Eval (scaled) theo notebook
                        evaluation_df = evaluate_multivariate_forecast(
                            y_test,
                            predictions,
                            test.columns
                        )
                        evaluation_df_overall = evaluate_overall_forecast(
                            y_test,
                            predictions,
                            execution_time
                        )

                        # Chuẩn bị bảng hiển thị scaled (flatten để xem dễ)
                        y_test_scaled_flat = y_test.reshape(-1, y_test.shape[-1])
                        y_test_scaled_df = pd.DataFrame(
                            y_test_scaled_flat,
                            columns=test.columns
                        )
                        pred_scaled_flat = predictions.reshape(-1, predictions.shape[-1])
                        preds_scaled_df = pd.DataFrame(
                            pred_scaled_flat,
                            columns=test.columns
                        )

                        st.success(f"Inference completed in {execution_time:.2f} seconds.")

                        # Vẽ loss (from training phase)
                        with open(training_history_path, "rb") as f:
                            loaded_history = pickle.load(f)
                        if "val_loss" in loaded_history:
                            loss_df = pd.DataFrame({
                                "Epoch": range(1, len(loaded_history["loss"]) + 1),
                                "Training Loss": loaded_history["loss"],
                                "Validation Loss": loaded_history["val_loss"],
                            })
                        else:
                            loss_df = pd.DataFrame({
                                "Epoch": range(1, len(loaded_history["loss"]) + 1),
                                "Training Loss": loaded_history["loss"],
                            })
                        st.subheader("Training & Validation Loss (from training phase)")
                        plot_loss_curve(loss_df)

                        # Save scaled eval
                        evaluation_df.to_csv(
                            os.path.join(deepvar2_directory_evaluation, "DeepVAR_evaluation_scaled.csv"),
                            index=False
                        )
                        evaluation_df_overall.to_csv(
                            os.path.join(deepvar2_directory_evaluation, "DeepVAR_evaluation_overall_scaled.csv"),
                            index=False
                        )

                        # RESTORE VỀ ORIGINAL 

                        scaler    = st.session_state.get("normalization_scaler", None)
                        flag_diff = st.session_state.get("flag_diff", False)
                        lag       = st.session_state.get("lag", 0)

                        pred_restore = None
                        y_test_restore = None

                        if scaler is None:
                            st.warning("No scaler found. Restoration to original scale will be skipped.")
                        else:
                            # Chuẩn bị dữ liệu gốc
                            dataset_for_restore = preprocess_data(dataset_original)

                            if flag_diff:
                                id_pos = train_full.shape[0] + best_lag + look_back
                            else:
                                id_pos = train_full.shape[0] + best_lag + look_back - 1

                            original_segment = dataset_for_restore.iloc[(id_pos - 1):(id_pos + 1)]
                            original_segment = original_segment[target_columns].copy()

                            exclude_cols = None

                            # Restore predictions & ground truth 
                            pred_restore = inverse_transformation(
                                predictions,
                                scaler,
                                original_segment.columns,
                                original_segment,
                                lag,
                                exclude_cols,
                                flag_diff
                            )
                            y_test_restore = inverse_transformation(
                                y_test,
                                scaler,
                                original_segment.columns,
                                original_segment,
                                lag,
                                exclude_cols,
                                flag_diff
                            )

                            # Bảo đảm DF để hiển thị/plot
                            pred_restore_df = (
                                pred_restore
                                if isinstance(pred_restore, pd.DataFrame)
                                else pd.DataFrame(pred_restore, columns=original_segment.columns)
                            )
                            y_test_restore_df = (
                                y_test_restore
                                if isinstance(y_test_restore, pd.DataFrame)
                                else pd.DataFrame(y_test_restore, columns=original_segment.columns)
                            )

                            # Eval (original) theo notebook
                            evaluation_df_org = evaluate_multivariate_forecast(
                                y_test_restore,
                                pred_restore,
                                y_test_restore_df.columns
                            )
                            evaluation_df_overall_org = evaluate_overall_forecast_restore(
                                y_test_restore,
                                pred_restore,
                                execution_time
                            )

                            # Save original eval
                            evaluation_df_org.to_csv(
                                os.path.join(deepvar2_directory_evaluation, "DeepVAR_evaluation_org.csv"),
                                index=False
                            )
                            evaluation_df_overall_org.to_csv(
                                os.path.join(deepvar2_directory_evaluation, "DeepVAR_evaluation_overall_org.csv"),
                                index=False
                            )


                        has_original = (
                            (scaler is not None)
                            and (pred_restore is not None)
                            and (y_test_restore is not None)
                        )

                        if has_original:
                            # st.subheader("Data View (scaled vs original)")
                            with st.expander("View testing results (transformed data vs inverse transformed data)", expanded=True):
                                tab_scaled, tab_original = st.tabs(["Transformed data", "Inverse transformed data"])

                                # ----- TAB 1: SCALED -----
                                with tab_scaled:
                                    st.markdown("#### Tables")
                                    st.markdown("**Actual**")
                                    st.dataframe(y_test_scaled_df)
                                    st.markdown("**Predictions**")
                                    st.dataframe(preds_scaled_df)

                                    st.markdown("#### Plots")
                                    variable_names = list(test.columns)  # chỉ các cột còn lại
                                    for idx, name in enumerate(variable_names):
                                        st.subheader(f"{name}")
                                        plot_actual_vs_predicted_new(
                                            y_test,
                                            predictions,
                                            variable_index=idx,
                                            step_index=0,
                                            variable_name=name
                                        )

                                    st.markdown("#### Evaluation")
                                    st.dataframe(evaluation_df)
                                    st.dataframe(evaluation_df_overall)

                                # ----- TAB 2: ORIGINAL -----
                                with tab_original:
                                    st.markdown("#### Tables")
                                    st.markdown("**Actual restore**")
                                    st.dataframe(y_test_restore_df)
                                    st.markdown("**Predictions restore**")
                                    st.dataframe(pred_restore_df)

                                    st.markdown("#### Plots")
                                    variable_names_org = list(y_test_restore_df.columns)  # đúng cột sau restore
                                    for idx, name in enumerate(variable_names_org):
                                        st.subheader(f"{name}")
                                        plot_actual_vs_predicted_new(
                                            y_test_restore,
                                            pred_restore,
                                            variable_index=idx,
                                            step_index=0,
                                            variable_name=name
                                        )

                                    st.markdown("#### Evaluation")
                                    st.dataframe(evaluation_df_org)
                                    st.dataframe(evaluation_df_overall_org)
                        else:
                            # Không có scaler/original → chỉ hiển thị scaled một dạng duy nhất
                            # st.subheader("Data View")
                            with st.expander("View results", expanded=True):
                                st.markdown("#### Tables")
                                st.markdown("**Actual**")
                                st.dataframe(y_test_scaled_df)
                                st.markdown("**Predictions**")
                                st.dataframe(preds_scaled_df)

                                st.markdown("#### Plots")
                                variable_names = list(test.columns)
                                for idx, name in enumerate(variable_names):
                                    st.subheader(f"{name}")
                                    plot_actual_vs_predicted_new(
                                        y_test,
                                        predictions,
                                        variable_index=idx,
                                        step_index=0,
                                        variable_name=name
                                    )

                                st.markdown("#### Evaluation")
                                st.dataframe(evaluation_df)
                                st.dataframe(evaluation_df_overall)

                
        elif model_type == "VAR-LSTM":

            # Tạo key
            varlstm_session_key = f"VARLSTM_{file_name}"

            # Tạo đường dẫn folder: model_type / file_name
            varlstm_directory_results    = f"results/{model_type}/{file_name}"
            varlstm_directory_models     = f"models/{model_type}/{file_name}"
            varlstm_directory_evaluation = f"evaluation/{model_type}/{file_name}"
            varlstm_directory_bestlag    = f"best_lag/{model_type}/{file_name}"

            # Tạo folders
            os.makedirs(varlstm_directory_results, exist_ok=True)
            os.makedirs(varlstm_directory_models, exist_ok=True)
            os.makedirs(varlstm_directory_evaluation, exist_ok=True)
            os.makedirs(varlstm_directory_bestlag, exist_ok=True)

            # 7. Training

            # Hiển thị thông báo ở trung tâm màn hình thay vì sidebar
            status_placeholder = st.empty()
            stop_training = threading.Event()

            class StopTrainingCallback(tf.keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    if stop_training.is_set():
                        self.model.stop_training = True
                        st.warning(f"Training stopped at epoch {epoch + 1}")

            if stop_button:
                stop_training.set()
                st.warning("Stopping training process...")

            # Nhấn Train
            if train_button and not stop_training.is_set():
                stop_training.clear()

                spinner_html = '''
                    <div style="display: flex; align-items: center;">
                        <div class="loader"></div>
                        <p style="margin-left: 10px; font-size: 30px">Optimizing parameters...</p>
                    </div>
                    <style>
                        .loader {
                            border: 4px solid #f3f3f3;
                            border-top: 4px solid #3498db;
                            border-radius: 50%;
                            width: 55px; height: 55px;
                            animation: spin 1s linear infinite;
                        }

                        @keyframes spin {
                            0% { transform: rotate(0deg); }
                            100% { transform: rotate(360deg); }
                        }
                    </style>
                    '''

                # Hiển thị spinner custom
                status_placeholder.markdown(spinner_html, unsafe_allow_html=True)

                st.write("### Model training results")
                status_placeholder.write("### Optimizing parameters...")

                var_result_path = os.path.join(varlstm_directory_results, "var_result.pkl")
                best_lag_path   = os.path.join(varlstm_directory_bestlag, "best_lag.csv")

                # Tìm best lag
                result_df_org_data, best_lag = find_bestlag(train_full, 31)

                # Lưu toàn bộ bảng AIC/lag 
                save_path = f"best_lag/{model_type}/{file_name}/best_lag.csv"
                result_df_org_data.to_csv(save_path, index=False)

                st.success(f"**Best lag:** {best_lag}")

                # Train VAR với best_lag
                var = VAR(endog=train_full.values)
                var_result = var.fit(maxlags=best_lag)

                st.success(f"Trained VAR model with best_lag = {best_lag}.")

                train_var_pred = create_var_predictions(
                    train_final, var_result, var_result.k_ar, dataset.columns
                )
                val_var_pred = create_var_predictions(
                    val, var_result, var_result.k_ar, dataset.columns
                )

                # st.write("Best lag (k_ar):", var_result.k_ar)

            ## Tìm lag tốt nhất cho tập train_var_pred
                # Tìm lag tốt nhất cho tập train_var_pred
                column_names = test.columns
                train_var_pred_df = pd.DataFrame(train_var_pred, columns=column_names)

                best_lag_var_pred_path = f"best_lag/{model_type}/{file_name}/best_lag_var_pred.csv"

                st.info("Computing best lag (AIC table) for VAR prediction...")

                result_df_var_pred, best_lag_var_pred = find_bestlag(train_var_pred_df, 31)

                # Đảm bảo thư mục tồn tại rồi mới lưu
                os.makedirs(os.path.dirname(best_lag_var_pred_path), exist_ok=True)
                result_df_var_pred.to_csv(best_lag_var_pred_path, index=False)

                st.success(f"Best lag for VAR predictions (best_lag_var_pred) = {best_lag_var_pred}")

                # Cập nhật status ở trên cùng: mô hình đang tìm kiếm tham số tối ưu
                local_status = st.empty()

                local_status.markdown(
                    """
                    <div style="
                        background-color:#FFF8E1;
                        border-left:5px solid #F9A825;
                        padding:10px;
                        border-radius:6px;
                        margin-top:10px;
                    ">
                        <b style="color:#E65100;">Model is searching for optimal parameters...</b><br>
                        <span style="color:#E65100;">
                            Please wait while the model performs hyperparameter optimization.
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                ## LSTM PROCESS

                # Chuyển đổi dữ liệu thành window
                look_back = best_lag_var_pred
                look_ahead = 1

                X_train = create_windows(train_var_pred, window_shape=look_back, end_id=-look_ahead)
                y_train = create_windows(train_final.values[var_result.k_ar :], window_shape=look_ahead, start_id=look_back)

                X_val = create_windows(val_var_pred, window_shape=look_back, end_id=-look_ahead)
                y_val = create_windows(val.values[var_result.k_ar :], window_shape=look_ahead, start_id=look_back)

                print(X_train.shape, y_train.shape)
                print(X_val.shape, y_val.shape)

                input_dim = X_train.shape[2]                           
                output_dim = y_train.shape[2]    

                # Tìm tham số tối ưu - Multiphase tunning
                # Phase 1-Coarse tuning: Tìm vùng tham số tốt với 3 giá trị thấp - trung bình - cao
                param_grid_phase1 = {
                    "learning_rate": [3e-4, 7e-4, 3e-3],
                    "batch_size": [64, 128, 256],
                    "units_lstm": [32, 96, 256],  # số nơ ron mạng LSTM
                    "epoch": [300],
                    "dropout": [0.0],
                    "L2_reg": [0.0],
                }

                best_params_phase1, best_mse_phase1, search_time_phase1 = grid_search(
                    input_dim,
                    output_dim,
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    param_grid_phase1,
                    look_back,
                    look_ahead
                )

                result_phase1 = {
                    "best_params": best_params_phase1,
                    "look_back": look_back,
                    "best_lag_VAR": int(best_lag),       # ép kiểu int để tránh lỗi JSON
                    "best_MSE": float(best_mse_phase1),        # lưu MSE
                    "search_time_seconds": float(search_time_phase1)  # lưu thời gian
                }
                with open(f"{varlstm_directory_results}/VAR_LSTM_grid_search_results_phase1.json", "w") as f:
                    json.dump(result_phase1, f)

                # Phase 2-Fine Tuning: Thu hẹp phạm vi tìm tham số tối ưu từ khoảng tốt nhất đã xác định
                param_grid_phase2 = {
                    "learning_rate": get_fine_values(best_params_phase1['learning_rate'], [3e-4, 5e-4, 7e-4, 1e-3, 3e-3]),
                    "batch_size": [best_params_phase1['batch_size']],  # giữ nguyên
                    "units_lstm": get_fine_values(best_params_phase1['units_lstm'], [32, 64, 96, 128, 256]),
                    "epoch": [300],
                    "dropout": [0.0, 0.05, 0.1],
                    "L2_reg": [0.0, 1e-6, 1e-5, 1e-4],
                    
                }

                best_params_phase2, best_mse_phase2, search_time_phase2 = grid_search(
                    input_dim,
                    output_dim,
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    param_grid_phase2,
                    look_back,
                    look_ahead
                )

                best_params = best_params_phase2
                best_mse = best_mse_phase2
                search_time = search_time_phase1 + search_time_phase2

                # Gom tất cả thông tin vào 1 dict
                results_to_save = {
                    "best_params": best_params,
                    "look_back": look_back,
                    "best_lag_VAR": int(best_lag),       # ép kiểu int để tránh lỗi JSON
                    "best_MSE": float(best_mse),        # lưu MSE
                    "search_time_seconds": float(search_time)  # lưu thời gian
                }
                # Lưu ra file JSON
                with open(f"{varlstm_directory_results}/VAR_LSTM_grid_search_results.json", "w") as f:
                    json.dump(results_to_save, f, indent=4)

                # Lưu VAR result
                with open(var_result_path, "wb") as f:
                    pickle.dump(var_result, f)

                # Lưu thêm danh sách tên cột để dùng khi predict
                trained_columns_path = os.path.join(varlstm_directory_results, "trained_columns.json")
                with open(trained_columns_path, "w", encoding="utf-8") as f:
                    json.dump(list(train_full.columns), f, ensure_ascii=False)

                # st.success("Parameter optimization completed.")
                st.write("**Best Parameters:**")
                st.json(best_params)
                st.write(f"**Best MSE:** {best_mse:.4f}")
                st.write(f"**Search Time:** {search_time:.2f} seconds")
                st.write(f"**Best Lags (VAR):** {best_lag}")
                st.write(f"**Look back:** {look_back}")

                # Cập nhật status: tối ưu tham số xong, chuẩn bị train VAR-LSTM
                local_status.markdown(
                    """
                    <div style="
                        background-color:#E3F2FD;
                        border-left:5px solid #1E88E5;
                        padding:10px;
                        border-radius:6px;
                        margin-top:10px;
                    ">
                        <b style="color:#0D47A1;">VAR-LSTM parameter optimization completed.</b><br>
                        <span style="color:#0D47A1;">
                            Best hyperparameters have been found. Proceeding to train the VAR-LSTM model...
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Toast nhỏ báo optimize xong
                st.toast("VAR-LSTM parameter optimization finished. Starting training...")

                # Training
                final_model = build_lstm(
                                    input_dim,
                                    output_dim,
                                    look_back = look_back,
                                    look_ahead = look_ahead,
                                    lr = best_params["learning_rate"],
                                    units_lstm = best_params["units_lstm"],
                                    dropout = best_params["dropout"],
                                    L2_reg = best_params["L2_reg"]
                                )

                start_time = time.time()
                callbacks = make_callbacks_for_epochs(best_params["epoch"])
                history = final_model.fit(
                    X_train, y_train,
                    validation_data = (X_val, y_val),
                        epochs=best_params["epoch"],
                        batch_size=best_params["batch_size"],
                        verbose=1,
                        callbacks=callbacks
                        )
                end_time = time.time()

                train_time = end_time - start_time
                st.write(f"Training time: {train_time:.2f} seconds")
                total_train_time = train_time + search_time
                st.write(f"Total time to find optimal parameters and training time: {total_train_time:.2f} seconds")

                final_model.save(
                    f"models/{model_type}/{file_name}/VAR_LSTM_final_model.keras"
                )
                st.write ("Training completed and model saved.")
                with open(
                                        f"{varlstm_directory_results}/VAR_LSTM_training_history.pkl", "wb"
                                    ) as f:
                                        pickle.dump(history.history, f)

                # st.success("Training completed and model saved.")

                # Lưu model & meta vào session để TEST dùng
                st.session_state[f"{varlstm_session_key}_model"] = final_model
                st.session_state[f"{varlstm_session_key}_best_params"] = best_params
                st.session_state[f"{varlstm_session_key}_look_back"] = look_back
                st.session_state[f"{varlstm_session_key}_look_ahead"] = look_ahead

                # Tạo DataFrame chứa loss theo từng epoch
                loss_df = pd.DataFrame(
                    {
                        "Epoch": range(1, len(history.history["loss"]) + 1),
                        "Training Loss": history.history["loss"],
                        "Validation Loss": history.history["val_loss"],
                    }
                )

                # Hàm plot_loss_curve
                plot_loss_curve(loss_df)

                # Thông báo tổng kết training VAR-LSTM
                st.success(
                    f"VAR-LSTM training completed successfully!"
                )

                # Cập nhật status lớn ở trên cùng
                local_status.markdown(
                    f"""
                    <div style="
                        background-color:#E8F5E9;
                        border-left:5px solid #43A047;
                        padding:10px;
                        border-radius:6px;
                        margin-top:10px;
                    ">
                        <b style="color:#1B5E20;">VAR-LSTM pipeline completed for dataset: <code>{file_name}</code></b><br>
                        <span style="color:#1B5E20;">
                            Model has been trained, saved to disk, and stored in memory for immediate testing.
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.session_state["manual_stop_requested"] = False
                # Toast nhỏ báo train xong
                st.toast("VAR-LSTM model trained and saved successfully!")

            # Sau khi nhấn Stop, hiển thị kết quả đã lưu (nếu có)
            if stop_training.is_set():

                st.subheader(f"Last saved {file_name} model training results")
                st.session_state["manual_stop_requested"] = True  

                grid_search_path = (
                    f"{varlstm_directory_results}/VAR_LSTM_grid_search_results.json"
                )
                training_history_path = (
                    f"{varlstm_directory_results}/VAR_LSTM_training_history.pkl"
                )
                var_result_path = os.path.join(varlstm_directory_results, "var_result.pkl")

                # === KIỂM TRA FILE ===
                if not os.path.exists(grid_search_path) or not os.path.exists(
                    training_history_path
                ):
                    st.error(
                        "The complete VAR-LSTM result files could not be found. Please train the model first."
                    )
                else:
                    # ===== LOAD GRID SEARCH RESULT =====
                    with open(grid_search_path, "r") as f:
                        grid_search_data = json.load(f)

                    # --- 1. Nếu đã có VAR train từ trước, load lên dùng ngay ---
                    if os.path.exists(var_result_path) and os.path.getsize(var_result_path) > 0:
                        try:
                            with open(var_result_path, "rb") as f:
                                var_result = pickle.load(f)
                            best_lag = var_result.k_ar

                            # st.info(f"Best_lag = {best_lag}.")
                        except Exception as e:
                            st.warning("Found VAR model but failed to load it. Recomputing VAR...")
                            var_result = None
                            best_lag = None

                    # Chỉ show note Apply for... khi người dùng bấm Stop Training
                    manual_stop = st.session_state.get("manual_stop_requested", False)

                    if manual_stop:
                        show_last_saved_training_results_box(
                            model_type=model_type,
                            file_name=file_name,
                            note_if_missing_cols=(
                                f"Apply for: {file_name} weather dataset with all columns "
                                "after first-order differencing (non-stationary) and min-max normalization. "
                                "No data augmentation techniques are performed."
                            ),
                        )
                    st.write("#### Hyperparameters")
                    # st.json(grid_search_data)
                    # st.info(f"**Best_lag =** **{best_lag}.**")


                    # 1) Best params table
                    best_params = grid_search_data.get("best_params", {})
                    if isinstance(best_params, dict) and best_params:
                        st.markdown("**Best params**")
                        df_params = pd.DataFrame(best_params.items(), columns=["Hyperparameter", "Value"])

                        # ÉP Value thành string
                        df_params["Value"] = df_params["Value"].astype(str)

                        st.dataframe(df_params, use_container_width=True)


                    # 2) Summary table
                    summary = {k: v for k, v in grid_search_data.items() if k != "best_params"}
                    if summary:
                        st.markdown("**Search summary**")
                        df_summary = pd.DataFrame(summary.items(), columns=["Metric", "Value"])

                        # ÉP Value thành string
                        df_summary["Value"] = df_summary["Value"].astype(str)

                        st.dataframe(df_summary, use_container_width=True)

                    # ===== LOAD TRAINING HISTORY =====
                    with open(training_history_path, "rb") as f:
                        loaded_history = pickle.load(f)

                    # ===== FORMAT LOSS DATAFRAME =====
                    if "val_loss" in loaded_history:
                        loss_df = pd.DataFrame(
                            {
                                "Epoch": range(1, len(loaded_history["loss"]) + 1),
                                "Training Loss": loaded_history["loss"],
                                "Validation Loss": loaded_history["val_loss"],
                            }
                        )
                    else:
                        loss_df = pd.DataFrame(
                            {
                                "Epoch": range(1, len(loaded_history["loss"]) + 1),
                                "Training Loss": loaded_history["loss"],
                            }
                        )

                    # ===== VẼ BIỂU ĐỒ LOSS =====
                    st.write("#### Training & Validation Loss (Last Saved)")
                    st.line_chart(loss_df.set_index("Epoch"))

                    # ===== HIỂN THỊ LOSS CUỐI CÙNG =====
                    st.write("#### Final Losses")
                    st.write(
                        f"**Training Loss (final)**: {loaded_history['loss'][-1]:.4f}"
                    )
                    if "val_loss" in loaded_history:
                        st.write(
                            f"**Validation Loss (final)**: {loaded_history['val_loss'][-1]:.4f}"
                        )

            # ====== TESTING PHASE ======
            if test_button:

                st.write("### VAR-LSTM Model Testing Results")

                # Paths to saved artifacts from TRAIN
                grid_search_path = (
                    f"results/{model_type}/{file_name}/VAR_LSTM_grid_search_results.json"
                )
                model_path = (
                    f"models/{model_type}/{file_name}/VAR_LSTM_final_model.keras"
                )
                var_result_path = (
                    f"results/{model_type}/{file_name}/var_result.pkl"
                )
                training_history_path = (
                    f"results/{model_type}/{file_name}/VAR_LSTM_training_history.pkl"
                )
                lag_file_path = (f"best_lag/{model_type}/{file_name}/best_lag_var_pred.csv")

                # Ensure evaluation directory exists
                os.makedirs(varlstm_directory_evaluation, exist_ok=True)

                # Check all required files
                if (
                    not os.path.exists(grid_search_path)
                    or not os.path.exists(model_path)
                    or not os.path.exists(var_result_path)
                    or not os.path.exists(training_history_path)
                ):
                    st.error(
                        "The complete VAR-LSTM result files could not be found. "
                        "Please train the VAR-LSTM model first."
                    )
                else:
                    # 1. Load grid search info (for display only)
                    with open(grid_search_path, "r") as f:
                        grid_search_data = json.load(f)

                    # 2. Load trained VAR-LSTM model
                    var_lstm_model = st.session_state.get(f"{varlstm_session_key}_model", None)
                    model_source = "memory"

                    if var_lstm_model is None:
                        model_source = "disk"

                        if (not os.path.exists(model_path)) or os.path.getsize(model_path) == 0:
                            st.error("VAR-LSTM model file not found or is empty. Please train the VAR-LSTM model first.")
                            st.stop()

                        var_lstm_model = load_model(model_path)
                        # Lưu lại vào session cho những lần test sau
                        st.session_state[f"{varlstm_session_key}_model"] = var_lstm_model

                    # Load hyperparameters
                    st.subheader("Loaded best hyperparameters (training phase)")
                    # st.json(grid_search_data)


                    # 1) Best params table
                    best_params = grid_search_data.get("best_params", {})
                    if isinstance(best_params, dict) and best_params:
                        st.markdown("**Best params**")
                        df_params = pd.DataFrame(best_params.items(), columns=["Hyperparameter", "Value"])

                        # ÉP Value thành string
                        df_params["Value"] = df_params["Value"].astype(str)

                        st.dataframe(df_params, use_container_width=True)


                    # 2) Summary table
                    summary = {k: v for k, v in grid_search_data.items() if k != "best_params"}
                    if summary:
                        st.markdown("**Search summary**")
                        df_summary = pd.DataFrame(summary.items(), columns=["Metric", "Value"])

                        # ÉP Value thành string
                        df_summary["Value"] = df_summary["Value"].astype(str)

                        st.dataframe(df_summary, use_container_width=True)

                    # 3. Load VAR result to recover best_lag = look_back
                    with open(var_result_path, "rb") as f:
                        var_result = pickle.load(f)

                    if os.path.exists(lag_file_path):
                        df_lag = pd.read_csv(lag_file_path)

                        # Read best_lag_var_pred directly from the file
                        if "AIC" in df_lag.columns and "p" in df_lag.columns:
                            idx_min = df_lag["AIC"].astype(float).idxmin()
                            best_lag_var_pred = int(df_lag.loc[idx_min, "p"])
                            st.info(f"Loaded existing AIC table. best_lag_var_pred = {best_lag_var_pred}")
                        else:
                            st.warning("Existing CSV missing columns 'p' or 'AIC'. Recomputing...")
                            best_lag_var_pred = None

                    best_lag = var_result.k_ar      
                    look_back = best_lag_var_pred
                    look_ahead = 1                  
                    k_model = var_result.neqs       # number of equations = number of variables

                    # Basic shape check
                    if test.shape[1] != k_model:
                        st.error(
                            f"Incompatible test data: VAR-LSTM model was trained with {k_model} variables, "
                            f"but test data has {test.shape[1]} columns.\n"
                            "Please ensure you use the same features as during training or retrain the VAR-LSTM model."
                        )
                    else: 
                        start_time = time.time()

                        test_var_pred = create_var_predictions(
                            test,           
                            var_result,
                            best_lag,
                            test.columns
                        )

                        X_test = create_windows(
                            test_var_pred,
                            window_shape=look_back,
                            end_id=-look_ahead
                        )

                        y_test = create_windows(
                            test.values[var_result.k_ar :],
                            window_shape=look_ahead,
                            start_id=look_back
                        )

                        # ===== 5) PREDICT =====
                        predictions = var_lstm_model.predict(X_test)
                        execution_time = time.time() - start_time

                        # Chuẩn bị dữ liệu để xem ở "scaled" (flatten 3D -> 2D để show bảng/plot)
                        y_test_scaled_flat = y_test.reshape(-1, y_test.shape[-1])
                        y_test_scaled_df = pd.DataFrame(
                            y_test_scaled_flat,
                            columns=test.columns
                        )

                        pred_scaled_flat = predictions.reshape(-1, predictions.shape[-1])
                        preds_scaled_df = pd.DataFrame(
                            pred_scaled_flat,
                            columns=test.columns
                        )

                        st.success(f"Inference completed in {execution_time:.2f} seconds.")

                        # ===== 6) VẼ LOSS TRAIN (NHƯ CŨ) =====
                        with open(training_history_path, "rb") as f:
                            loaded_history = pickle.load(f)

                        if "val_loss" in loaded_history:
                            loss_df = pd.DataFrame({
                                "Epoch": range(1, len(loaded_history["loss"]) + 1),
                                "Training Loss": loaded_history["loss"],
                                "Validation Loss": loaded_history["val_loss"],
                            })
                        else:
                            loss_df = pd.DataFrame({
                                "Epoch": range(1, len(loaded_history["loss"]) + 1),
                                "Training Loss": loaded_history["loss"],
                            })

                        st.subheader("Training & Validation Loss (from training phase)")
                        plot_loss_curve(loss_df)

                        # ===== 8) RESTORE VỀ THANG GỐC =====
                        # Lấy các biến/hyper quan trọng từ session_state
                        scaler    = st.session_state.get("normalization_scaler", None)
                        flag_diff = st.session_state.get("flag_diff", False)
                        lag       = st.session_state.get("lag", 0)

                        target_columns = st.session_state.get("target_columns", list(test.columns))

                        pred_restore_df = None
                        y_test_restore_df = None

                        if scaler is None:
                            # Không có scaler -> KHÔNG hoàn nguyên, chỉ đánh giá trên dữ liệu hiện tại
                            st.warning(
                                "No scaler found. Restoration to original scale will be skipped. "
                                "Evaluation will be done directly on the model outputs."
                            )
                        else:
                            # st.subheader("Restoring predictions and ground truth to original scale")

                            # Chuẩn bị chuỗi gốc đã preprocess
                            dataset_for_restore = preprocess_data(dataset_original)

                            # TÍNH CHỈ SỐ CẮT SEGMENT
                            train_len = train_full.shape[0]  

                            if flag_diff:
                                id_restore = train_len + best_lag + look_back
                            else:
                                id_restore = train_len + best_lag + look_back - 1

                            original_segment = dataset_for_restore.iloc[(id_restore - 1):(id_restore + 1)]
                            # Lọc theo target_columns
                            original_segment = original_segment[target_columns].copy()

                            exclude_cols = None

                            # --- Restore predictions ---
                            pred_restore = inverse_transformation(
                                predictions,                 # scaled preds (shape: [n_samples, look_ahead, n_vars'])
                                scaler,
                                original_segment.columns,    # đúng bộ cột sau khi drop
                                original_segment,
                                lag,
                                exclude_cols,
                                flag_diff,
                            )

                            # --- Restore ground truth ---
                            y_test_restore = inverse_transformation(
                                y_test,                      # scaled y_test windows
                                scaler,
                                original_segment.columns,
                                original_segment,
                                lag,
                                exclude_cols,
                                flag_diff,
                            )

                            # Bảo đảm DataFrame để hiển thị & evaluate
                            pred_restore_df = (
                                pred_restore
                                if isinstance(pred_restore, pd.DataFrame)
                                else pd.DataFrame(pred_restore, columns=original_segment.columns)
                            )
                            y_test_restore_df = (
                                y_test_restore
                                if isinstance(y_test_restore, pd.DataFrame)
                                else pd.DataFrame(y_test_restore, columns=original_segment.columns)
                            )

                        # Cờ để biết có thể hiển thị được bản original hay không
                        has_scaler_and_restore = (
                            (scaler is not None)
                            and (y_test_restore_df is not None)
                            and (pred_restore_df is not None)
                        )

                        # ===== 9 + 10) VIEW & EVALUATION =====

                        # --- EVAL (scaled) dùng 3D y_test & predictions ---
                        evaluation_df_scaled = evaluate_multivariate_forecast(
                            y_test,
                            predictions,
                            test.columns
                        )
                        evaluation_df_overall_scaled = evaluate_overall_forecast(
                            y_test,
                            predictions,
                            execution_time
                        )

                        # Chuẩn bị bảng hiển thị scaled (flatten 3D -> 2D)
                        y_test_scaled_flat = y_test.reshape(-1, y_test.shape[-1])
                        y_test_scaled_df = pd.DataFrame(y_test_scaled_flat, columns=test.columns)

                        pred_scaled_flat = predictions.reshape(-1, predictions.shape[-1])
                        preds_scaled_df = pd.DataFrame(pred_scaled_flat, columns=test.columns)

                        # Lưu scaled eval 
                        eval_scaled_path = os.path.join(varlstm_directory_evaluation, "VAR_LSTM_evaluation_scaled.csv")
                        eval_overall_scaled_path = os.path.join(varlstm_directory_evaluation, "VAR_LSTM_evaluation_overall_scaled.csv")
                        evaluation_df_scaled.to_csv(eval_scaled_path, index=False)
                        evaluation_df_overall_scaled.to_csv(eval_overall_scaled_path, index=False)

                        # Cờ để biết có thể hiển thị bản original hay không
                        has_original = (
                            (scaler is not None)
                            and (pred_restore_df is not None)
                            and (y_test_restore_df is not None)
                        )

                        if has_original:
                            # Eval (original) theo đúng kiểu DeepVAR:
                            # - Nếu bạn muốn đánh giá trên đúng "mảng" restore (pred_restore / y_test_restore) thì dùng chúng
                            # - Nếu inverse_transformation trả về DataFrame thì y_test_restore_df/pred_restore_df vẫn ok
                            evaluation_df_org = evaluate_multivariate_forecast(
                                y_test_restore,
                                pred_restore,
                                y_test_restore_df.columns
                            )
                            evaluation_df_overall_org = evaluate_overall_forecast_restore(
                                y_test_restore,
                                pred_restore,
                                execution_time
                            )

                            eval_org_path = os.path.join(varlstm_directory_evaluation, "VAR_LSTM_evaluation_org.csv")
                            eval_overall_org_path = os.path.join(varlstm_directory_evaluation, "VAR_LSTM_evaluation_overall_org.csv")
                            evaluation_df_org.to_csv(eval_org_path, index=False)
                            evaluation_df_overall_org.to_csv(eval_overall_org_path, index=False)

                            # ===== VIEW: Expander + Tabs + Tables/Plots/Eval =====
                            with st.expander("View testing results (transformed data vs inverse transformed data)", expanded=True):
                                tab_scaled, tab_original = st.tabs(["Transformed data", "Inverse transformed data"])

                                # ----- TAB 1: SCALED -----
                                with tab_scaled:
                                    st.markdown("#### Tables")
                                    st.markdown("**Actual**")
                                    st.dataframe(y_test_scaled_df)
                                    st.markdown("**Predictions**")
                                    st.dataframe(preds_scaled_df)

                                    st.markdown("#### Plots")
                                    variable_names = list(test.columns)
                                    for idx, name in enumerate(variable_names):
                                        st.subheader(f"{name}")
                                        plot_actual_vs_predicted_new(
                                            y_test,
                                            predictions,
                                            variable_index=idx,
                                            step_index=0,
                                            variable_name=name
                                        )

                                    st.markdown("#### Evaluation")
                                    st.dataframe(evaluation_df_scaled)
                                    st.dataframe(evaluation_df_overall_scaled)

                                # ----- TAB 2: ORIGINAL -----
                                with tab_original:
                                    st.markdown("#### Tables")
                                    st.markdown("**Actual restore**")
                                    st.dataframe(y_test_restore_df)
                                    st.markdown("**Prediction restore**")
                                    st.dataframe(pred_restore_df)

                                    st.markdown("#### Plots")
                                    variable_names_org = list(y_test_restore_df.columns)
                                    for idx, name in enumerate(variable_names_org):
                                        st.subheader(f"{name}")
                                        plot_actual_vs_predicted_new(
                                            y_test_restore,
                                            pred_restore,
                                            variable_index=idx,
                                            step_index=0,
                                            variable_name=name
                                        )

                                    st.markdown("#### Evaluation")
                                    st.dataframe(evaluation_df_org)
                                    st.dataframe(evaluation_df_overall_org)

                        else:
                            # Không có scaler/original → giống DeepVAR: 1 view duy nhất
                            with st.expander("View results", expanded=True):
                                st.markdown("#### Tables")
                                st.markdown("**Actual**")
                                st.dataframe(y_test_scaled_df)
                                st.markdown("**Predictions**")
                                st.dataframe(preds_scaled_df)

                                st.markdown("#### Plots")
                                variable_names = list(test.columns)
                                for idx, name in enumerate(variable_names):
                                    st.subheader(f"{name}")
                                    plot_actual_vs_predicted_new(
                                        y_test,
                                        predictions,
                                        variable_index=idx,
                                        step_index=0,
                                        variable_name=name
                                    )

                                st.markdown("#### Evaluation")
                                st.dataframe(evaluation_df_scaled)
                                st.dataframe(evaluation_df_overall_scaled)


        else:

            # Tạo key
            vardeepvar_session_key = f"VAR_LAI_DEEPVAR_{file_name}"

            # Tạo đường dẫn folder: model_type / file_name
            vardeepvar_directory_results    = f"results/{model_type}/{file_name}"
            vardeepvar_directory_models     = f"models/{model_type}/{file_name}"
            vardeepvar_directory_evaluation = f"evaluation/{model_type}/{file_name}"
            vardeepvar_directory_bestlag    = f"best_lag/{model_type}/{file_name}"

            # Tạo folders
            os.makedirs(vardeepvar_directory_results, exist_ok=True)
            os.makedirs(vardeepvar_directory_models, exist_ok=True)
            os.makedirs(vardeepvar_directory_evaluation, exist_ok=True)
            os.makedirs(vardeepvar_directory_bestlag, exist_ok=True)

            # 7. Training

            # Hiển thị thông báo ở trung tâm màn hình thay vì sidebar
            status_placeholder = st.empty()
            stop_training = threading.Event()

            # Callback to Stop Training (giữ nguyên thuật toán, chỉ phục vụ UI nếu dùng)
            class StopTrainingCallback(tf.keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    if stop_training.is_set():
                        self.model.stop_training = True
                        st.warning(f"Training stopped at epoch {epoch + 1}")

            # Nhấn nút Stop → set cờ dừng
            if stop_button:
                stop_training.set()
                st.warning("Stopping training process...")

            if train_button and not stop_training.is_set():
                stop_training.clear()
                
                spinner_html = '''
                    <div style="display: flex; align-items: center;">
                        <div class="loader"></div>
                        <p style="margin-left: 10px; font-size: 30px">Optimizing parameters...</p>
                    </div>
                    <style>
                        .loader {
                            border: 4px solid #f3f3f3;
                            border-top: 4px solid #3498db;
                            border-radius: 50%;
                            width: 55px; height: 55px;
                            animation: spin 1s linear infinite;
                        }

                        @keyframes spin {
                            0% { transform: rotate(0deg); }
                            100% { transform: rotate(360deg); }
                        }
                    </style>
                    '''

                status_placeholder.markdown(spinner_html, unsafe_allow_html=True)

                st.write("### Model training results")

                status_placeholder.write("### Optimizing parameters...")

                # Train VAR
                var_result_path = os.path.join(vardeepvar_directory_results, "var_result.pkl")
                best_lag_path   = os.path.join(vardeepvar_directory_bestlag, "best_lag.csv")

                # st.info("Training VAR model to determine best lag (no reuse of previous VAR).")

                # Tìm best lag
                result_df_org_data, best_lag = find_bestlag(train_full, 31)

                # Lưu toàn bộ bảng AIC/lag 
                save_path = f"best_lag/{model_type}/{file_name}/best_lag.csv"
                result_df_org_data.to_csv(save_path, index=False)

                st.success(f"**Best lag:** {best_lag}")

                # Train VAR với best_lag
                var = VAR(endog=train_full.values)
                var_result = var.fit(maxlags=best_lag)

                st.success(f"Trained VAR model with best_lag = {best_lag}.")

                train_var_pred = create_var_predictions(
                                    train_final, var_result, var_result.k_ar, dataset.columns
                                )
                val_var_pred = create_var_predictions(
                                    val, var_result, var_result.k_ar, dataset.columns
                                )
                                
                k_ar = var_result.k_ar

                # Đưa pred về DataFrame để trừ an toàn + align theo index/columns
                train_var_pred_df = pd.DataFrame(
                    train_var_pred,
                    columns=train_final.columns,
                    index=train_final.index[k_ar:]
                )
                val_var_pred_df = pd.DataFrame(
                    val_var_pred,
                    columns=val.columns,
                    index=val.index[k_ar:]
                )

                # Actual lấy từ đúng tập đã dùng để dự báo (train_final và val)
                train_actual = train_final.iloc[k_ar:]
                val_actual   = val.iloc[k_ar:]

                print(f"Train actual shape: {train_actual.shape}, Train VAR pred shape: {train_var_pred.shape}")
                print(f"Val actual shape: {val_actual.shape}, Val VAR pred shape: {val_var_pred.shape}")

                # Error = actual - pred
                train_var_error = train_actual - train_var_pred_df
                val_var_error   = val_actual   - val_var_pred_df
                
                # Xác định khoảng index tương ứng
                # train_start_idx = 0 + var_result.k_ar 
                # train_end_idx = train_final.shape[0] - 1
                # val_start_idx = train_final.shape[0] + var_result.k_ar
                # val_end_idx = train_full.shape[0] - 1

                # Lấy giá trị thực tế tương ứng với khoảng dự báo
                # train_actual = dataset.iloc[train_start_idx:train_end_idx + 1]
                # val_actual = dataset.iloc[val_start_idx:val_end_idx + 1]

                # Kiểm tra xem index có khớp không
                # st.write(f"Train actual shape: {train_actual.shape}, Train VAR pred shape: {train_var_pred.shape}")
                # st.write(f"Val actual shape: {val_actual.shape}, Val VAR pred shape: {val_var_pred.shape}")

                # Tính lỗi error = giá trị thực tế - giá trị dự đoán từ VAR
                # train_var_error = train_actual - train_var_pred
                # val_var_error = val_actual - val_var_pred
                # st.write("train_var_error shape:", train_var_error.shape)
                # st.write("val_var_error shape:", val_var_error.shape)

                # Chuyển đổi dữ liệu thành window
                look_back = 1
                look_ahead = 1

                X_train = create_windows(train_var_error, window_shape=look_back, end_id=-look_ahead)
                y_train = create_windows(train_var_error, window_shape=look_ahead, start_id=look_back)

                X_val = create_windows(val_var_error, window_shape=look_back, end_id=-look_ahead)
                y_val = create_windows(val_var_error, window_shape=look_ahead, start_id=look_back)

                input_dim = X_train.shape[2]                           
                output_dim = y_train.shape[2]    

                # Cập nhật status ở trên cùng: mô hình đang tìm kiếm tham số tối ưu
                local_status = st.empty()

                local_status.markdown(
                    """
                    <div style="
                        background-color:#FFF8E1;
                        border-left:5px solid #F9A825;
                        padding:10px;
                        border-radius:6px;
                        margin-top:10px;
                    ">
                        <b style="color:#E65100;">Model is searching for optimal parameters...</b><br>
                        <span style="color:#E65100;">
                            Please wait while the model performs hyperparameter optimization.
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Tìm tham số tối ưu - Multiphase tunning
                # Phase 1-Coarse tuning: Tìm vùng tham số tốt với 3 giá trị thấp - trung bình - cao
                param_grid_phase1 = {
                    "learning_rate": [3e-4, 7e-4, 3e-3],
                    "batch_size": [64, 128, 256],
                    "units_lstm": [32, 96, 256],  # số nơ ron mạng LSTM
                    "epoch": [300],
                    "dropout": [0.0],
                    "L2_reg": [0.0],
                }

                best_params_phase1, best_mse_phase1, search_time_phase1 = grid_search(
                    input_dim,
                    output_dim,
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    param_grid_phase1,
                    look_back,
                    look_ahead
                )

                result_phase1 = {
                    "best_params": best_params_phase1,
                    "look_back": look_back,
                    "best_lag_VAR": int(best_lag),       # ép kiểu int để tránh lỗi JSON
                    "best_MSE": float(best_mse_phase1),        # lưu MSE
                    "search_time_seconds": float(search_time_phase1)  # lưu thời gian
                }
                with open(f"results/{model_type}/{file_name}/VAR_Lai_DeepVAR_grid_search_results_phase1.json", "w") as f:
                    json.dump(result_phase1, f)

                # Phase 2-Fine Tuning: Thu hẹp phạm vi tìm tham số tối ưu từ khoảng tốt nhất đã xác định
                param_grid_phase2 = {
                    "learning_rate": get_fine_values(best_params_phase1['learning_rate'], [3e-4, 5e-4, 7e-4, 1e-3, 3e-3]),
                    "batch_size": [best_params_phase1['batch_size']],  # giữ nguyên
                    "units_lstm": get_fine_values(best_params_phase1['units_lstm'], [32, 64, 96, 128, 256]),
                    "epoch": [300],
                    "dropout": [0.0, 0.05, 0.1],
                    "L2_reg": [0.0, 1e-6, 1e-5, 1e-4],
                    
                }

                best_params_phase2, best_mse_phase2, search_time_phase2 = grid_search(
                    input_dim,
                    output_dim,
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    param_grid_phase2,
                    look_back,
                    look_ahead
                )

                best_params = best_params_phase2
                best_mse = best_mse_phase2
                search_time = search_time_phase1 + search_time_phase2

                # Gom tất cả thông tin vào 1 dict
                results_to_save = {
                    "best_params": best_params,
                    "look_back": look_back,
                    "best_lag_VAR": int(best_lag),       # ép kiểu int để tránh lỗi JSON
                    "best_MSE": float(best_mse),        # lưu MSE
                    "search_time_seconds": float(search_time)  # lưu thời gian
                }
                # Lưu ra file JSON
                with open(f"results/{model_type}/{file_name}/VAR_Lai_DeepVAR_grid_search_results.json", "w") as f:
                    json.dump(results_to_save, f, indent=4)

                # Lưu VAR result vào results/{model_type}/{file_name}/var_result.pkl
                with open(var_result_path, "wb") as f:
                    pickle.dump(var_result, f)

                # Lưu danh sách tên cột để dùng khi predict
                trained_columns_path = os.path.join(vardeepvar_directory_results, "trained_columns.json")
                with open(trained_columns_path, "w", encoding="utf-8") as f:
                    json.dump(list(train_full.columns), f, ensure_ascii=False)

                # st.success("Parameter optimization completed.")
                st.write("**Best Parameters:**")
                st.json(best_params)
                st.write(f"**Best MSE:** {best_mse:.4f}")
                st.write(f"**Search Time:** {search_time:.2f} seconds")
                st.write(f"**Best Lags (VAR):** {best_lag}")
                st.write(f"**Look back:** {look_back}")

                # Cập nhật status: tối ưu tham số xong, chuẩn bị train VAR lai DeepVAR
                local_status.markdown(
                    """
                    <div style="
                        background-color:#E3F2FD;
                        border-left:5px solid #1E88E5;
                        padding:10px;
                        border-radius:6px;
                        margin-top:10px;
                    ">
                        <b style="color:#0D47A1;">'VAR lai DeepVAR' parameter optimization completed.</b><br>
                        <span style="color:#0D47A1;">
                            Best hyperparameters have been found. Proceeding to train the 'VAR lai DeepVAR' model...
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Toast nhỏ báo optimize xong
                st.toast("'VAR lai DeepVAR' parameter optimization finished. Starting training...")

                # Training
                final_model = build_lstm(
                                    input_dim,
                                    output_dim,
                                    look_back = look_back,
                                    look_ahead = look_ahead,
                                    lr = best_params["learning_rate"],
                                    units_lstm = best_params["units_lstm"],
                                    dropout = best_params["dropout"],
                                    L2_reg = best_params["L2_reg"]
                                )

                start_time = time.time()
                callbacks = make_callbacks_for_epochs(best_params["epoch"])
                history = final_model.fit(
                    X_train, y_train,
                    validation_data = (X_val, y_val),
                        epochs=best_params["epoch"],
                        batch_size=best_params["batch_size"],
                        verbose=1,
                        callbacks=callbacks
                        )
                end_time = time.time()

                train_time = end_time - start_time
                st.write(f"Training time: {train_time:.2f} seconds")
                total_train_time = train_time + search_time
                st.write(f"Total time to find optimal parameters and training time: {total_train_time:.2f} seconds")

                final_model.save(
                    f"models/{model_type}/{file_name}/VAR_Lai_DeepVAR_final_model.keras"
                )
                st.write ("Training completed and model saved.")
                with open(
                                        f"results/{model_type}/{file_name}/VAR_Lai_DeepVAR_training_history.pkl", "wb"
                                    ) as f:
                                        pickle.dump(history.history, f)

                # Lưu model & meta vào session để TEST dùng ngay
                st.session_state[f"{vardeepvar_session_key}_model"] = final_model
                st.session_state[f"{vardeepvar_session_key}_best_params"] = best_params
                st.session_state[f"{vardeepvar_session_key}_look_back"] = look_back
                st.session_state[f"{vardeepvar_session_key}_look_ahead"] = look_ahead

                print ("Training completed and model saved.")
                # Tạo DataFrame chứa loss theo từng epoch
                loss_df = pd.DataFrame(
                    {
                        "Epoch": range(
                            1, len(history.history["loss"]) + 1
                            ),  # noqa: E501
                            "Training Loss": history.history["loss"],
                            "Validation Loss": history.history["val_loss"],
                            }
                            )
                plot_loss_curve(loss_df)

                # Thông báo tổng kết training 'VAR lai DeepVAR'
                st.success(
                    f"'VAR lai DeepVAR' training completed successfully!"
                )

                # Cập nhật status lớn ở trên
                local_status.markdown(
                    f"""
                    <div style="
                        background-color:#E8F5E9;
                        border-left:5px solid #43A047;
                        padding:10px;
                        border-radius:6px;
                        margin-top:10px;
                    ">
                        <b style="color:#1B5E20;">'VAR lai DeepVAR' pipeline completed for dataset: <code>{file_name}</code></b><br>
                        <span style="color:#1B5E20;">
                            Model has been trained, saved to disk, and stored in memory for immediate testing.
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.session_state["manual_stop_requested"] = False

                # Toast nhỏ báo train xong
                st.toast("'VAR lai DeepVAR' model trained and saved successfully!")

            if stop_training.is_set():

                st.subheader(f"Last saved {file_name} model training results")

                st.session_state["manual_stop_requested"] = True  # NEW

                grid_search_path = (
                    f"results/{model_type}/{file_name}/VAR_Lai_DeepVAR_grid_search_results.json"
                )
                training_history_path = (
                    f"results/{model_type}/{file_name}/VAR_Lai_DeepVAR_training_history.pkl"
                )
                var_result_path = os.path.join(vardeepvar_directory_results, "var_result.pkl")

                # === KIỂM TRA FILE ===
                if not os.path.exists(grid_search_path) or not os.path.exists(training_history_path):
                    st.error("The complete result files could not be found. Please train the model first.")

                else:
                    # ===== LOAD GRID SEARCH RESULT =====
                    with open(grid_search_path, "r") as f:
                        grid_search_data = json.load(f)

                    manual_stop = st.session_state.get("manual_stop_requested", False)

                    if manual_stop:
                        show_last_saved_training_results_box(
                            model_type=model_type,
                            file_name=file_name,
                            note_if_missing_cols=(
                                f"Apply for: {file_name} weather dataset with all columns "
                                "after first-order differencing (non-stationary) and min-max normalization. "
                                "No data augmentation techniques are performed."
                            ),
                        )

                    st.write("#### Hyperparameters")
                    # st.json(grid_search_data)


                    # 1) Best params table
                    best_params = grid_search_data.get("best_params", {})
                    if isinstance(best_params, dict) and best_params:
                        st.markdown("**Best params**")
                        df_params = pd.DataFrame(best_params.items(), columns=["Hyperparameter", "Value"])

                        # ÉP Value thành string
                        df_params["Value"] = df_params["Value"].astype(str)

                        st.dataframe(df_params, use_container_width=True)


                    # 2) Summary table
                    summary = {k: v for k, v in grid_search_data.items() if k != "best_params"}
                    if summary:
                        st.markdown("**Search summary**")
                        df_summary = pd.DataFrame(summary.items(), columns=["Metric", "Value"])

                        # ÉP Value thành string
                        df_summary["Value"] = df_summary["Value"].astype(str)

                        st.dataframe(df_summary, use_container_width=True)

                    # --- 1. Nếu đã có VAR train từ trước → load lên dùng ngay ---
                    if os.path.exists(var_result_path) and os.path.getsize(var_result_path) > 0:
                        try:
                            with open(var_result_path, "rb") as f:
                                var_result = pickle.load(f)
                            best_lag = var_result.k_ar

                            # st.info(f"Best_lag = {best_lag}.")
                        except Exception as e:
                            st.warning("Found VAR model but failed to load it. Recomputing VAR...")
                            var_result = None
                            best_lag = None

                    # ===== LOAD TRAINING HISTORY =====
                    with open(training_history_path, "rb") as f:
                        loaded_history = pickle.load(f)

                    # ===== FORMAT LOSS DATAFRAME =====
                    if "val_loss" in loaded_history:
                        loss_df = pd.DataFrame(
                            {
                                "Epoch": range(1, len(loaded_history["loss"]) + 1),
                                "Training Loss": loaded_history["loss"],
                                "Validation Loss": loaded_history["val_loss"],
                            }
                        )
                    else:
                        loss_df = pd.DataFrame(
                            {
                                "Epoch": range(1, len(loaded_history["loss"]) + 1),
                                "Training Loss": loaded_history["loss"],
                            }
                        )

                    # ===== VẼ BIỂU ĐỒ LOSS =====
                    st.write("#### Training & Validation Loss")
                    st.line_chart(loss_df.set_index("Epoch"))

                    # ===== HIỂN THỊ LOSS =====
                    st.write("#### Final Losses")
                    st.write(f"**Training Loss (final)**: {loaded_history['loss'][-1]:.4f}")

                    if "val_loss" in loaded_history:
                        st.write(f"**Validation Loss (final)**: {loaded_history['val_loss'][-1]:.4f}")

            if test_button:

                st.write("### 'VAR lai DeepVAR' Model Testing Results")

                # Paths to saved artifacts from TRAIN
                grid_search_path = (
                    f"results/{model_type}/{file_name}/VAR_Lai_DeepVAR_grid_search_results.json"
                )
                model_path = (
                    f"models/{model_type}/{file_name}/VAR_Lai_DeepVAR_final_model.keras"
                )
                var_result_path = (
                    f"results/{model_type}/{file_name}/var_result.pkl"
                )
                training_history_path = (
                    f"results/{model_type}/{file_name}/VAR_Lai_DeepVAR_training_history.pkl"
                )

                # Ensure evaluation directory exists
                os.makedirs(vardeepvar_directory_evaluation, exist_ok=True)

                # Check all required files
                if (
                    not os.path.exists(grid_search_path)
                    or not os.path.exists(model_path)
                    or not os.path.exists(var_result_path)
                    or not os.path.exists(training_history_path)
                ):
                    st.error(
                        "The complete 'VAR lai DeepVAR' result files could not be found. "
                        "Please train the 'VAR lai DeepVAR' model first."
                    )
                else:
                    # 1. Load grid search info (for display only)
                    with open(grid_search_path, "r") as f:
                        grid_search_data = json.load(f)

                    # 2. Lấy trained model: ưu tiên trong session, nếu không thì load từ file
                    var_lai_deepvar_model = st.session_state.get(f"{vardeepvar_session_key}_model", None)
                    model_source = "memory"

                    if var_lai_deepvar_model is None:
                        model_source = "disk"

                        if (not os.path.exists(model_path)) or os.path.getsize(model_path) == 0:
                            st.error(
                                "'VAR lai DeepVAR' model file not found or is empty. "
                                "Please train the 'VAR lai DeepVAR' model first."
                            )
                            st.stop()

                        var_lai_deepvar_model = load_model(model_path)
                        # lưu ngược lại vào session cho những lần test sau
                        st.session_state[f"{vardeepvar_session_key}_model"] = var_lai_deepvar_model

                    # Load hyperparameters
                    st.subheader("Loaded best hyperparameters (training phase)")
                    # st.json(grid_search_data)


                    # 1) Best params table
                    best_params = grid_search_data.get("best_params", {})
                    if isinstance(best_params, dict) and best_params:
                        st.markdown("**Best params**")
                        df_params = pd.DataFrame(best_params.items(), columns=["Hyperparameter", "Value"])

                        # ÉP Value thành string
                        df_params["Value"] = df_params["Value"].astype(str)

                        st.dataframe(df_params, use_container_width=True)


                    # 2) Summary table
                    summary = {k: v for k, v in grid_search_data.items() if k != "best_params"}
                    if summary:
                        st.markdown("**Search summary**")
                        df_summary = pd.DataFrame(summary.items(), columns=["Metric", "Value"])

                        # ÉP Value thành string
                        df_summary["Value"] = df_summary["Value"].astype(str)

                        st.dataframe(df_summary, use_container_width=True)

                    # 3. Load VAR result to recover best_lag = look_back
                    with open(var_result_path, "rb") as f:
                        var_result = pickle.load(f)

                    best_lag = var_result.k_ar      
                    look_back = 1
                    look_ahead = 1                  
                    k_model = var_result.neqs       # number of equations = number of variables

                    # Basic shape check
                    if test.shape[1] != k_model:
                        st.error(
                            f"Incompatible test data: 'VAR lai DeepVAR' model was trained with {k_model} variables, "
                            f"but test data has {test.shape[1]} columns.\n"
                            "Please ensure you use the same features as during training or retrain the 'VAR lai DeepVAR' model."
                        )
                    else: 
                        start_time = time.time()

                        # 1) Dự đoán VAR trên test
                        test_var_pred = create_var_predictions(test, var_result, best_lag, test.columns)

                        # 2) Lấy giá trị thực tế tương ứng khoảng dự báo
                        #    notebook: test_start_idx = train.shape[0] + var_result.k_ar
                        test_start_idx = train_full.shape[0] + var_result.k_ar
                        test_end_idx   = dataset.shape[0] - 1
                        test_actual    = dataset.iloc[test_start_idx:test_end_idx + 1]

                        # 3) Sai số: thực tế - dự đoán VAR
                        test_var_error = test_actual - test_var_pred

                        # 4) Tạo windows cho mô hình học sai số
                        X_test = create_windows(
                            test_var_error,
                            window_shape=look_back,
                            end_id=-look_ahead
                        )
                        y_test = create_windows(
                            test_var_error,
                            window_shape=look_ahead,
                            start_id=look_back
                        )

                        # 5) Dự đoán sai số bằng model
                        prediction_error = var_lai_deepvar_model.predict(X_test)
                        execution_time = time.time() - start_time

                        # 6) Tạo dự đoán cuối cùng: VAR + error_model
                        y_test_var_pred = create_windows(
                            test_var_pred,
                            window_shape=look_ahead,
                            start_id=look_back
                        )
                        predictions = y_test_var_pred + prediction_error
                        actual      = y_test_var_pred + y_test   

                        st.success(f"Inference completed in {execution_time:.2f} seconds.")

                        # ===== LOAD TRAINING HISTORY =====
                        with open(training_history_path, "rb") as f:
                            loaded_history = pickle.load(f)

                        # ===== FORMAT LOSS DATAFRAME =====
                        if "val_loss" in loaded_history:
                            loss_df = pd.DataFrame(
                                {
                                    "Epoch": range(1, len(loaded_history["loss"]) + 1),
                                    "Training Loss": loaded_history["loss"],
                                    "Validation Loss": loaded_history["val_loss"],
                                }
                            )
                        else:
                            loss_df = pd.DataFrame(
                                {
                                    "Epoch": range(1, len(loaded_history["loss"]) + 1),
                                    "Training Loss": loaded_history["loss"],
                                }
                            )

                        # ===== VẼ BIỂU ĐỒ LOSS =====
                        st.subheader("Training & Validation Loss (from training phase)")
                        plot_loss_curve(loss_df)

                        # ===== ĐÁNH GIÁ (SCALED) =====
                        evaluation_df = evaluate_multivariate_forecast(
                            actual,          # 3D: (N, look_ahead, n_vars')
                            predictions,     # 3D
                            test.columns
                        )
                        evaluation_df_overall = evaluate_overall_forecast(
                            actual,
                            predictions,
                            execution_time
                        )

                        # Chuẩn bị bảng (scaled) để xem
                        actual_flat = actual.reshape(-1, actual.shape[-1])
                        preds_flat  = predictions.reshape(-1, predictions.shape[-1])
                        actual_scaled_df = pd.DataFrame(actual_flat, columns=test.columns)
                        preds_scaled_df  = pd.DataFrame(preds_flat,  columns=test.columns)

                        # Lưu CSV (scaled)
                        evaluation_df.to_csv(
                            os.path.join(vardeepvar_directory_evaluation, "VAR_Lai_DeepVAR_evaluation_scaled.csv"),
                            index=False
                        )
                        evaluation_df_overall.to_csv(
                            os.path.join(vardeepvar_directory_evaluation, "VAR_Lai_DeepVAR_evaluation_overall_scaled.csv"),
                            index=False
                        )

                        # ===== RESTORE VỀ ORIGINAL =====
                        scaler    = st.session_state.get("normalization_scaler", None)
                        flag_diff = st.session_state.get("flag_diff", False)
                        lag       = st.session_state.get("lag", 0)
                        target_columns = st.session_state.get("target_columns", list(test.columns))

                        pred_restore = None
                        actual_restore = None
                        evaluation_df_org = None
                        evaluation_df_overall_org = None

                        has_original_view = False  # cờ để quyết định hiển thị 1 hay 2 phần

                        if scaler is not None:
                            # Chuẩn bị dữ liệu gốc
                            dataset_for_restore = preprocess_data(dataset_original)

                            if flag_diff:
                                id_restore = train_full.shape[0] + best_lag + look_back
                            else:
                                id_restore = train_full.shape[0] + best_lag + look_back - 1

                            original_segment = dataset_for_restore.iloc[(id_restore - 1):(id_restore + 1)]
                            original_segment = original_segment[target_columns].copy()

                            exclude_cols = None

                            # Khôi phục dự đoán & thực tế 
                            pred_restore = inverse_transformation(
                                predictions,
                                scaler,
                                original_segment.columns,
                                original_segment,
                                lag,
                                exclude_cols,
                                flag_diff
                            )
                            actual_restore = inverse_transformation(
                                actual,
                                scaler,
                                original_segment.columns,
                                original_segment,
                                lag,
                                exclude_cols,
                                flag_diff
                            )

                            # DF để hiển thị/plot
                            pred_restore_df = (
                                pred_restore
                                if isinstance(pred_restore, pd.DataFrame)
                                else pd.DataFrame(pred_restore, columns=original_segment.columns)
                            )
                            actual_restore_df = (
                                actual_restore
                                if isinstance(actual_restore, pd.DataFrame)
                                else pd.DataFrame(actual_restore, columns=original_segment.columns)
                            )

                            # Đánh giá (original)
                            evaluation_df_org = evaluate_multivariate_forecast(
                                actual_restore, pred_restore, actual_restore.columns
                            )
                            evaluation_df_overall_org = evaluate_overall_forecast_restore(
                                actual_restore, pred_restore, execution_time
                            )

                            # Lưu CSV (original)
                            evaluation_df_org.to_csv(
                                os.path.join(vardeepvar_directory_evaluation, "VAR_Lai_DeepVAR_evaluation_org.csv"),
                                index=False
                            )
                            evaluation_df_overall_org.to_csv(
                                os.path.join(vardeepvar_directory_evaluation, "VAR_Lai_DeepVAR_evaluation_overall_org.csv"),
                                index=False
                            )

                            has_original_view = True
                        else:
                            # Không có scaler: không restore
                            has_original_view = False

                        # ================= HIỂN THỊ BẢNG / PLOT / EVALUATION =================

                        if has_original_view:
                            # CÓ SCALER
                            # st.subheader("Data View (scaled vs original)")
                            with st.expander("View testing results (transformed data vs inverse transformed data)", expanded=True):
                                tab_scaled, tab_original = st.tabs(
                                    ["Transformed data", "Inverse transformed data"]
                                )

                                # ----- TAB 1: SCALED -----
                                with tab_scaled:
                                    st.markdown("#### Tables")
                                    st.markdown("**Actual**")
                                    st.dataframe(actual_scaled_df)
                                    st.markdown("**Predictions**")
                                    st.dataframe(preds_scaled_df)

                                    st.markdown("#### Plots")
                                    variable_names = list(test.columns)
                                    for idx, name in enumerate(variable_names):
                                        st.subheader(f"{name}")
                                        plot_actual_vs_predicted_new(
                                            actual,
                                            predictions,
                                            variable_index=idx,
                                            step_index=0,
                                            variable_name=name
                                        )

                                    st.markdown("#### Evaluation")
                                    st.dataframe(evaluation_df)
                                    st.dataframe(evaluation_df_overall)

                                # ----- TAB 2: ORIGINAL -----
                                with tab_original:
                                    if (scaler is None) or (pred_restore is None) or (actual_restore is None):
                                        st.warning(
                                            "Original scale data is not available because restoration was not performed."
                                        )
                                    else:
                                        st.markdown("#### Tables")
                                        st.markdown("**Actual restore**")
                                        st.dataframe(actual_restore_df)
                                        st.markdown("**Predictions restore**")
                                        st.dataframe(pred_restore_df)

                                        st.markdown("#### Plots")
                                        variable_names_org = list(actual_restore_df.columns)
                                        for idx, name in enumerate(variable_names_org):
                                            st.subheader(f"{name}")
                                            plot_actual_vs_predicted_new(
                                                actual_restore,
                                                pred_restore,
                                                variable_index=idx,
                                                step_index=0,
                                                variable_name=name
                                            )

                                        st.markdown("#### Evaluation")
                                        st.dataframe(evaluation_df_org)
                                        st.dataframe(evaluation_df_overall_org)

                        else:
                            # KHÔNG CÓ SCALER
                            # st.subheader("Data View")
                            with st.expander("View", expanded=True):
                                st.markdown("#### Tables")
                                st.markdown("**Actual**")
                                st.dataframe(actual_scaled_df)
                                st.markdown("**Predictions**")
                                st.dataframe(preds_scaled_df)

                                st.markdown("#### Plots")
                                variable_names = list(test.columns)
                                for idx, name in enumerate(variable_names):
                                    st.subheader(f"{name}")
                                    plot_actual_vs_predicted_new(
                                        actual,
                                        predictions,
                                        variable_index=idx,
                                        step_index=0,
                                        variable_name=name
                                    )

                                st.markdown("#### Evaluation")
                                st.dataframe(evaluation_df)
                                st.dataframe(evaluation_df_overall)

        # 10. Predict 
        st.markdown("### Weather dataset prediction (.csv)")
        uploaded_file = st.file_uploader("Select CSV file", type=["csv"])

        if uploaded_file:
            st.write("Dataset has been uploaded:")
            weather_data = pd.read_csv(uploaded_file)
            
            weather_recent = preprocess_data_predict(weather_data)
            st.dataframe(weather_recent)
            st.write(weather_recent.shape)

            # ======= Auto-pick columns for prediction (no user selection) =======
            all_columns = weather_recent.columns.tolist()

            trained_cols = None
            result_dir = f"results/{model_type}/{file_name}"
            var_dir = f"results/{model_type}/{file_name}"
            var_result_path_predict = os.path.join(var_dir, "var_result.pkl")
            trained_cols_json = os.path.join(var_dir, "trained_columns.json")  # optional companion file

            # 1) Prefer JSON list saved during training
            if os.path.exists(trained_cols_json):
                try:
                    with open(trained_cols_json, "r", encoding="utf-8") as f:
                        candidate = json.load(f)
                    if isinstance(candidate, (list, tuple)):
                        trained_cols = [c for c in candidate if c in all_columns]
                except Exception:
                    trained_cols = None

            # 2) Else, try extracting from var_result.pkl
            if trained_cols is None and os.path.exists(var_result_path_predict) and os.path.getsize(var_result_path_predict) > 0:
                try:
                    with open(var_result_path_predict, "rb") as f:
                        _vr = pickle.load(f)

                    candidate = None
                    if hasattr(_vr, "names") and isinstance(_vr.names, (list, tuple)):
                        candidate = list(_vr.names)
                    elif hasattr(_vr, "model") and hasattr(_vr.model, "endog_names"):
                        en = _vr.model.endog_names
                        if isinstance(en, str):
                            candidate = [en]
                        elif isinstance(en, (list, tuple)):
                            candidate = list(en)

                    if candidate:
                        trained_cols = [c for c in candidate if c in all_columns]
                except Exception:
                    trained_cols = None

            # 3) Decide final columns to use (no multiselect)
            selected_columns = trained_cols if trained_cols else all_columns

            if trained_cols:
                st.markdown(
                    f"""
                    <div style="
                        border-left: 6px solid #4CAF50;
                        background-color: #F0FFF0;
                        padding: 10px;
                        margin: 10px 0;
                        border-radius: 8px;
                    ">
                        <b style="color:#2E7D32;">Using columns from previous training:</b><br>
                        <span style="font-size:16px; color:#333;">{', '.join(selected_columns)}</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div style="
                        border-left: 6px solid #FF9800;
                        background-color: #FFF3E0;
                        padding: 10px;
                        margin: 10px 0;
                        border-radius: 8px;
                    ">
                        <b style="color:#E65100;">Could not detect trained columns</b><br>
                        <span style="font-size:16px; color:#333;">Using all columns from the uploaded file.</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # Subset data to used columns
            weather_selected = weather_recent[selected_columns]

            st.write("### Columns used for prediction")
            st.dataframe(weather_selected)
            st.write("Shape (selected):", weather_selected.shape)

            # ======= Normalize (unchanged) =======
            if "norm_min_val" in st.session_state and "norm_max_val" in st.session_state:
                min_val = st.session_state.norm_min_val
                max_val = st.session_state.norm_max_val
            else:
                min_val = None
                max_val = None

            if min_val is not None and max_val is not None:
                weather_recent_scaled, scaler_pred = min_max_normalize(min_val, max_val, weather_selected)
            else:
                weather_recent_scaled, scaler_pred = min_max_normalize(0, 1, weather_selected)

            weather_selected = weather_recent_scaled

            # Predict button
            predict_button = st.button("### Prediction")
            
            if predict_button:               

                # Đường dẫn chung                     
                result_dir = f"results/{model_type}/{file_name}"
                var_dir = f"results/{model_type}/{file_name}"
                var_result_path = os.path.join(var_dir, "var_result.pkl")

                # VAR result để lấy look_back
                if not os.path.exists(var_result_path) or os.path.getsize(var_result_path) == 0:
                    st.error("VAR result file not found or is empty. Please train the VAR model first.")
                    st.stop()

                # Load VAR result (để lấy best_lag = look_back)
                with open(var_result_path, "rb") as f:
                    var_result = pickle.load(f)
                best_lag = var_result.k_ar

                predictions = None

                if model_type == "VAR":
                    # Bước 4: Đưa dữ liệu vào mô hình VAR để dự đoán
                    predictions = create_var_predictions(
                        weather_recent_scaled,
                        var_result,
                        best_lag,
                        weather_recent_scaled.columns
                    )

                elif model_type == "VAR-LSTM":
                    # * VAR-LSTM:

                    model_path = f"models/{model_type}/{file_name}/VAR_LSTM_final_model.keras"
                    lag_file_path = f"best_lag/{model_type}/{file_name}/best_lag_var_pred.csv"

                    # Lấy look_back từ file best_lag_var_pred.csv
                    if os.path.exists(lag_file_path):
                        df_lag = pd.read_csv(lag_file_path)
                        if "AIC" in df_lag.columns and "p" in df_lag.columns:
                            idx_min = df_lag["AIC"].astype(float).idxmin()
                            best_lag_var_pred = int(df_lag.loc[idx_min, "p"])
                            # st.info(f"Loaded existing AIC table. best_lag_var_pred = {best_lag_var_pred}")
                        else:
                            st.warning("Existing CSV missing columns 'p' or 'AIC'.")
                            best_lag_var_pred = None
                    else:
                        raise FileNotFoundError(f"File not found: {lag_file_path}")
                    
                    if not os.path.exists(model_path) or os.path.getsize(model_path) == 0:
                        st.error("VAR-LSTM model file not found or is empty. Please train the VAR-LSTM model first.")
                        st.stop()
                        
                    look_back = best_lag_var_pred  # tương đương biến look_back trong notebook

                    # Bước 4: Đưa dữ liệu vào mô hình
                    model = load_model(model_path)
                    var_pred = create_var_predictions(
                        weather_recent_scaled,
                        var_result,
                        best_lag,
                        weather_recent_scaled.columns
                    )
                    var_pred_input = create_windows(var_pred, window_shape=look_back)
                    predictions = model.predict(var_pred_input)

                elif model_type == "DEEPVAR":
                    # * DEEPVAR:

                    look_back = best_lag
                    model_path = f"models/{model_type}/{file_name}/DeepVAR_final_model.keras"
                    if not os.path.exists(model_path) or os.path.getsize(model_path) == 0:
                        st.error("DeepVAR model file not found or is empty. Please train the DEEPVAR model first.")
                        st.stop()

                    # Bước 4: Đưa dữ liệu vào mô hình
                    model = load_model(model_path)
                    weather_input = create_windows(weather_recent_scaled, window_shape=look_back)
                    predictions = model.predict(weather_input)

                elif model_type == "VAR_LAI_DEEPVAR":
                    # * VAR lai DeepVAR:

                    look_back = 1
                    model_path = f"models/{model_type}/{file_name}/VAR_Lai_DeepVAR_final_model.keras"
                    if not os.path.exists(model_path) or os.path.getsize(model_path) == 0:
                        st.error("'VAR lai DeepVAR' model file not found or is empty. Please train the model first.")
                        st.stop()

                    # Bước 4: Đưa dữ liệu vào mô hình
                    model = load_model(model_path)
                    var_pred = create_var_predictions(
                        weather_recent_scaled,
                        var_result,
                        best_lag,
                        weather_recent_scaled.columns
                    )
                    var_pred_input = create_windows(var_pred, window_shape=look_back)
                    predictions = model.predict(var_pred_input)

                else:
                    st.error("Unsupported model type for prediction.")
                    st.stop()

                # ===== Bước 5: Hoàn nguyên dự đoán (đúng notebook) =====

                # Chuẩn bị dữ liệu gốc
                dataset_for_restore_predict = preprocess_data_predict(weather_data)

                # target_columns: các cột đã dùng để train VAR
                target_columns = st.session_state.get(
                    "target_columns",
                    list(weather_recent.columns)  # fallback
                )

                # Xác định vị trí cuối cùng dùng để khôi phục
                id_pre = len(dataset_for_restore_predict) - 1
                original_segment = dataset_for_restore_predict.iloc[(id_pre - 1):(id_pre + 1)]
                original_segment = original_segment[target_columns].copy()

                # Ở bước dự báo này không sai phân, lag_pre = 0, flag_diff_pre = False
                exclude_cols = None
                lag_pre = 0
                flag_diff_pre = False

                # Dùng scaler_pred (từ bước min_max_normalize phía trên)
                predictions_restore = inverse_transformation(
                    predictions,
                    scaler_pred,
                    original_segment.columns,
                    original_segment,
                    lag_pre,
                    exclude_cols,
                    flag_diff_pre
                )

                # Lấy hàng cuối cùng – giá trị dự báo cho ngày tiếp theo
                if isinstance(dataset_for_restore_predict.index, pd.DatetimeIndex):
                    future_date = dataset_for_restore_predict.index[-1] + pd.Timedelta(days=1)
                else:
                    future_date = dataset_for_restore_predict.index[-1] + 1

                predicted_df = pd.DataFrame(
                    [predictions_restore.iloc[-1].values],
                    columns=predictions_restore.columns,
                    index=[future_date]
                )
                predicted_df.index.name = "Date"

                st.subheader("Weather forecast for the next day.")
                st.dataframe(predicted_df)

                # ===== Bước 6: Vẽ biểu đồ đường dự báo cho từng biến =====

                with st.expander("Forecast plots", expanded=False):

                    # 1) Lấy dữ liệu lịch sử
                    history_df = preprocess_data_restore(weather_data)

                    # 2) Xác định các cột có thể plot (phải tồn tại ở cả history & forecast)
                    columns_for_plot = [
                        c for c in history_df.columns
                        if c in predicted_df.columns
                    ]

                    if not columns_for_plot:
                        st.warning("No common columns available for plotting.")
                    else:
                        st.caption(
                            "Forecast plots are shown for all available variables "
                            "that exist in both historical data and prediction results."
                        )
                        # 3) Plot từng biến
                        title_prefix = f"{model_type} Forecast – {file_name}"

                        for col in columns_for_plot:
                            plot_weather_forecast_single_variable(
                                variable_name=col,
                                history_df=history_df[[col]],
                                forecast_df=predicted_df[[col]],
                                title_prefix=title_prefix
                            )

if __name__ == "__main__":
    main()
