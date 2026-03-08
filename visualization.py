import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd
import numpy as np
import os
import json

def plot_normalized_data(data, normalization_method):
    """Vẽ biểu đồ dữ liệu sau khi chuẩn hóa với tùy chọn cột."""
    
    # Cho người dùng chọn cột cần hiển thị
    column_to_plot = st.selectbox("Select column to display chart:", data.columns, key="norm_column_select")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data.index, data[column_to_plot], label=normalization_method, color='blue')
    ax.set_title(f"{normalization_method} Normalized Data - {column_to_plot}")
    ax.set_xlabel("Time")
    ax.set_ylabel(column_to_plot)
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)

# Vẽ biểu đồ trực quan hóa dữ liệu
def visualize_data(data):
    """Visualize data with histograms, boxplots, or line plots."""

    column = st.selectbox("Select column to visualize", data.columns)
    chart_type = st.radio("Select chart type", ["Histogram", "Boxplot", "Line Plot"])

    if chart_type == "Histogram":
        fig, ax = plt.subplots()
        ax.hist(data[column].dropna(), bins=10, color='skyblue', edgecolor='black')
        st.pyplot(fig)

    elif chart_type == "Boxplot":
        fig, ax = plt.subplots()
        sns.boxplot(x=data[column].dropna(), ax=ax, color='lightgreen')
        st.pyplot(fig)

    elif chart_type == "Line Plot":
        # Nhóm theo tháng
        grouped_data = data[column].resample('M').mean()
        date_fmt = "%m-%Y"  # Hiển thị: 01-2025, 02-2025, ...

        # Tăng kích thước biểu đồ (chiều ngang)
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(grouped_data.index, grouped_data.values, color='purple')
        ax.set_title(f"{column}")
        ax.set_xlabel("Month")
        ax.set_ylabel(column)
        ax.set_xticks(grouped_data.index)
        ax.set_xticklabels([d.strftime(date_fmt) for d in grouped_data.index], rotation=45)
        st.pyplot(fig)

# Vẽ ma trận tương quan.
def plot_correlation_matrix(dataset, figsize=(12, 8)):
    # Chỉ lấy dữ liệu số
    numeric_data = dataset.select_dtypes(include=[np.number])

    if numeric_data.empty:
        st.error("Dataset không có cột dạng số để tính tương quan!")
        return

    corr_matrix = numeric_data.corr()

    # Tạo figure
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap='coolwarm',
        linewidths=0.5,
        ax=ax
    )

    ax.set_title("Correlation Matrix")

    # Hiển thị trên Streamlit
    st.pyplot(fig)

    # Giải phóng bộ nhớ
    plt.close(fig)

# So sánh dữ liệu tăng cường và dữ liệu gốc
def compare_original_augmented(dataset, dataset_aug, column, figsize=(10, 8)):
    if column not in dataset.columns or column not in dataset_aug.columns:
        st.error(f"Cột '{column}' không tồn tại trong một trong hai DataFrame.")
        return
    
    # Tạo figure và ax cho 2 subplot
    fig, axes = plt.subplots(2, 1, figsize=figsize)

    # === Biểu đồ 1: So sánh toàn bộ dữ liệu ===
    axes[0].plot(dataset[column], label="Original", alpha=0.7)
    axes[0].plot(dataset_aug[column], label="Augmented", alpha=0.5, linestyle='--')
    axes[0].legend()
    axes[0].set_title(f"Comparison of training data and augmented data - {column}")

    # === Biểu đồ 2: Chỉ phần dữ liệu mới ===
    augmented_part = dataset_aug.loc[dataset_aug.index < dataset.index.min()]
    axes[1].plot(
        augmented_part.index,
        augmented_part[column],
        label="Generated Data",
        linestyle='--'
    )
    axes[1].legend()
    axes[1].set_title(f"Only augmented data - {column}")

    plt.tight_layout()

    # Hiển thị trong Streamlit
    st.pyplot(fig)

    # Giải phóng bộ nhớ
    plt.close(fig)

# Vẽ biểu đồ phân phối giá trị các trường
def plot_distribution(data, column, bins=30, figsize=(10, 6)):
    if column not in data.columns:
        st.error(f"Column '{column}' does not exist in the DataFrame.")
        return

    # Lọc bỏ NaN cho an toàn
    col_data = data[column].dropna()

    fig, ax = plt.subplots(figsize=figsize)
    sns.histplot(col_data, bins=bins, kde=True, ax=ax)
    ax.set_title(f'Distribution of {column}')
    ax.set_xlabel(column)
    ax.set_ylabel('Frequency')

    st.pyplot(fig)      # <── Đưa figure lên Streamlit
    plt.close(fig)      # Giải phóng bộ nhớ (tốt khi vẽ nhiều)


# Vẽ biểu đồ đường quan sát sự thay đổi theo thời gian
def plot_smoothed_time_series(dataset, column, window=30, 
                              title=None, xlabel="Date", ylabel=None, 
                              figsize=(10, 5)):
    if column not in dataset.columns:
        st.error(f"Column '{column}' does not exist in the DataFrame.")
        return
    
    smoothed = dataset[column].rolling(window=window, center=True).mean()

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(dataset.index, smoothed, label=f'{window}-day Rolling Mean', alpha=0.9)
    ax.plot(dataset.index, dataset[column], label='Original', alpha=0.3)
    
    ax.set_title(title if title else f"Smoothed Change in {column} through date")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel if ylabel else column)
    ax.legend()

    st.pyplot(fig)
    plt.close(fig)

def plot_dataset_split(column, train_df, val_df, test_df):
    """
    Vẽ biểu đồ Train - Validation - Test cho 1 cột dữ liệu.
    Phù hợp để dùng trong Streamlit.
    """

    # Kiểm tra cột có tồn tại không
    for df_name, df in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
        if column not in df.columns:
            st.warning(f"Column '{column}' does not exist in {df_name} dataset.")
            return

    fig, ax = plt.subplots(figsize=(16, 4))

    # Train
    ax.plot(train_df.index, train_df[column], label="Train", color="blue")

    # Validation (có thể rỗng nếu user chọn val_ratio = 0)
    if len(val_df) > 0:
        ax.plot(val_df.index, val_df[column], label="Validation", color="orange")

    # Test
    ax.plot(test_df.index, test_df[column], label="Test", color="green")

    ax.set_title(f"Train / Validation / Test Split - {column}")
    ax.set_xlabel("Time")
    ax.set_ylabel(column)
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)

# Vẽ biểu đồ loss
def plot_loss_curve(loss_df):
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(loss_df["Epoch"], loss_df["Training Loss"], label="Training Loss")
    ax.plot(loss_df["Epoch"], loss_df["Validation Loss"], label="Validation Loss")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title("Training vs Validation Loss")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)   
    plt.close(fig)   

def plot_actual_vs_predicted(y_test, predictions, variable_index=0, step_index=0, variable_name=None):
    """
    Vẽ biểu đồ so sánh giữa giá trị thực tế và dự đoán cho từng biến.
    Tự động nhận dạng dữ liệu 2D hoặc 3D.
    
    Parameters
    ----------
    y_test : np.ndarray hoặc pd.DataFrame
        Dữ liệu thực tế (2D: samples x features, hoặc 3D: samples x timesteps x features)
    predictions : np.ndarray hoặc pd.DataFrame
        Dữ liệu dự đoán, cùng shape với y_test
    variable_index : int
        Vị trí biến (cột) cần vẽ
    step_index : int
        Nếu dữ liệu là 3D, chọn bước thời gian (ví dụ step 0 = giá trị đầu tiên)
        Nếu dữ liệu 2D, sẽ bị bỏ qua
    variable_name : str
        Tên biến hiển thị trên biểu đồ
    """
    # Nếu là DataFrame → chuyển sang numpy
    if hasattr(y_test, "values"):
        y_test = y_test.values
    if hasattr(predictions, "values"):
        predictions = predictions.values

    # Nếu dữ liệu là 3D → chọn step_index
    if y_test.ndim == 3 and predictions.ndim == 3:
        actual = y_test[:, step_index, variable_index]
        predicted = predictions[:, step_index, variable_index]
    else:
        # Dữ liệu 2D
        actual = y_test[:, variable_index]
        predicted = predictions[:, variable_index]

    # Tạo biểu đồ
    plt.figure(figsize=(12, 5))
    plt.plot(actual, label='Actual', color='tab:blue', linewidth=2)
    plt.plot(predicted, label='Predicted', color='tab:orange', linestyle='--', linewidth=2)

    var_label = variable_name if variable_name else f"Variable {variable_index}"
    plt.title(f"Actual vs Predicted - {var_label}", fontsize=14)
    plt.xlabel("Time step")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def plot_actual_vs_predicted_new (y_test, predictions, variable_index=0, step_index=0, variable_name=None):
    """
    Vẽ biểu đồ so sánh giữa giá trị thực tế và dự đoán cho từng biến.
    Tự động nhận dạng dữ liệu 2D hoặc 3D, hiển thị rõ ràng và dễ đọc.
    """
    # Nếu là DataFrame → chuyển sang numpy
    if hasattr(y_test, "values"):
        y_test = y_test.values
    if hasattr(predictions, "values"):
        predictions = predictions.values

    # Nếu dữ liệu là 3D → chọn step_index
    if y_test.ndim == 3 and predictions.ndim == 3:
        actual = y_test[:, step_index, variable_index]
        predicted = predictions[:, step_index, variable_index]
    else:
        # Dữ liệu 2D
        actual = y_test[:, variable_index]
        predicted = predictions[:, variable_index]

    # --- Biểu đồ ---
    plt.figure(figsize=(12, 5))
    plt.plot(actual, label='Actual', color='tab:blue', linewidth=2)
    plt.plot(predicted, label='Predicted', color='tab:orange', linewidth=2, alpha=0.8)

    var_label = variable_name if variable_name else f"Variable {variable_index}"
    plt.title(f"Actual vs Predicted - {var_label}", fontsize=14, fontweight='bold')
    plt.xlabel("Time step", fontsize=12)
    plt.ylabel("Value", fontsize=12)
    plt.legend(frameon=True, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()

    # 🔁 Hiển thị trên Streamlit (thay cho plt.show())
    st.pyplot(plt.gcf())

def plot_deepvar_forecast(y_true, y_pred, time_index, column_names, col_to_plot):

    if col_to_plot not in column_names:
        print(f"[!] Column '{col_to_plot}' does not exist in the DataFrame.")
        return
    
    col_idx = column_names.index(col_to_plot)

    # Flatten thành (samples * steps,)
    y_true_flat = y_true[:, :, col_idx].reshape(-1)
    y_pred_flat = y_pred[:, :, col_idx].reshape(-1)

    if len(time_index) != len(y_true_flat):
        print("[!] The length of time_index does not match the number of values.")
        return

    df_actual = pd.Series(y_true_flat, index=time_index)
    df_pred = pd.Series(y_pred_flat, index=time_index)

    plt.figure(figsize=(12, 5))
    plt.plot(df_actual.index, df_actual, label='Actual', color='blue')
    plt.plot(df_pred.index, df_pred, label='Predicted', color='red', linestyle='--')
    plt.title(f'Compare actual and predicted values - {col_to_plot} (DeepVAR)')
    plt.xlabel('Time')
    plt.ylabel(col_to_plot)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def show_scaled_data(y_test_scaled_df, preds_scaled_df):
    """Hiển thị bảng & biểu đồ dữ liệu scaled"""
    st.subheader("Test data (scaled)")
    st.dataframe(y_test_scaled_df)

    st.subheader("Predictions (scaled)")
    st.dataframe(preds_scaled_df)

    st.subheader("Actual vs Predicted (scaled)")
    for col in y_test_scaled_df.columns:
        st.subheader(f"{col} (scaled)")

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(y_test_scaled_df[col].values, label="Actual", linewidth=2)
        ax.plot(preds_scaled_df[col].values, label="Predicted", linewidth=2, linestyle="--")
        ax.set_xlabel("Time")
        ax.set_ylabel(col)
        ax.legend()

        st.pyplot(fig)
        plt.close(fig)


def show_original_data(y_test_restore_df, pred_restore_df):
    """Hiển thị bảng & biểu đồ dữ liệu đã hoàn nguyên"""
    st.subheader("Test data (original scale)")
    st.dataframe(y_test_restore_df)

    st.subheader("Predictions (original scale)")
    st.dataframe(pred_restore_df)

    st.subheader("Actual vs Predicted (original scale)")

    min_len = min(len(y_test_restore_df), len(pred_restore_df))
    y_plot = y_test_restore_df.iloc[:min_len].reset_index(drop=True)
    pred_plot = pred_restore_df.iloc[:min_len].reset_index(drop=True)

    for col in y_plot.columns:
        st.subheader(f"{col} (original)")

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(y_plot[col].values, label="Actual", linewidth=2)
        ax.plot(pred_plot[col].values, label="Predicted", linewidth=2, linestyle="--")
        ax.set_xlabel("Time")
        ax.set_ylabel(col)
        ax.legend()

        st.pyplot(fig)
        plt.close(fig)

def view_data_selector(y_test_scaled_df, preds_scaled_df,
                       y_test_restore_df=None, pred_restore_df=None):
    """radio selector để chọn chế độ hiển thị"""
    st.subheader("Data View Options")

    options = ["Scaled (not restored)"]
    if (y_test_restore_df is not None) and (pred_restore_df is not None):
        options.append("Original scale (restored)")

    view_mode = st.radio(
        "Select data view:",
        options=options,
        horizontal=True,
    )

    if view_mode == "Scaled (not restored)":
        show_scaled_data(y_test_scaled_df, preds_scaled_df)
    else:
        show_original_data(y_test_restore_df, pred_restore_df)

def _model_display_name(model_type: str) -> str:
    mapping = {
        "VAR": "VAR",
        "DEEPVAR": "DeepVAR",
        "VAR-LSTM": "VAR-LSTM",
        "VAR_LAI_DEEPVAR": "VAR_Lai_DeepVAR",
    }
    return mapping.get(model_type, model_type)


def show_last_saved_training_results_box(model_type: str, file_name: str, note_if_missing_cols: str | None = None):
    display_name = _model_display_name(model_type)

    trained_cols_path = f"results/{model_type}/{file_name}/trained_columns.json"
    trained_cols = None

    if os.path.exists(trained_cols_path) and os.path.getsize(trained_cols_path) > 0:
        try:
            with open(trained_cols_path, "r", encoding="utf-8") as f:
                trained_cols = json.load(f)
        except Exception:
            trained_cols = None

    # ---- UI block giống style bạn đưa ----
    st.markdown(
        f"""
        <div style="
            background-color:#E8F5E9;
            border-left:5px solid #43A047;
            padding:10px;
            border-radius:6px;
            margin-top:10px;
        ">
            <b style="color:#1B5E20;">
                Dataset: <code>{file_name}</code>
            </b>
        </div>
        """,
        unsafe_allow_html=True,
    )


    # Không có file chứa cột → hiển thị note theo file_name hiện tại
    fallback = note_if_missing_cols or (
        f"Apply for: <b>{file_name}</b> weather dataset with all columns "
        "after first-order differencing (if non-stationary) and min-max normalization. "
        "No data augmentation techniques are performed."
    )

    st.markdown(
        f"""
        <div style="margin-top:12px;">
            <div style="
                background-color:#E3F2FD;
                border-left:5px solid #1E88E5;
                padding:10px;
                border-radius:6px;
                color:#0D47A1;
            ">
                {fallback}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

def plot_weather_forecast_single_variable(
    variable_name,
    history_df,
    forecast_df,
    title_prefix="Weather Forecast",
    save_path=None
):
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        history_df.index,
        history_df[variable_name],
        label="Historical data",
        linewidth=2
    )

    ax.scatter(
        forecast_df.index,
        forecast_df[variable_name],
        s=80,
        label="Forecast",
        zorder=5
    )

    ax.plot(
        [history_df.index[-1], forecast_df.index[0]],
        [history_df[variable_name].iloc[-1], forecast_df[variable_name].iloc[0]],
        linestyle="--",
        linewidth=1.5
    )

    ax.set_title(f"{title_prefix} – {variable_name}", fontsize=12)
    ax.set_xlabel("Date")
    ax.set_ylabel(variable_name)
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    st.pyplot(fig)
    plt.close(fig)