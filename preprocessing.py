import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from statsmodels.tsa.stattools import adfuller

# Kiểm tra thông tin tập dữ liệu
def check_information_dataset (dataset):
    # Thông tin kích thước tập dữ liệu
    print (dataset.shape)
    # Liệt kê các cột của tập dữ liệu
    print(dataset.columns)
    # Thông tin tổng quan về các trường dữ liệu
    print(dataset.info())
    # Thống kê mô tả (mean, std, min, max) của các trường số
    print(dataset.describe())

# Hàm đổi datetime -> số phút kể từ nửa đêm
def dt_to_minutes(dt):
    return dt.hour * 60 + dt.minute + dt.second / 60

# Encode chu kỳ cho hướng gió (0-360 độ)
def encode_wdir(df, col="winddir"):
    df[col + "_sin"] = np.sin(2 * np.pi * df[col] / 360)
    df[col + "_cos"] = np.cos(2 * np.pi * df[col] / 360)
    # Bỏ cột gốc
    df = df.drop(columns=[col])
    return df

# Tiền xử lý dữ liệu hoàn thiện
def preprocess_data(dataset_original, drop_threshold=1, zero_threshold = 0.95):
    # Tạo bản sao để tránh làm thay đổi dataset gốc
    dataset = dataset_original.copy(deep=True)
    # Chuyển đổi trường Date sang kiểu dữ liệu datetime để có thể sử dụng trong các phép toán thời gian.
    dataset['datetime'] = pd.to_datetime(dataset['datetime'], format='%d/%m/%y')
    # Đặt cột Date làm chỉ mục
    dataset.set_index('datetime', inplace=True)
    # Tạo cột biên độ dao động nhiệt
    dataset['temprange'] = dataset['tempmax'] - dataset['tempmin']
    # Quy đổi sunrise, sunset ra số phút tính từ lúc nửa đêm
    dataset['sunrise'] = pd.to_datetime(dataset['sunrise'], dayfirst=True)
    dataset['sunset']  = pd.to_datetime(dataset['sunset'],  dayfirst=True)
    dataset['sunrise'] = dataset['sunrise'].apply(dt_to_minutes)
    dataset['sunset']  = dataset['sunset'].apply(dt_to_minutes)
    # Gộp sunrise, sunset thành daylight
    dataset['daylight'] = dataset['sunset'] - dataset['sunrise']
    # Loại bỏ một số cột
    cols_to_drop = [
    "name", "tempmax", "tempmin", "precipprob", "preciptype", "snowdepth",
    "feelslikemax", "feelslikemin",
    "solarenergy", "moonphase",
    "conditions", "description",
    "icon", "stations", "sunrise", "sunset", "feelslike",
    "precipcover", "uvindex", "windgust", "snow"
    ]
    dataset.drop(columns=cols_to_drop, inplace=True)
    # Chuyển đổi dữ liệu thành dạng số, các giá trị không thể chuyển thành số sẽ bị thay bằng NaN
    for col in dataset.columns:
        if dataset[col].dtype == object: # correct type
            dataset[col] = pd.to_numeric(dataset[col].str.replace(',', '.'))
    dataset = dataset.apply(pd.to_numeric, errors='coerce')
    # Điền 0 cho riêng 2 cột severerisk và windgust
    dataset[['severerisk']] = dataset[['severerisk']].fillna(0)
    # Loại bỏ các cột có quá nhiều giá trị null
    missing_ratios = dataset.isnull().mean()
    cols_drop_null = missing_ratios[missing_ratios > drop_threshold].index
    zero_ratios = (dataset == 0).mean()          # Tính tỷ lệ giá trị 0 trên mỗi cột
    cols_drop_zero = zero_ratios[zero_ratios > zero_threshold].index
    columns_to_drop = cols_drop_null.union(cols_drop_zero)
    dataset.drop(columns=columns_to_drop, inplace=True)
    print("Đã loại bỏ các cột 95% giá trị null:", list(columns_to_drop))
    # Nội suy tuyến tính dựa trên khoảng cách thời gian
    dataset.interpolate(method='time', inplace=True)
    # Điền nốt giá trị còn thiếu nếu có
    dataset.fillna(dataset.mean(), inplace=True)  
    # Loại bỏ các hàng có chỉ mục NaN
    dataset = dataset[~dataset.index.isnull()]
    # Loại bỏ các dòng trùng lặp
    dataset.drop_duplicates(inplace=True)
    dataset = encode_wdir(dataset)
    dataset.drop(['severerisk','daylight'], axis=1, inplace=True)
    return dataset 

# Tiền xử lý dữ liệu cho ứng dụng dự báo
# Tiền xử lý dữ liệu cho ứng dụng dự báo
def preprocess_data_predict(dataset_original, drop_threshold=1, zero_threshold = 0.95):
    # Tạo bản sao để tránh làm thay đổi dataset gốc
    dataset = dataset_original.copy(deep=True)
    # Chuyển đổi trường Date sang kiểu dữ liệu datetime để có thể sử dụng trong các phép toán thời gian.
    dataset['datetime'] = pd.to_datetime(dataset['datetime'], format='%d/%m/%Y')
    # Đặt cột Date làm chỉ mục
    dataset.set_index('datetime', inplace=True)
    # Tạo cột biên độ dao động nhiệt
    dataset['temprange'] = dataset['tempmax'] - dataset['tempmin']
    # Quy đổi sunrise, sunset ra số phút tính từ lúc nửa đêm
    dataset['sunrise'] = pd.to_datetime(dataset['sunrise'], format='ISO8601')
    dataset['sunset']  = pd.to_datetime(dataset['sunset'], format='ISO8601')
    dataset['sunrise'] = dataset['sunrise'].apply(dt_to_minutes)
    dataset['sunset']  = dataset['sunset'].apply(dt_to_minutes)
    # Gộp sunrise, sunset thành daylight
    dataset['daylight'] = dataset['sunset'] - dataset['sunrise']
    # Điền 0 cho riêng 2 cột severerisk và windgust
    dataset[['severerisk']] = dataset[['severerisk']].fillna(0)
    # Loại bỏ một số cột
    cols_to_drop = [
    "name", "tempmax", "tempmin", "precipprob", "preciptype", "snowdepth",
    "feelslikemax", "feelslikemin",
    "solarenergy", "moonphase",
    "conditions", "description",
    "icon", "stations", "sunrise", "sunset", "feelslike",
    "precipcover", "uvindex", "windgust", 'severerisk', "snow", 'daylight'
    ]
    dataset.drop(columns=cols_to_drop, inplace=True)
    # Chuyển đổi dữ liệu thành dạng số, các giá trị không thể chuyển thành số sẽ bị thay bằng NaN
    for col in dataset.columns:
        if dataset[col].dtype == object: # correct type
            dataset[col] = pd.to_numeric(dataset[col].str.replace(',', '.'))
    dataset = dataset.apply(pd.to_numeric, errors='coerce')
    # Loại bỏ các cột có quá nhiều giá trị null
    missing_ratios = dataset.isnull().mean()
    cols_drop_null = missing_ratios[missing_ratios > drop_threshold].index
    zero_ratios = (dataset == 0).mean()          # Tính tỷ lệ giá trị 0 trên mỗi cột
    cols_drop_zero = zero_ratios[zero_ratios > zero_threshold].index
    columns_to_drop = cols_drop_null.union(cols_drop_zero)
    dataset.drop(columns=columns_to_drop, inplace=True)
    print("Đã loại bỏ các cột 95% giá trị null:", list(columns_to_drop))
    # Nội suy tuyến tính dựa trên khoảng cách thời gian
    dataset.interpolate(method='time', inplace=True)
    # Điền nốt giá trị còn thiếu nếu có
    dataset.fillna(dataset.mean(), inplace=True)  
    # Loại bỏ các hàng có chỉ mục NaN
    dataset = dataset[~dataset.index.isnull()]
    # Loại bỏ các dòng trùng lặp
    dataset.drop_duplicates(inplace=True)
    dataset = encode_wdir(dataset)
    return dataset

# Tiền xử lý dữ liệu cho ứng dụng dự báo
def preprocess_data_restore (dataset_original, drop_threshold=1, zero_threshold = 0.95):
    # Tạo bản sao để tránh làm thay đổi dataset gốc
    dataset = dataset_original.copy(deep=True)
    # Chuyển đổi trường Date sang kiểu dữ liệu datetime để có thể sử dụng trong các phép toán thời gian.
    dataset['datetime'] = pd.to_datetime(dataset['datetime'], format='%d/%m/%Y')
    # Đặt cột Date làm chỉ mục
    dataset.set_index('datetime', inplace=True)
    # Tạo cột biên độ dao động nhiệt
    dataset['temprange'] = dataset['tempmax'] - dataset['tempmin']
    # Quy đổi sunrise, sunset ra số phút tính từ lúc nửa đêm
    dataset['sunrise'] = pd.to_datetime(dataset['sunrise'], format='ISO8601')
    dataset['sunset']  = pd.to_datetime(dataset['sunset'], format='ISO8601')
    dataset['sunrise'] = dataset['sunrise'].apply(dt_to_minutes)
    dataset['sunset']  = dataset['sunset'].apply(dt_to_minutes)
    # Gộp sunrise, sunset thành daylight
    dataset['daylight'] = dataset['sunset'] - dataset['sunrise']
    # Điền 0 cho riêng 2 cột severerisk và windgust
    dataset[['severerisk']] = dataset[['severerisk']].fillna(0)
    # Loại bỏ một số cột
    cols_to_drop = [
    "name", "tempmax", "tempmin", "precipprob", "preciptype", "snowdepth",
    "feelslikemax", "feelslikemin",
    "solarenergy", "moonphase",
    "conditions", "description",
    "icon", "stations", "sunrise", "sunset", "feelslike",
    "precipcover", "uvindex", "windgust", 'severerisk', "snow", 'daylight'
    ]
    dataset.drop(columns=cols_to_drop, inplace=True)
    # Chuyển đổi dữ liệu thành dạng số, các giá trị không thể chuyển thành số sẽ bị thay bằng NaN
    for col in dataset.columns:
        if dataset[col].dtype == object: # correct type
            dataset[col] = pd.to_numeric(dataset[col].str.replace(',', '.'))
    dataset = dataset.apply(pd.to_numeric, errors='coerce')
    # Loại bỏ các cột có quá nhiều giá trị null
    missing_ratios = dataset.isnull().mean()
    cols_drop_null = missing_ratios[missing_ratios > drop_threshold].index
    zero_ratios = (dataset == 0).mean()          # Tính tỷ lệ giá trị 0 trên mỗi cột
    cols_drop_zero = zero_ratios[zero_ratios > zero_threshold].index
    columns_to_drop = cols_drop_null.union(cols_drop_zero)
    dataset.drop(columns=columns_to_drop, inplace=True)
    print("Đã loại bỏ các cột 95% giá trị null:", list(columns_to_drop))
    # Nội suy tuyến tính dựa trên khoảng cách thời gian
    dataset.interpolate(method='time', inplace=True)
    # Điền nốt giá trị còn thiếu nếu có
    dataset.fillna(dataset.mean(), inplace=True)  
    # Loại bỏ các hàng có chỉ mục NaN
    dataset = dataset[~dataset.index.isnull()]
    # Loại bỏ các dòng trùng lặp
    dataset.drop_duplicates(inplace=True)
    return dataset

# Phát sinh dữ liệu
# Thêm nhiễu Gaussian (phân phối chuẩn) vào một chuỗi thời gian.
def add_gaussian_noise(time_series, mean, stddev):
    noise = np.random.normal(mean, stddev, len(time_series))  # Tạo nhiễu Gaussian
    noisy_series = time_series + noise  # Thêm nhiễu vào dữ liệu gốc
    return noisy_series


# Sinh tập chỉ mục thời gian mới với số lượng periods bằng số lượng dòng của dữ liệu gốc
def generate_new_dates(df, periods):
    first_date = df.index.min()  # Lấy ngày nhỏ nhất trong dữ liệu gốc
    return pd.date_range(end=first_date - pd.DateOffset(days=1),
                         periods=periods,
                         freq='D')  # Sinh danh sách ngày mới, cách nhau 1 ngày


# Tạo một bản sao dữ liệu gốc có nhiễu Gaussian và ghép vào dữ liệu ban đầu
# stddev = 0.1 * data.std()  # Chọn 10% độ lệch chuẩn của dữ liệu gốc
# mean=0.0
def augment_with_gaussian(data, mean, stddev): 
    augmented_datasets = []
    augmented_data = data.copy()
        
    for column in data.columns:
        if np.issubdtype(data[column].dtype, np.number):  # Kiểm tra cột có phải là số không
            augmented_data[column] = add_gaussian_noise(
                data[column].dropna(), mean, stddev)
        
    new_dates = generate_new_dates(data, len(data))  # Sinh index mới
    augmented_data.index = new_dates  # Gán index mới cho dữ liệu có nhiễu
    augmented_datasets.append(augmented_data)
        
    return pd.concat([data] + augmented_datasets, axis=0).sort_index()  # Hợp nhất và sắp xếp index


# Phát sinh dữ liệu mới bằng phương pháp Numpy dựa trên xu hướng (trend), mùa vụ (seasonality) và nhiễu Gaussian (noise)
def augment_timeseries_data(data, n_periods):    
    new_dates = generate_new_dates(data, n_periods)
    augmented_data = []
    
    for column in data.columns:
        # Tính toán thống kê của dữ liệu gốc
        mean = data[column].mean()
        std = data[column].std()
        trend = np.polyfit(range(len(data)), data[column].values, 1)[0]
        
        # Tạo thành phần xu hướng
        base_trend = -np.arange(n_periods)[::-1] * trend + mean
        # Tạo thành phần mùa vụ
        seasonality = np.sin(np.linspace(0, 2*np.pi, 12)) * std * 0.5
        # Thêm nhiễu Gaussian
        noise = np.random.normal(0, std * 0.1, n_periods)
        
        # Tạo chuỗi thời gian mới
        # Chuỗi mới có xu hướng, mùa vụ và nhiễu giống dữ liệu gốc
        new_series = pd.Series(
            base_trend + np.tile(seasonality, n_periods//12 + 1)[:n_periods] + noise,
            index=new_dates
        )
        # Ghép dữ liệu mới vào dữ liệu gốc & sắp xếp lại theo thời gian.
        augmented_data.append(new_series)
    
    return (pd.concat([pd.DataFrame(dict(zip(data.columns, augmented_data))), data])).sort_index()

# Kiểm tra tính dừng
def adf_test(series, title='', alpha=0.05):
    series_clean = series.dropna()
    result = adfuller(series_clean)
    p_value = result[1]
    
    return {
        'Column': title,
        'ADF Statistic': result[0],
        'p-value': p_value,
        'Stationary': 'Yes' if p_value < alpha else 'No'
        # Nếu muốn, có thể thêm 'alpha': alpha để tiện debug / hiển thị
        # 'alpha': alpha
    }

# In kết quả kiểm tra tính dừng
def check_stationarity(data, alpha=0.05):
    results = []
    
    for column in data.columns:
        result = adf_test(data[column], title=column, alpha=alpha)
        results.append(result)

    return pd.DataFrame(results)

# Lấy sai phân bậc 1
def difference_series(series, lag=1):
    return series.diff(periods=lag).dropna()

# Dừng hoá dữ liệu bằng cách lấy sai phân bậc 1
def make_stationary0(data, lag=1):
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame.")
    
    non_numeric_columns = data.select_dtypes(exclude=['number']).columns
    if not non_numeric_columns.empty:
        print(f"Warning: Non-numeric columns excluded: {list(non_numeric_columns)}")

    numeric_data = data.select_dtypes(include=['number'])
    differenced_data = numeric_data.apply(difference_series, lag=lag)

    return differenced_data

# Dừng hoá dữ liệu bằng cách lấy sai phân bậc 1 tùy trường hợp
def make_stationary (df, mode, lag=1, alpha=0.05):
    df_diff = df.copy()
    adf_results = {}
    for col in df.columns:
        # Bỏ qua cyclical features
        if col.endswith("_sin") or col.endswith("_cos"):
            continue
        # Kiểm tra ADF test
        try:
            adf_pvalue = adfuller(df[col].dropna(), autolag="AIC")[1]
        except Exception:
            adf_pvalue = np.nan
        adf_results[col] = adf_pvalue
        # Sai phân theo chế độ
        if mode == "all":
            df_diff[col] = df[col].diff(periods=lag)
        elif mode == "non-stationary":
            if adf_pvalue >= alpha:  # không dừng
                df_diff[col] = df[col].diff(periods=lag)
    return df_diff.dropna()

# Chuẩn hoá dữ liệu bằng min-max    
def min_max_normalize(min, max, dataset):
    # chọn cột không phải sin/cos
    cols_to_scale = [col for col in dataset.columns 
                     if not col.endswith("_sin") and not col.endswith("_cos")]
    
    scaler = MinMaxScaler(feature_range=(min, max))
    scaled_values = scaler.fit_transform(dataset[cols_to_scale])
    
    scaled_data = dataset.copy()
    scaled_data[cols_to_scale] = scaled_values
    
    return scaled_data, scaler


# Chuẩn hoá dữ liệu bằng z-score
def z_score_normalize(dataset):
    # chỉ chọn cột không phải sin/cos
    cols_to_scale = [col for col in dataset.columns 
                     if not col.endswith("_sin") and not col.endswith("_cos")]
    
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(dataset[cols_to_scale])
    
    scaled_data = dataset.copy()
    scaled_data[cols_to_scale] = scaled_values
    
    return scaled_data, scaler

# Hoàn nguyên chuẩn hóa bỏ qua cột sin/cos
def inverse_normalize(predictions, scaler, dataset_columns):
    cols_to_scale = [col for col in dataset_columns 
                     if not col.endswith("_sin") and not col.endswith("_cos")]

    reshaped_preds = predictions.reshape(-1, predictions.shape[-1])

    preds_df = pd.DataFrame(reshaped_preds, columns=dataset_columns)
    preds_df[cols_to_scale] = scaler.inverse_transform(preds_df[cols_to_scale])

    return preds_df  # 2D dataframe

# Hoàn nguyên sai phân bậc 1, bỏ qua các cột sin/cos hoặc biến không sai phân
def inverse_difference(differenced, original, lag=1, exclude_cols=None):
    if exclude_cols is None:
        exclude_cols = ["winddir_sin", "winddir_cos"]

    restored = pd.DataFrame(index=differenced.index, columns=differenced.columns)

    for col in differenced.columns:
        if col in exclude_cols:
            # Giữ nguyên vì không áp dụng sai phân
            restored[col] = differenced[col]
        else:
            # Hoàn nguyên bằng cộng dồn
            last_values = original[col].iloc[-lag:]
            inv = []
            for diff in differenced[col]:
                restored_value = diff + last_values.iloc[-1]
                inv.append(restored_value)
                last_values = pd.concat([last_values, pd.Series([restored_value])], ignore_index=True)
            restored[col] = inv

    return restored  # 2D dataframe

# Chuyển winddir_sin/cos thành winddir
def reconstruct_wind_direction(df):
    if 'winddir_sin' in df.columns and 'winddir_cos' in df.columns:
        df['winddir'] = (np.degrees(np.arctan2(df['winddir_sin'], df['winddir_cos'])) + 360) % 360
        # Xoá 2 cột gốc để tránh trùng lặp
        df = df.drop(columns=['winddir_sin', 'winddir_cos'])
    return df

# Hoàn nguyên dự đoán
def inverse_transformation(predictions, scaler, dataset_columns, original_dataset,
                      lag=1, exclude_cols=None, flag_diff=False):
    
    if isinstance(predictions, pd.DataFrame):
        predictions = predictions.values

    if exclude_cols is None:
        exclude_cols = ["winddir_sin", "winddir_cos"]

    # ===== B1: Hoàn nguyên chuẩn hóa =====
    cols_to_scale = [col for col in dataset_columns
                     if not col.endswith("_sin") and not col.endswith("_cos")]
    reshaped_preds = predictions.reshape(-1, predictions.shape[-1])
    preds_df = pd.DataFrame(reshaped_preds, columns=dataset_columns)
    preds_df[cols_to_scale] = scaler.inverse_transform(preds_df[cols_to_scale])

    # ===== B2: Hoàn nguyên sai phân (nếu có) =====
    if flag_diff:
        restored_df = inverse_difference(preds_df, original_dataset, lag=lag, exclude_cols=exclude_cols)
    else:
        restored_df = preds_df.copy()

    # ===== B3: Tính lại hướng gió =====
    restored_df = reconstruct_wind_direction(restored_df)

    return restored_df  # 2D DataFrame