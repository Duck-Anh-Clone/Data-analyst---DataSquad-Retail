import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

file_path = "marketing_campaign.csv"

try:
    customer_data = pd.read_csv(file_path, sep='\t')
    print("Dữ liệu đã tải thành công!")
except Exception as e:
    print(f"Lỗi khi đọc file: {e}")

customer_data.info()

# Kiểm tra giá trị thiếu
def check_missing(df):
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("Dữ liệu không có giá trị thiếu.")
    else:
        print("Cột và số lượng giá trị thiếu:")
        print(missing[missing > 0])
check_missing(customer_data)

# Kiểm tra và điền giá trị thiếu trong 'Income'
customer_data['Income'].fillna(customer_data['Income'].median(), inplace=True)

# Kiểm tra giá trị trùng lặp
print(f"Số dòng bị trùng lặp: {customer_data.duplicated().sum()}")
customer_data.drop_duplicates(inplace=True)

# Kiểm tra cột có giá trị duy nhất
constant_columns = [col for col in customer_data.columns if customer_data[col].nunique() == 1]
customer_data.drop(columns=constant_columns, inplace=True)

# Xử lý ngoại lệ bằng IQR
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

customer_data = remove_outliers(customer_data, 'Income')

# Chuẩn hóa cột Education
education_map = {
    'Graduation': 'Tốt nghiệp',
    'PhD': 'Tiến sĩ',
    'Master': 'Thạc sĩ',
    'Basic': 'Cơ bản',
    '2n Cycle': 'Chu kỳ 2'
}
customer_data['Education'] = customer_data['Education'].map(education_map).fillna('Khác')

# Chuẩn hóa cột Marital_Status
marital_status_map = {
    'Single': 'Độc thân', 'Together': 'Sống chung', 'Married': 'Sống chung',
    'Divorced': 'Độc thân', 'Widow': 'Độc thân', 'Alone': 'Độc thân',
    'Absurd': 'Độc thân', 'YOLO': 'Độc thân'
}
customer_data['Marital_Status'] = customer_data['Marital_Status'].map(marital_status_map).fillna('Không xác định')

# Chuyển đổi kiểu dữ liệu
customer_data['Dt_Customer'] = pd.to_datetime(customer_data['Dt_Customer'], format='%d-%m-%Y', errors='coerce')
customer_data['ID'] = customer_data['ID'].astype(str)
boolean_columns = ['AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1', 'AcceptedCmp2', 'Complain', 'Response']
customer_data[boolean_columns] = customer_data[boolean_columns].astype(bool)

# Đổi tên cột
column_rename = {
    'ID': 'Mã_KH', 'Education': 'Trình_Độ_Học_Vấn', 'Marital_Status': 'Tình_Trạng_Hôn_Nhân',
    'Income': 'Thu_Nhập', 'Recency': 'Khoảng_Cách_Lần_Mua_Cuối',
    'MntWines': 'Chi_Tiêu_Rượu', 'MntFruits': 'Chi_Tiêu_Trái_Cây',
    'MntMeatProducts': 'Chi_Tiêu_Thịt', 'MntFishProducts': 'Chi_Tiêu_Cá',
    'MntSweetProducts': 'Chi_Tiêu_Bánh_Kẹo', 'MntGoldProds': 'Chi_Tiêu_Vàng_Bạc',
    'NumDealsPurchases': 'Số_Lần_Mua_Khuyến_Mãi', 'NumWebPurchases': 'Số_Lần_Mua_Qua_Web',
    'NumCatalogPurchases': 'Số_Lần_Mua_Qua_Catalog', 'NumStorePurchases': 'Số_Lần_Mua_Tại_Cửa_Hàng',
    'NumWebVisitsMonth': 'Số_Lần_Truy_Cập_Web', 'AcceptedCmp3': 'Chấp_Nhận_Chiến_Dịch_3',
    'AcceptedCmp4': 'Chấp_Nhận_Chiến_Dịch_4', 'AcceptedCmp5': 'Chấp_Nhận_Chiến_Dịch_5',
    'AcceptedCmp1': 'Chấp_Nhận_Chiến_Dịch_1', 'AcceptedCmp2': 'Chấp_Nhận_Chiến_Dịch_2',
    'Complain': 'Khiếu_Nại', 'Response': 'Phản_Hồi'
}
customer_data.rename(columns=column_rename, inplace=True)

# Tạo cột mới
customer_data['Tổng_Số_Con'] = customer_data['Kidhome'] + customer_data['Teenhome']
customer_data['Tuổi_KH'] = pd.to_datetime('today').year - customer_data['Year_Birth']
customer_data['Tổng_Chi_Tiêu'] = customer_data[['Chi_Tiêu_Rượu', 'Chi_Tiêu_Trái_Cây', 'Chi_Tiêu_Thịt', 'Chi_Tiêu_Cá', 'Chi_Tiêu_Bánh_Kẹo', 'Chi_Tiêu_Vàng_Bạc']].sum(axis=1)
customer_data['Thâm_Niên_KH'] = ((pd.to_datetime('today') - customer_data['Dt_Customer']).dt.days / 30.44).round()

# Nhóm tuổi
customer_data['Nhóm_Tuổi'] = pd.cut(customer_data['Tuổi_KH'], bins=[0, 35, 60, float('inf')], labels=['Thanh niên', 'Trưởng thành', 'Cao tuổi'])

# Nhóm thu nhập
customer_data['Nhóm_Thu_Nhập'] = pd.qcut(customer_data['Thu_Nhập'], q=4, labels=['Thu nhập thấp', 'Thu nhập trung bình thấp', 'Thu nhập trung bình cao', 'Thu nhập cao'])

# Lưu dữ liệu đã làm sạch
customer_data.to_csv('Customer_data_cleaned.csv', index=False, encoding='utf-8-sig')
print("Dữ liệu đã được làm sạch và lưu thành công!")
