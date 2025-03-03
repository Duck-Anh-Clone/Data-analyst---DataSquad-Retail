import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, classification_report, roc_auc_score
from dash import Dash, dcc, html, Input, Output
import plotly.express as px

# 📌 Load dữ liệu
file_path = "Customer_data_cleaned.csv"
clean_data = pd.read_csv(file_path)

# 🔍 Kiểm tra tổng quan dữ liệu
def overview_data(df):
    print("\n📊 Thông tin dữ liệu:")
    print(clean_data.info())
    print("\n🔹 Mô tả dữ liệu:")
    print(clean_data.describe())
    print("\n🔎 Giá trị thiếu:")
    print(clean_data.isnull().sum())
overview_data(clean_data)

#📊 Ma trận tương quan của các doanh mục
numerical_data = clean_data.select_dtypes(include=[np.number])

# Tính toán ma trận tương quan
correlation_matrix = numerical_data.corr()

# Trực quan hóa ma trận tương quan
plt.figure(figsize=(15, 10))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Ma trận tương quan giữa các biến", fontsize=16)
plt.show()

# 🔍 Tổng chi tiêu của các sản phẩm
chi_tieu_sp = clean_data[
    ['Chi_Tiêu_Rượu', 'Chi_Tiêu_Trái_Cây', 'Chi_Tiêu_Thịt',
                             'Chi_Tiêu_Cá', 'Chi_Tiêu_Bánh_Kẹo', 'Chi_Tiêu_Vàng_Bạc']].sum()
chi_tieu_sp.sort_values(ascending=False, inplace=True)

plt.figure(figsize=(10, 6))
colors = plt.cm.tab10(range(len(chi_tieu_sp)))
ax = chi_tieu_sp.plot.bar(color = colors, legend=False)

plt.title("Tổng chi tiêu của các sản phẩm", fontsize=16, fontweight='bold')
plt.xlabel("Sản phẩm")
plt.ylabel("Tổng chi tiêu")

for i, value in enumerate(chi_tieu_sp.values):
    plt.text(i, value + 0.1, str(value), ha='center')

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=45, ha="right")
plt.show()


# 📊 Biểu đồ tỷ lệ khiếu nại của khách hàng
# Tính toán số lượng cho từng giá trị trong cột 'Khiếu_Nại'
complaint_counts = clean_data['Khiếu_Nại'].value_counts()

# Định nghĩa nhãn và màu sắc cho biểu đồ 'Khiếu_Nại'
labels_complaint = ['Không khiếu nại', 'Có khiếu nại']
colors_complaint = ['#2ecc71', '#e74c3c']

# Tạo biểu đồ hình tròn cho cột 'Khiếu_Nại'
plt.figure(figsize=(8, 6))
plt.pie(complaint_counts, labels=labels_complaint, colors=colors_complaint, autopct='%1.1f%%', startangle=90, explode=(0, 0.1))
plt.title('Tỷ lệ khiếu nại của Khách hàng')
plt.axis('equal')
plt.show()

# 🔹Tỉ lệ phản hồi của khách hàng
response_counts = clean_data['Phản_Hồi'].value_counts() # Tính toán số lượng cho từng giá trị trong cột 'Phản_Hồi'

# Định nghĩa nhãn và màu sắc cho biểu đồ 'Phản_Hồi'
labels_response = ['Không phản hồi', 'Có phản hồi']
colors_response = ['#3498db', '#f1c40f']

# Tạo biểu đồ hình tròn cho cột 'Phản_Hồi'
plt.figure(figsize=(8, 6))
plt.pie(response_counts, labels=labels_response, colors=colors_response, autopct='%1.1f%%', startangle=90, explode=(0, 0.1))
plt.title('Tỷ lệ phản hồi của Khách hàng')
plt.axis('equal')
plt.show()

# 📌 Tạo Dashboard với Dash
marketing_campaigns = ['Chấp_Nhận_Chiến_Dịch_1', 'Chấp_Nhận_Chiến_Dịch_2',
        'Chấp_Nhận_Chiến_Dịch_3', 'Chấp_Nhận_Chiến_Dịch_4', 'Chấp_Nhận_Chiến_Dịch_5']

campaign_acceptance_counts = clean_data[marketing_campaigns].sum()

highest_campaign = campaign_acceptance_counts.idxmax()

income_vs_campaign = clean_data.groupby('Nhóm_Thu_Nhập')[marketing_campaigns].sum()

app = Dash(__name__)

app.layout = html.Div([
    html.H1("Dashboard: Hiệu quả các chiến dịch Marketing", style={'text-align': 'center'}),

    # Bố cục 2 cột, 2 hàng
    html.Div([
        # Viz 1: Số lượng chấp nhận của các chiến dịch
        html.Div([
            dcc.Graph(
                id='viz1',
                figure=px.bar(
                    campaign_acceptance_counts,
                    x=campaign_acceptance_counts.index,
                    y=campaign_acceptance_counts.values,
                    title="Số lượng chấp nhận của các Chiến dịch",
                    labels={"x": "Chiến dịch", "y": "Số lượng chấp nhận"},
                    color=campaign_acceptance_counts.index,
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
            )
        ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'}),

        # Viz 2: So sánh toàn bộ chiến dịch (làm nổi bật chiến dịch cao nhất)
        html.Div([
            dcc.Graph(
                id='viz2',
                figure=px.bar(
                    income_vs_campaign,
                    barmode="group",
                    title="So sánh tất cả các chiến dịch với Nhóm Thu nhập",
                    labels={"index": "Chiến dịch", "value": "Số lượng"}
                ).update_traces(
                    marker_color=['#FF5733' if col == highest_campaign else '#3498DB' for col in income_vs_campaign.index]
                )
            )
        ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'}),
    ]),

    html.Div([
        # Viz 3: Heatmap mối quan hệ giữa Phản Hồi và Khiếu Nại (di chuyển lên trên)
        html.Div([
            dcc.Graph(
                id='viz3',
                figure=px.imshow(
                    pd.crosstab(clean_data['Khiếu_Nại'], clean_data['Phản_Hồi']),
                    text_auto=True,
                    color_continuous_scale='Blues',
                    title="Mối quan hệ giữa Phản Hồi và Khiếu Nại",
                    labels=dict(x="Phản Hồi", y="Khiếu Nại", color="Số lượng")
                )
            )
        ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'}),

        # Viz 4: Boxplot cho Tổng Chi Tiêu so với Nhóm Thu Nhập và Khiếu Nại
        html.Div([
            dcc.Graph(
                id='viz4-boxplot',
                figure=px.box(
                    clean_data,
                    x='Nhóm_Thu_Nhập',  
                    y='Tổng_Chi_Tiêu', 
                    color='Khiếu_Nại', 
                    title="So sánh Tổng Chi Tiêu giữa Nhóm Thu Nhập và Khiếu Nại",
                    labels={"Nhóm_Thu_Nhập": "Nhóm Thu Nhập", "Tổng_Chi_Tiêu": "Tổng Chi Tiêu", "Khiếu_Nại": "Khiếu Nại"},
                    category_orders={"Khiếu_Nại": ["True", "False"]}, 
                    height=600
                )
            )
        ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'}),
    ])
])

if __name__ == '__main__':
    app.run_server(debug=True, port=8051)

# 📌 Mô hình dự báo tổng chi tiêu
X = numerical_data.drop(columns=['Tổng_Chi_Tiêu'])
y = numerical_data['Tổng_Chi_Tiêu']

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Khởi tạo mô hình hồi quy tuyến tính và huấn luyện
model = LinearRegression()
model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred_linear = model.predict(X_test)

# Đánh giá hiệu suất của mô hình
mae = mean_absolute_error(y_test, y_pred_linear)
mse = mean_squared_error(y_test, y_pred_linear)
r2 = r2_score(y_test, y_pred_linear)

print("Sai số tuyệt đối trung bình (MAE):", mae)
print("Sai số bình phương trung bình (MSE):", mse)
print("Hệ số xác định (R^2):", r2)

# Tính toán residuals
residuals = y_test - y_pred_linear

plt.figure(figsize=(8, 6))
plt.scatter(y_pred_linear, residuals, color='blue', alpha=0.6)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Giá trị dự đoán')
plt.ylabel('Residuals (Sai số)')
plt.title('Biểu đồ Residuals - Hồi quy tuyến tính')
plt.show()

# 📌 Mô hình dự báo phản hồi marketing
X = clean_data[["Tuổi_KH", "Thu_Nhập", "Tổng_Chi_Tiêu"]]
y = clean_data["Phản_Hồi"]

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Xây dựng mô hình hồi quy logistic
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred_logistic = logistic_model.predict(X_test)

# Đánh giá mô hình
accuracy = accuracy_score(y_test, y_pred_logistic)

print(f"Accuracy: {accuracy}")
print(classification_report(y_test, y_pred_logistic))

# Tính toán confusion matrix
cm_logistic = confusion_matrix(y_test, y_pred_logistic)

disp_logistic = ConfusionMatrixDisplay(confusion_matrix=cm_logistic, display_labels=["Không phản hồi", "Phản hồi"])
disp_logistic.plot(cmap=plt.cm.Blues)

plt.title("Confusion Matrix - Logistic Regression")
plt.show()
