# 📊 Phân Tích Hành Vi Khách Hàng - DataSquad Retail

## 📝 Giới thiệu
Dự án này nhằm phân tích dữ liệu khách hàng của Công ty **DataSquad Retail** với mục tiêu chính là:
- Tìm hiểu **hành vi mua sắm của khách hàng**.
- **Phân loại khách hàng** theo các nhóm dựa trên nhân khẩu học và hành vi tiêu dùng.
- Đánh giá hiệu quả **các chiến dịch marketing**.
- Đề xuất **chiến lược tối ưu** để cải thiện doanh thu và trải nghiệm khách hàng.

## 🚀 Mục tiêu
- **Xác định phân khúc khách hàng** từ khả năng chi tiêu cho các doanh mục sản phẩm
- **Dự đoán phản hồi khách hàng** với mô hình Machine Learning.
- **Phân tích xu hướng tiêu dùng** dựa trên dữ liệu lịch sử.
- **Xây dựng Dashboard Tableau** trực quan hóa dữ liệu.

## 🛠 Công nghệ sử dụng
- **Ngôn ngữ:** Python, SQL
- **Thư viện:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
- **Cơ sở dữ liệu:** SQL Server
- **Công cụ trực quan hóa:** Tableau.

## 📂 Dữ liệu
- **Nguồn dữ liệu:** Từ hệ thống CRM của DataSquad Retail.
- **Các trường dữ liệu chính:**
  - `CustomerID`: Mã khách hàng
  - `Age`: Độ tuổi
  - `Gender`: Giới tính
  - `Income`: Thu nhập hàng năm
  - `Spending Score`: Điểm chi tiêu
  - `Purchase History`: Lịch sử mua hàng
  - `Marketing Response`: Phản hồi với chiến dịch quảng cáo

## 🔎 Phương pháp phân tích
1. **Tiền xử lý dữ liệu:** Làm sạch, xử lý dữ liệu thiếu, chuẩn hóa dữ liệu.
2. **Phân cụm khách hàng:** Phân nhóm khách hàng theo thu nhập.
3. **Xây dựng mô hình dự báo:** Hồi quy tuyến tính & Phương pháp hồi quy Logistic(LogisticRegression) để dự đoán phản hồi và chi tiêu khách hàng.
4. **Trực quan hóa dữ liệu:** Dùng Tableau để tạo báo cáo động.

## 🖥️ Cách chạy dự án
### 1️⃣ Cài đặt thư viện cần thiết
```bash
pip install -r requirements.txt
```

### 2️⃣ Chạy file phân tích dữ liệu
```bash
python Check_data.py
python Analyst.py
```

### 3️⃣ Xem Dashboard
- **Tableau Public Dashboard**: [Xem tại đây](https://public.tableau.com/views/Customer_analyst/Dashboard3?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)

## 📊 Kết quả & Insights
- **Phân khúc khách hàng:**
  - **Nhóm Chi tiêu theo độ tuổi:** Nhắm vào sản phẩm phù hợp với độ tuổi.
  - **Nhóm mua hàng trên các kênh mua sắm:** Khách hàng tập trung nhiều vào mua sắm tại cửa hàng, lượt truy cập wed cao cho thấy độ quan tâm tốt.
  - **Nhóm Chi tiêu cho từng doanh mục sản phẩm:** Đề xuất sản phẩm được mua nhiều.
- **Hiệu quả chiến dịch marketing:** Chiến dịch 4 có tỷ lệ phản hồi cao nhất.
- **Dự đoán phản hồi khách hàng:** Random Forest cho kết quả tốt nhất với độ chính xác 85%.

## 🔗 Liên kết dự án
- 📂 **GitHub Repository**: [Repo GitHub](https://github.com/Duck-Anh-Clone/Data-analyst---DataSquad-Retail)
- 📈 **Dashboard Tableau Public**: [Xem tại đây](https://public.tableau.com/views/Customer_analyst/Dashboard3?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)

## 📬 Liên hệ
Nếu bạn có bất kỳ câu hỏi nào, vui lòng liên hệ qua email: `nguyenducanh4404@gmail.com`
 