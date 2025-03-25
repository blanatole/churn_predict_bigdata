# Dự đoán Khả năng Khách hàng Rời khỏi Dịch vụ

## 📌 Giới thiệu
Dự án này nhằm dự đoán khả năng khách hàng rời khỏi dịch vụ dựa trên các đặc trưng như phương thức thanh toán, loại dịch vụ internet. Hệ thống được xây dựng trên nền tảng **Big Data với PySpark** để xử lý dữ liệu lớn và **MongoDB** để lưu trữ dữ liệu.

## 🎯 Mục tiêu
✔️ Xây dựng mô hình dự đoán khả năng khách hàng rời đi.  
✔️ Tạo giao diện thân thiện, dễ thao tác để tìm kiếm và thống kê dữ liệu.  
✔️ Tối ưu hóa quy trình xử lý dữ liệu và triển khai mô hình trên dữ liệu lớn.  
✔️ Tự động hóa pipeline từ tiền xử lý đến huấn luyện mô hình.

---

## 🛠 Công nghệ sử dụng
- **Ngôn ngữ lập trình**: Python
- **Framework xử lý dữ liệu lớn**: PySpark
- **Cơ sở dữ liệu**: MongoDB
- **Machine Learning**: Scikit-learn (Random ForestForest)
- **Giao diện**: Streamlit
- **Thư viện hỗ trợ**: Pandas, NumPy, Matplotlib, Seaborn,...

---

## 📂 Dataset
- **Tên tập dữ liệu**: [WA_Fn-UseC_-Telco-Customer-Churn.csv](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Các đặc trưng chính**:
  - `PaymentMethod`: Phương thức thanh toán (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic))
  - `InternetService`: Loại dịch vụ internet (DSL, Fiber optic, No)
  - `Churn`: Khả năng rời đi (Yes/No)

---

## 🚀 Các bước thực hiện
### 🔹 1. Tiền xử lý dữ liệu
✅ Đọc dữ liệu từ file CSV và lưu vào MongoDB.  
✅ Chuyển đổi các giá trị nhị phân từ `Yes/No` thành `0/1` để phù hợp với mô hình.  
✅ Xử lý dữ liệu bị thiếu và chuẩn hóa dữ liệu.  

### 🔹 2. Xây dựng mô hình
✅ Trích lọc các đặc trưng quan trọng.  
✅ Thử nghiệm nhiều mô hình Machine Learning khác nhau.  
✅ Tối ưu hóa mô hình bằng Grid Search hoặc Random Search.  

### 🔹 3. Dự đoán và đánh giá
✅ Chia tập dữ liệu thành tập huấn luyện và kiểm tra.  
✅ Đánh giá mô hình bằng các chỉ số như **Accuracy, Precision, Recall, F1-score**.  
✅ Triển khai mô hình trên tập dữ liệu mới.  

### 🔹 4. Xây dựng giao diện
✅ Hiển thị danh sách khách hàng với thông tin chi tiết.  
✅ Chức năng tìm kiếm theo **ID khách hàng, phương thức thanh toán, loại dịch vụ**.  
✅ Hiển thị biểu đồ thống kê dữ liệu khách hàng.  
✅ Dự đoán khả năng rời đi của khách hàng.  

---

## 🎯 Kết quả mong đợi
✨ Hệ thống có khả năng dự đoán chính xác khả năng khách hàng rời đi.  
✨ Cung cấp giao diện trực quan, dễ sử dụng để tra cứu và thống kê dữ liệu.  
✨ Tích hợp pipeline tự động hóa từ tiền xử lý đến huấn luyện mô hình.  

---

## 🔮 Hướng phát triển
🚀 Nâng cấp mô hình với dữ liệu thời gian thực.  
🚀 Tích hợp thêm các kỹ thuật **Deep Learning** để cải thiện hiệu suất.  
🚀 Mở rộng phạm vi dự đoán cho nhiều loại dịch vụ khác nhau.  

---

## ⚡ Hướng dẫn chạy dự án
### 🔹 1. Cài đặt Python
Tải và cài đặt **Python 3.11.0** từ [python.org](https://www.python.org/downloads/).

### 🔹 2. Tạo và kích hoạt môi trường ảo
```bash
python -m venv venv
venv/Scripts/activate  # Windows
source venv/bin/activate  # macOS/Linux
```

### 🔹 3. Cài đặt các thư viện cần thiết
```bash
pip install -r requirements.txt
```

### 🔹 4. Chạy hệ thống với Streamlit
```bash
streamlit run app.py
```

---

## 👥 Nhóm phát triển
- **Nguyễn Minh Ý** (Owner)  
- **Huỳnh Lý Tân Khoa**  
- **Nguyễn Thị Phương Anh**  
- **Võ Thị Như Ý**  

📧 Email liên hệ: `nguyenminhy7714@gmail.com`

