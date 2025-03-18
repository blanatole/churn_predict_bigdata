# README: Dự đoán Khả năng Khách hàng Rời khỏi Dịch vụ

## 1. Giới thiệu
Dự án này nhằm dự đoán khả năng khách hàng rời khỏi dịch vụ dựa trên các đặc trưng như phương thức thanh toán, loại dịch vụ internet, thời gian sử dụng, và các yếu tố khác. Hệ thống được xây dựng trên nền tảng Big Data với PySpark để xử lý dữ liệu lớn và MongoDB để lưu trữ dữ liệu.

## 2. Mục tiêu
- Xây dựng mô hình dự đoán khả năng khách hàng rời đi.
- Xây dựng giao diện giúp người dùng tìm kiếm, thống kê dữ liệu dễ dàng.
- Tối ưu hóa quy trình xử lý dữ liệu và triển khai mô hình trên dữ liệu lớn.
- Tự động hóa pipeline từ tiền xử lý đến huấn luyện mô hình.

## 3. Công nghệ sử dụng
- **Ngôn ngữ lập trình**: Python
- **Framework xử lý dữ liệu lớn**: PySpark
- **Cơ sở dữ liệu**: MongoDB
- **Machine Learning**: Scikit-learn, XGBoost
- **Giao diện**: Streamlit 
- **Các thư viện hỗ trợ**: Pandas, NumPy, Matplotlib, Seaborn,...

## 4. Dataset
- Tập dữ liệu: `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- Các đặc trưng chính:
  - Phương thức thanh toán
  - Loại dịch vụ internet
  - Thời gian sử dụng dịch vụ
  - Tổng tiền thanh toán
  - Khả năng rời đi (`Churn`: Yes/No)

## 5. Các bước thực hiện
### 5.1 Tiền xử lý dữ liệu
- Đọc dữ liệu từ file CSV và lưu vào MongoDB.
- Chuyển đổi các giá trị nhị phân từ `Yes/No` thành `0/1` để phù hợp với mô hình.
- Xử lý dữ liệu bị thiếu và chuẩn hóa dữ liệu.

### 5.2 Xây dựng mô hình
- Trích lọc đặc trưng quan trọng.
- Thử nghiệm nhiều mô hình Machine Learning khác nhau.
- Tối ưu hóa mô hình bằng Grid Search hoặc Random Search.

### 5.3 Dự đoán và đánh giá
- Chia tập dữ liệu thành tập huấn luyện và kiểm tra.
- Đánh giá mô hình bằng các chỉ số như Accuracy, Precision, Recall, F1-score.
- Triển khai mô hình trên tập dữ liệu mới.

### 5.4 Xây dựng giao diện
- Hiển thị danh sách khách hàng với thông tin chi tiết.
- Chức năng tìm kiếm theo ID khách hàng, phương thức thanh toán, loại dịch vụ.
- Hiển thị biểu đồ thống kê dựa trên dữ liệu khách hàng.
- Dự đoán khả năng rời đi của khách hàng.

## 6. Kết quả mong đợi
- Hệ thống có khả năng dự đoán chính xác khả năng khách hàng rời đi.
- Cung cấp giao diện trực quan và dễ sử dụng để tra cứu và thống kê dữ liệu.
- Tích hợp pipeline tự động để huấn luyện và đánh giá mô hình.

## 7. Hướng phát triển
- Nâng cấp mô hình với dữ liệu thời gian thực.
- Tích hợp thêm các kỹ thuật Deep Learning.
- Mở rộng phạm vi dự đoán cho nhiều loại dịch vụ khác nhau.

## 8. Hướng dẫn chạy dự án
### 8.1 Kích hoạt môi trường ảoảo
```bash
venv/Scripts/activate
```
### 8.2 Chạy hệ thống
Sử dụng Streamlit:
```bash
streamlit run app.py
```

## 9. Tác giả
Repo thuộc sở hữu: [Nguyễn Minh Ý]
Thành viên đóng góp: [Huỳnh Lý Tân Khoa, Nguyễn Thị Phương Anh, Võ Thị Như Ý]
Email: [nguyenminhy7714@gmail.com] 