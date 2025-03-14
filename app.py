import streamlit as st
import joblib
import numpy as np

# Tải mô hình scikit-learn
try:
    model = joblib.load("rf_model.pkl")
    st.write("Mô hình Random Forest đã được tải thành công!")
except Exception as e:
    st.error(f"Lỗi khi tải mô hình: {str(e)}")
    st.stop()

# Hàm dự đoán
def predict_churn(payment_method, internet_service):
    # Ánh xạ thủ công - điều chỉnh để tạo 5 đặc trưng
    payment_method_map = {"Credit card": 0, "Electronic check": 1, "Mailed check": 2, "Bank transfer": 3}
    internet_service_map = {"DSL": 0, "Fiber optic": 1, "No": 2}

    pm_idx = payment_method_map.get(payment_method, 0)
    is_idx = internet_service_map.get(internet_service, 0)

    # Giả sử mô hình được huấn luyện với 5 đặc trưng
    # Ví dụ: Chỉ dùng PaymentMethod (4 chiều) và InternetService (1 chiều, không One-Hot)
    features = [0.0] * 5  # Khởi tạo vector 5 chiều
    features[pm_idx] = 1.0  # One-Hot cho PaymentMethod (4 giá trị)
    features[4] = is_idx  # Giá trị số cho InternetService (không One-Hot)

    # Chuyển thành mảng NumPy
    features = np.array(features).reshape(1, -1)
    result = model.predict(features)[0]
    return "Yes" if result == 1 else "No"

# Giao diện Streamlit
st.title("Dự đoán khả năng rời bỏ dịch vụ (Churn Prediction)")

payment_method = st.selectbox("Phương thức thanh toán", ["Credit card", "Electronic check", "Mailed check", "Bank transfer"])
internet_service = st.selectbox("Loại dịch vụ Internet", ["DSL", "Fiber optic", "No"])

if st.button("Dự đoán"):
    churn_result = predict_churn(payment_method, internet_service)
    st.success(f"Kết quả dự đoán: Khách hàng {'có' if churn_result == 'Yes' else 'không'} khả năng rời bỏ dịch vụ.")