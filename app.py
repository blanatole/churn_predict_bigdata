import streamlit as st
from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.linalg import Vectors
from pymongo import MongoClient
import os
import sys

# Đặt biến môi trường cho PySpark sử dụng Python từ môi trường ảo
python_path = os.path.join(os.getcwd(), "venv", "Scripts", "python.exe")
os.environ["PYSPARK_PYTHON"] = python_path
os.environ["PYSPARK_DRIVER_PYTHON"] = python_path

# Hàm khởi tạo SparkSession (chỉ chạy một lần)
@st.cache_resource
def init_spark():
    try:
        spark = SparkSession.builder \
            .appName("Churn Prediction Streamlit") \
            .config("spark.driver.memory", "4g") \
            .config("spark.executor.memory", "4g") \
            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
            .config("spark.python.worker.memory", "1g") \
            .master("local[1]") \
            .getOrCreate()
        return spark
    except Exception as e:
        st.error(f"Lỗi khi khởi tạo SparkSession: {str(e)}")
        st.stop()

# Hàm tải mô hình Random Forest (chỉ chạy một lần)
@st.cache_resource
def load_rf_model(_spark, model_path="rf_churn_model"):
    try:
        rf_model = RandomForestClassificationModel.load(model_path)
        return rf_model
    except Exception as e:
        st.error(f"Lỗi khi tải mô hình: {str(e)}")
        st.stop()

# Hàm kết nối MongoDB (chỉ chạy một lần)
@st.cache_resource
def init_mongo():
    client = MongoClient("mongodb+srv://nguyenminhy7714:minhy112@cluster0.xxkrzas.mongodb.net/")
    db = client["telco_churn"]  # Thay "database_name" bằng tên database
    # Tạo chỉ mục để tối ưu truy vấn
    db.churn_info.create_index("customerID")
    db.contract_info.create_index("PaymentMethod")
    db.internet_service.create_index("InternetService")
    return db

# Khởi tạo các tài nguyên
spark = init_spark()
rf_model = load_rf_model(spark)
db = init_mongo()

# Hiển thị thông tin khởi tạo
st.write("SparkSession đã được khởi tạo thành công!")
st.write("Mô hình Random Forest đã được tải thành công!")
st.write(f"Python version in driver: {sys.version}")

# Hàm dự đoán churn
def predict_churn(payment_method, internet_service):
    payment_method_map = {"Credit card": 0, "Electronic check": 1, "Mailed check": 2, "Bank transfer": 3}
    internet_service_map = {"DSL": 0, "Fiber optic": 1, "No": 2}

    pm_idx = payment_method_map.get(payment_method, 0)
    is_idx = internet_service_map.get(internet_service, 0)

    pm_vec = [0.0] * 4
    is_vec = [0.0] * 3
    pm_vec[pm_idx] = 1.0
    is_vec[is_idx] = 1.0

    features = pm_vec + is_vec
    feature_vector = Vectors.dense(features)

    input_df = spark.createDataFrame([(feature_vector,)], ["features"])
    prediction = rf_model.transform(input_df)

    try:
        result = prediction.select("prediction").first()[0]
        return "Yes" if result == 1.0 else "No"
    except Exception as e:
        st.error(f"Lỗi khi dự đoán: {str(e)}")
        return None

# Hàm tìm kiếm theo CustomerID
def search_by_customer_id(customer_id):
    pipeline = [
        {"$match": {"customerID": customer_id}},
        {"$lookup": {"from": "contract_info", "localField": "customerID", "foreignField": "customerID", "as": "contract"}},
        {"$lookup": {"from": "internet_service", "localField": "customerID", "foreignField": "customerID", "as": "internet"}},
        {"$unwind": "$contract"},
        {"$unwind": "$internet"},
        {"$project": {"_id": 0, "contract._id": 0, "internet._id": 0}}
    ]
    result = list(db.churn_info.aggregate(pipeline))
    return result[0] if result else None

# Hàm lọc theo PaymentMethod, InternetService, Churn
def filter_customers(payment_method=None, internet_service=None, churn=None):
    match_conditions = {}
    if payment_method:
        match_conditions["contract.PaymentMethod"] = payment_method
    if internet_service:
        match_conditions["internet.InternetService"] = internet_service
    if churn:
        match_conditions["Churn"] = churn

    pipeline = [
        {"$lookup": {"from": "contract_info", "localField": "customerID", "foreignField": "customerID", "as": "contract"}},
        {"$lookup": {"from": "internet_service", "localField": "customerID", "foreignField": "customerID", "as": "internet"}},
        {"$unwind": "$contract"},
        {"$unwind": "$internet"},
        {"$project": {"_id": 0, "contract._id": 0, "internet._id": 0}}
    ]
    if match_conditions:
        pipeline.append({"$match": match_conditions})

    return list(db.churn_info.aggregate(pipeline))

# Giao diện Streamlit
st.title("Ứng dụng Quản lý Khách hàng và Dự đoán Churn")

# Tab để chọn chức năng
tab1, tab2, tab3 = st.tabs(["Dự đoán Churn", "Tìm kiếm theo CustomerID", "Lọc khách hàng"])

# Tab 1: Dự đoán Churn
with tab1:
    st.header("Dự đoán khả năng rời bỏ dịch vụ")
    payment_method = st.selectbox("Phương thức thanh toán", ["Credit card", "Electronic check", "Mailed check", "Bank transfer"], key="predict_pm")
    internet_service = st.selectbox("Loại dịch vụ Internet", ["DSL", "Fiber optic", "No"], key="predict_is")
    if st.button("Dự đoán"):
        churn_result = predict_churn(payment_method, internet_service)
        if churn_result is not None:
            st.success(f"Kết quả dự đoán: Khách hàng {'có' if churn_result == 'Yes' else 'không'} khả năng rời bỏ dịch vụ.")

# Tab 2: Tìm kiếm theo CustomerID
with tab2:
    st.header("Tìm kiếm theo CustomerID")
    customer_id = st.text_input("Nhập CustomerID:")
    if st.button("Tìm kiếm"):
        customer_info = search_by_customer_id(customer_id)
        if customer_info:
            st.write("**Thông tin khách hàng:**")
            st.write(f"**CustomerID:** {customer_info['customerID']}")
            st.write(f"**Churn:** {customer_info['Churn']}")
            st.write("**Contract Info:**")
            for key, value in customer_info['contract'].items():
                st.write(f"{key}: {value}")
            st.write("**Internet Service Info:**")
            for key, value in customer_info['internet'].items():
                st.write(f"{key}: {value}")
        else:
            st.error(f"Không tìm thấy khách hàng với CustomerID: {customer_id}")

# Tab 3: Lọc khách hàng
with tab3:
    st.header("Lọc khách hàng")
    payment_method_filter = st.selectbox("Phương thức thanh toán", ["", "Credit card", "Electronic check", "Mailed check", "Bank transfer"], index=0, key="filter_pm")
    internet_service_filter = st.selectbox("Loại dịch vụ Internet", ["", "DSL", "Fiber optic", "No"], index=0, key="filter_is")
    churn_filter = st.selectbox("Trạng thái Churn", ["", "Yes", "No"], index=0, key="filter_churn")
    if st.button("Lọc"):
        filtered_customers = filter_customers(
            payment_method_filter if payment_method_filter else None,
            internet_service_filter if internet_service_filter else None,
            churn_filter if churn_filter else None
        )
        if filtered_customers:
            st.write(f"**Tìm thấy {len(filtered_customers)} khách hàng thỏa mãn điều kiện:**")
            for i, customer in enumerate(filtered_customers, 1):
                st.write(f"### Khách hàng {i}:")
                st.write(f"**CustomerID:** {customer['customerID']}")
                st.write(f"**Churn:** {customer['Churn']}")
                st.write("**Contract Info:**")
                for key, value in customer['contract'].items():
                    st.write(f"{key}: {value}")
                st.write("**Internet Service Info:**")
                for key, value in customer['internet'].items():
                    st.write(f"{key}: {value}")
                st.write("---")
        else:
            st.warning("Không tìm thấy khách hàng nào thỏa mãn điều kiện.")