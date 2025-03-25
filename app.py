import streamlit as st
from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.linalg import Vectors
from pymongo import MongoClient
import os
import sys
import pandas as pd
import plotly.express as px

# Đặt biến môi trường cho PySpark sử dụng Python từ môi trường ảo
python_path = os.path.join(os.getcwd(), "venv", "Scripts", "python.exe")
os.environ["PYSPARK_PYTHON"] = python_path
os.environ["PYSPARK_DRIVER_PYTHON"] = python_path

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

# Hàm tải mô hình Random Forest
@st.cache_resource
def load_rf_model(_spark, model_path="rf_churn_model"):
    try:
        rf_model = RandomForestClassificationModel.load(model_path)
        return rf_model
    except Exception as e:
        st.error(f"Lỗi khi tải mô hình: {str(e)}")
        st.stop()

# Kết nối MongoDB
@st.cache_resource
def init_mongo():
    client = MongoClient("mongodb+srv://nguyenminhy7714:minhy112@cluster0.xxkrzas.mongodb.net/")
    db = client["telco_churn"]
    db.personal_info.create_index("customerID")
    db.services.create_index("customerID")
    db.contract_payment.create_index("customerID")
    db.customer_history.create_index("customerID")
    return db

# Khởi tạo
spark = init_spark()
rf_model = load_rf_model(spark)
db = init_mongo()

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
        {"$lookup": {"from": "personal_info", "localField": "customerID", "foreignField": "customerID", "as": "personal"}},
        {"$lookup": {"from": "services", "localField": "customerID", "foreignField": "customerID", "as": "service"}},
        {"$lookup": {"from": "contract_payment", "localField": "customerID", "foreignField": "customerID", "as": "contract"}},
        {"$lookup": {"from": "customer_history", "localField": "customerID", "foreignField": "customerID", "as": "history"}},
        {"$unwind": {"path": "$personal", "preserveNullAndEmptyArrays": True}},
        {"$unwind": {"path": "$service", "preserveNullAndEmptyArrays": True}},
        {"$unwind": {"path": "$contract", "preserveNullAndEmptyArrays": True}},
        {"$unwind": {"path": "$history", "preserveNullAndEmptyArrays": True}},
        {"$project": {"_id": 0, "personal._id": 0, "service._id": 0, "contract._id": 0, "history._id": 0}}
    ]
    result = list(db.customer_history.aggregate(pipeline))
    return result[0] if result else None

# Hàm lọc khách hàng
def filter_customers(payment_method=None, internet_service=None, churn=None):
    match_conditions = {}
    if payment_method:
        match_conditions["contract.PaymentMethod"] = payment_method
    if internet_service:
        match_conditions["service.InternetService"] = internet_service
    if churn:
        match_conditions["history.Churn"] = churn

    pipeline = [
        {"$lookup": {"from": "personal_info", "localField": "customerID", "foreignField": "customerID", "as": "personal"}},
        {"$lookup": {"from": "services", "localField": "customerID", "foreignField": "customerID", "as": "service"}},
        {"$lookup": {"from": "contract_payment", "localField": "customerID", "foreignField": "customerID", "as": "contract"}},
        {"$lookup": {"from": "customer_history", "localField": "customerID", "foreignField": "customerID", "as": "history"}},
        {"$unwind": {"path": "$personal", "preserveNullAndEmptyArrays": True}},
        {"$unwind": {"path": "$service", "preserveNullAndEmptyArrays": True}},
        {"$unwind": {"path": "$contract", "preserveNullAndEmptyArrays": True}},
        {"$unwind": {"path": "$history", "preserveNullAndEmptyArrays": True}},
        {"$project": {"_id": 0, "personal._id": 0, "service._id": 0, "contract._id": 0, "history._id": 0}}
    ]
    if match_conditions:
        pipeline.append({"$match": match_conditions})

    return list(db.customer_history.aggregate(pipeline))

# Hàm thống kê
def statistics(payment_method=None, internet_service=None):
    match_conditions = {}
    if payment_method:
        match_conditions["contract.PaymentMethod"] = payment_method
    if internet_service:
        match_conditions["service.InternetService"] = internet_service

    pipeline = [
        {"$lookup": {"from": "personal_info", "localField": "customerID", "foreignField": "customerID", "as": "personal"}},
        {"$lookup": {"from": "services", "localField": "customerID", "foreignField": "customerID", "as": "service"}},
        {"$lookup": {"from": "contract_payment", "localField": "customerID", "foreignField": "customerID", "as": "contract"}},
        {"$lookup": {"from": "customer_history", "localField": "customerID", "foreignField": "customerID", "as": "history"}},
        {"$unwind": "$personal"},
        {"$unwind": "$service"},
        {"$unwind": "$contract"},
        {"$unwind": "$history"}
    ]
    if match_conditions:
        pipeline.append({"$match": match_conditions})

    pipeline.extend([
        {"$group": {
            "_id": {
                "PaymentMethod": "$contract.PaymentMethod",
                "InternetService": "$service.InternetService",
                "Churn": "$history.Churn"
            },
            "count": {"$sum": 1}
        }},
        {"$sort": {"count": -1}}
    ])

    stats = list(db.customer_history.aggregate(pipeline))
    return stats

# Giao diện Streamlit
st.title("Ứng dụng Quản lý Khách hàng và Dự đoán Churn")

tab1, tab2, tab3, tab4 = st.tabs(["Dự đoán Churn", "Tìm kiếm theo CustomerID", "Lọc khách hàng", "Thống kê"])

# Tab 1: Dự đoán Churn
with tab1:
    st.header("Dự đoán khả năng rời bỏ dịch vụ")
    customer_id = st.text_input("CustomerID", key="predict_id")
    gender = st.selectbox("Gender", ["Male", "Female"], key="predict_gender")
    senior_citizen = st.selectbox("SeniorCitizen", [0, 1], key="predict_senior")
    partner = st.selectbox("Partner", ["Yes", "No"], key="predict_partner")
    dependents = st.selectbox("Dependents", ["Yes", "No"], key="predict_dependents")
    phone_service = st.selectbox("PhoneService", ["Yes", "No"], key="predict_phone")
    multiple_lines = st.selectbox("MultipleLines", ["Yes", "No", "No phone service"], key="predict_lines")
    internet_service = st.selectbox("InternetService", ["DSL", "Fiber optic", "No"], key="predict_internet")
    online_security = st.selectbox("OnlineSecurity", ["Yes", "No", "No internet service"], key="predict_security")
    online_backup = st.selectbox("OnlineBackup", ["Yes", "No", "No internet service"], key="predict_backup")
    device_protection = st.selectbox("DeviceProtection", ["Yes", "No", "No internet service"], key="predict_protection")
    tech_support = st.selectbox("TechSupport", ["Yes", "No", "No internet service"], key="predict_support")
    streaming_tv = st.selectbox("StreamingTV", ["Yes", "No", "No internet service"], key="predict_tv")
    streaming_movies = st.selectbox("StreamingMovies", ["Yes", "No", "No internet service"], key="predict_movies")
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], key="predict_contract")
    paperless_billing = st.selectbox("PaperlessBilling", ["Yes", "No"], key="predict_paperless")
    payment_method = st.selectbox("PaymentMethod", ["Credit card (automatic)", "Electronic check", "Mailed check", "Bank transfer (automatic)"], key="predict_payment")
    monthly_charges = st.number_input("MonthlyCharges", min_value=0.0, key="predict_monthly")
    total_charges = st.number_input("TotalCharges", min_value=0.0, key="predict_total")
    tenure = st.number_input("Tenure", min_value=0, key="predict_tenure")

    if st.button("Dự đoán"):
        churn_result = predict_churn(payment_method, internet_service)
        if churn_result is not None:
            st.success(f"Kết quả dự đoán: Khách hàng {'có' if churn_result == 'Yes' else 'không'} khả năng rời bỏ dịch vụ.")
            # Lưu thông tin vào MongoDB
            personal_info = {"customerID": customer_id, "gender": gender, "SeniorCitizen": senior_citizen, "Partner": partner, "Dependents": dependents}
            services = {"customerID": customer_id, "PhoneService": phone_service, "MultipleLines": multiple_lines, "InternetService": internet_service, 
                        "OnlineSecurity": online_security, "OnlineBackup": online_backup, "DeviceProtection": device_protection, 
                        "TechSupport": tech_support, "StreamingTV": streaming_tv, "StreamingMovies": streaming_movies}
            contract_payment = {"customerID": customer_id, "Contract": contract, "PaperlessBilling": paperless_billing, "PaymentMethod": payment_method, 
                                "MonthlyCharges": monthly_charges, "TotalCharges": total_charges}
            customer_history = {"customerID": customer_id, "tenure": tenure, "Churn": churn_result}
            db.personal_info.insert_one(personal_info)
            db.services.insert_one(services)
            db.contract_payment.insert_one(contract_payment)
            db.customer_history.insert_one(customer_history)
            # Lưu kết quả dự đoán vào collection riêng
            db.prediction_results.insert_one({"customerID": customer_id, "prediction": churn_result})
            st.write("Thông tin đã được lưu vào MongoDB.")

# Tab 2: Tìm kiếm theo CustomerID
with tab2:
    st.header("Tìm kiếm theo CustomerID")
    customer_id_search = st.text_input("Nhập CustomerID:", key="search_id")
    if st.button("Tìm kiếm", key="search_btn"):
        customer_info = search_by_customer_id(customer_id_search)
        if customer_info:
            flat_data = {}
            for key, value in customer_info.items():
                if isinstance(value, dict):
                    flat_data.update(value)
                else:
                    flat_data[key] = value
            flat_data.pop("customerID", None)  # Loại bỏ customerID trùng
            df = pd.DataFrame([flat_data])
            st.write(f"Thông tin khách hàng {customer_id_search}:")
            st.dataframe(df)
        else:
            st.error(f"Không tìm thấy khách hàng với CustomerID: {customer_id_search}")

# Tab 3: Lọc khách hàng
with tab3:
    st.header("Lọc khách hàng")
    payment_method_filter = st.selectbox("Phương thức thanh toán", ["", "Credit card (automatic)", "Electronic check", "Mailed check", "Bank transfer (automatic)"], index=0, key="filter_pm")
    internet_service_filter = st.selectbox("Loại dịch vụ Internet", ["", "DSL", "Fiber optic", "No"], index=0, key="filter_is")
    churn_filter = st.selectbox("Trạng thái Churn", ["", "Yes", "No"], index=0, key="filter_churn")
    if st.button("Lọc", key="filter_btn"):
        filtered_customers = filter_customers(payment_method_filter if payment_method_filter else None,
                                             internet_service_filter if internet_service_filter else None,
                                             churn_filter if churn_filter else None)
        if filtered_customers:
            flat_data = []
            for customer in filtered_customers:
                flat_customer = {}
                for key, value in customer.items():
                    if isinstance(value, dict):
                        flat_customer.update(value)
                    else:
                        flat_customer[key] = value
                flat_data.append(flat_customer)
            df = pd.DataFrame(flat_data)
            st.write(f"Tìm thấy {len(filtered_customers)} khách hàng:")
            st.dataframe(df)
        else:
            st.warning("Không tìm thấy khách hàng nào thỏa mãn điều kiện.")

# Tab 4: Thống kê
with tab4:
    st.header("Thống kê khách hàng")
    payment_method_stat = st.selectbox("Lọc theo PaymentMethod", ["", "Credit card (automatic)", "Electronic check", "Mailed check", "Bank transfer (automatic)"], index=0, key="stat_pm")
    internet_service_stat = st.selectbox("Lọc theo InternetService", ["", "DSL", "Fiber optic", "No"], index=0, key="stat_is")
    if st.button("Thống kê", key="stat_btn"):
        stats = statistics(payment_method_stat if payment_method_stat else None,
                          internet_service_stat if internet_service_stat else None)
        if stats:
            df_stats = pd.DataFrame([{"PaymentMethod": stat["_id"]["PaymentMethod"], 
                                     "InternetService": stat["_id"]["InternetService"], 
                                     "Churn": stat["_id"]["Churn"], 
                                     "Count": stat["count"]} for stat in stats])
            st.write("Bảng thống kê:")
            st.dataframe(df_stats.T)  # Hiển thị bảng ngang
            # Vẽ biểu đồ tỷ lệ Yes/No
            churn_counts = df_stats.groupby("Churn")["Count"].sum().reset_index()
            fig = px.pie(churn_counts, values="Count", names="Churn", title="Tỷ lệ Churn (Yes/No)")
            st.plotly_chart(fig)
        else:
            st.warning("Không có dữ liệu thống kê.")