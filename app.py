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

# st.write("SparkSession đã được khởi tạo thành công!")
# st.write("Mô hình Random Forest đã được tải thành công!")
# st.write(f"Python version in driver: {sys.version}")

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

@st.cache_data
def get_all_customers(limit=100, skip=0):
    pipeline = [
        {"$lookup": {"from": "personal_info", "localField": "customerID", "foreignField": "customerID", "as": "personal"}},
        {"$lookup": {"from": "services", "localField": "customerID", "foreignField": "customerID", "as": "service"}},
        {"$lookup": {"from": "contract_payment", "localField": "customerID", "foreignField": "customerID", "as": "contract"}},
        {"$lookup": {"from": "customer_history", "localField": "customerID", "foreignField": "customerID", "as": "history"}},
        {"$unwind": {"path": "$personal", "preserveNullAndEmptyArrays": True}},
        {"$unwind": {"path": "$service", "preserveNullAndEmptyArrays": True}},
        {"$unwind": {"path": "$contract", "preserveNullAndEmptyArrays": True}},
        {"$unwind": {"path": "$history", "preserveNullAndEmptyArrays": True}},
        {"$project": {"_id": 0, "personal._id": 0, "service._id": 0, "contract._id": 0, "history._id": 0}},
        {"$skip": skip},
        {"$limit": limit}
    ]
    try:
        return list(db.customer_history.aggregate(pipeline))
    except Exception as e:
        st.error(f"Lỗi khi lấy dữ liệu: {str(e)}")
        return []

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
def statistics(filtered_customers):
    if not filtered_customers:
        return []
    
    # Tạo danh sách tổ hợp từ dữ liệu đã lọc
    stats_data = []
    for customer in filtered_customers:
        stats_data.append({
            "PaymentMethod": customer["contract"]["PaymentMethod"],
            "InternetService": customer["service"]["InternetService"],
            "Churn": customer["history"]["Churn"]
        })

    # Nhóm và đếm số lượng cho từng tổ hợp
    stats_dict = {}
    total_count = len(filtered_customers)
    for item in stats_data:
        key = (item["PaymentMethod"], item["InternetService"], item["Churn"])
        stats_dict[key] = stats_dict.get(key, 0) + 1

    # Tạo danh sách thống kê cho bảng
    stats_for_table = [
        {
            "PaymentMethod": k[0],
            "InternetService": k[1],
            "Churn": k[2],
            "Count": v
        }
        for k, v in stats_dict.items()
    ]
    
    # Tạo danh sách thống kê cho biểu đồ (với Combination)
    stats_for_chart = [
        {
            "Combination": f"{k[0]} | {k[1]} | {k[2]}",
            "Count": v,
            "Percentage": (v / total_count) * 100 if total_count > 0 else 0
        }
        for k, v in stats_dict.items()
    ]
    return stats_for_table, stats_for_chart

# Giao diện Streamlit
st.title("Ứng dụng Quản lý Khách hàng và Dự đoán Churn")

tab1, tab2, tab3, tab4 = st.tabs(["Danh sách khách hàng","Dự đoán", "Tìm kiếm", "Thống kê"])

# tab 1: Danh sách khách hàng
with tab1:
    st.header("Danh sách khách hàng")
    page_size = 100
    page = st.number_input("Trang", min_value=1, value=1, step=1)
    skip = (page - 1) * page_size
    all_customers = get_all_customers(limit=page_size, skip=skip)
    if all_customers:
        flat_data = []
        for customer in all_customers:
            flat_customer = {}
            for key, value in customer.items():
                if isinstance(value, dict):
                    flat_customer.update(value)
                else:
                    flat_customer[key] = value
            flat_data.append(flat_customer)
        df_all = pd.DataFrame(flat_data)
        st.write(f"Danh sách khách hàng (Trang {page}):")
        customer_table = st.dataframe(df_all)
    else:
        st.warning("Không có dữ liệu khách hàng.")

# Tab 2: Dự đoán Churn
with tab2:
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
        if db.prediction_results.find_one({"customerID": customer_id}):
            st.error(f"CustomerID {customer_id} đã tồn tại trong kết quả dự đoán.")
        else:
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

# Tab 3: Tìm kiếm theo CustomerID
with tab3:
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

# Tab 4: Lọc khách hàng
with tab4:
    st.header("Lọc và Thống kê khách hàng")
    
    # Phần lọc
    payment_method_filter = st.selectbox(
        "Phương thức thanh toán", 
        ["", "Credit card (automatic)", "Electronic check", "Mailed check", "Bank transfer (automatic)"], 
        index=0, 
        key="filter_pm"
    )
    internet_service_filter = st.selectbox(
        "Loại dịch vụ Internet", 
        ["", "DSL", "Fiber optic", "No"], 
        index=0, 
        key="filter_is"
    )
    churn_filter = st.selectbox(
        "Trạng thái Churn", 
        ["", "Yes", "No"], 
        index=0, 
        key="filter_churn"
    )
    
    if st.button("Lọc và Thống kê", key="filter_stat_btn"):
        # Lọc dữ liệu
        filtered_customers = filter_customers(
            payment_method_filter if payment_method_filter else None,
            internet_service_filter if internet_service_filter else None,
            churn_filter if churn_filter else None
        )
        
        if filtered_customers:
            # Hiển thị kết quả lọc (không thay đổi)
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

            # Thống kê và hiển thị
            stats_for_table, stats_for_chart = statistics(filtered_customers)
            if stats_for_table:
                # Hiển thị bảng thống kê với PaymentMethod, InternetService, Churn, Count
                df_stats_table = pd.DataFrame(stats_for_table)
                st.write("Bảng thống kê:")
                st.dataframe(df_stats_table.T)  # Hiển thị bảng ngang

                # Vẽ biểu đồ tròn theo tổ hợp feature
                df_stats_chart = pd.DataFrame(stats_for_chart)
                fig = px.pie(
                    df_stats_chart, 
                    values="Count", 
                    names="Combination", 
                    title="Tỷ lệ các tổ hợp feature (PaymentMethod | InternetService | Churn)",
                    labels={"Combination": "Tổ hợp", "Count": "Số lượng"},
                    hover_data=["Percentage"],
                    hole=0.3
                )
                fig.update_traces(textposition="inside", textinfo="percent")
                st.plotly_chart(fig)
            else:
                st.warning("Không có dữ liệu thống kê từ kết quả lọc.")
        else:
            st.warning("Không tìm thấy khách hàng nào thỏa mãn điều kiện.")

