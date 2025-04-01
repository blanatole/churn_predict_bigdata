import streamlit as st
from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.linalg import Vectors
from pymongo import MongoClient
import os
import sys
import pandas as pd
import plotly.express as px
import json
from collections import defaultdict

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

@st.cache_resource
def load_rf_model(_spark, model_path="rf_churn_model"):
    try:
        rf_model = RandomForestClassificationModel.load(model_path)
        return rf_model
    except Exception as e:
        st.error(f"Lỗi khi tải mô hình: {str(e)}")
        st.stop()

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

# Hàm dự đoán churn
def predict_churn(payment_method, internet_service):
    payment_method_map = {"Credit card (automatic)": 0, "Electronic check": 1, "Mailed check": 2, "Bank transfer (automatic)": 3}
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
        {"$project": {
            "_id": 0,
            "customerID": 1,
            "PaymentMethod": "$contract.PaymentMethod",
            "InternetService": "$service.InternetService",
            "Churn": "$history.Churn"
        }},
        {"$skip": skip},
        {"$limit": limit}
    ]
    try:
        return list(db.customer_history.aggregate(pipeline))
    except Exception as e:
        st.error(f"Lỗi khi lấy dữ liệu: {str(e)}")
        return []

# Hàm filter_customers
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

# Hàm statistics
def statistics(filtered_customers):
    if not filtered_customers:
        return [], []
    
    stats_dict = {}
    total_count = len(filtered_customers)
    for customer in filtered_customers:
        key = (
            customer["contract"]["PaymentMethod"],
            customer["service"]["InternetService"],
            customer["history"]["Churn"]
        )
        stats_dict[key] = stats_dict.get(key, 0) + 1

    stats_for_table = [
        {
            "PaymentMethod": k[0],
            "InternetService": k[1],
            "Churn": k[2],
            "Count": v
        }
        for k, v in stats_dict.items()
    ]
    
    stats_for_chart = [
        {
            "Combination": f"{k[0]} | {k[1]} | {k[2]}",
            "Count": v,
            "Percentage": (v / total_count) * 100 if total_count > 0 else 0
        }
        for k, v in stats_dict.items()
    ]
    return stats_for_table, stats_for_chart

# Hàm hiển thị theo Churn
def display_by_churn(all_customers):
    if not all_customers:
        return {}
    
    churn_dict = defaultdict(list)
    for customer in all_customers:
        churn = customer["Churn"]
        churn_dict[churn].append({
            "CustomerID": customer["customerID"],
            "InternetService": customer["InternetService"],
            "PaymentMethod": customer["PaymentMethod"]
        })
    return dict(churn_dict)

# Giao diện Streamlit
st.title("Ứng dụng Quản lý Khách hàng và Dự đoán Churn")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Hiển thị đầy đủ", "Dự đoán", "Tìm kiếm", "Thống kê", "Hiển thị theo Churn"])

# Tab 1: Hiển thị đầy đủ (4 feature)
with tab1:
    st.header("Hiển thị đầy đủ khách hàng")
    page_size = 100
    page = st.number_input("Trang", min_value=1, value=1, step=1)
    skip = (page - 1) * page_size
    all_customers = get_all_customers(limit=page_size, skip=skip)
    if all_customers:
        df_all = pd.DataFrame(all_customers)
        st.dataframe(df_all)
        json_data = df_all.to_json(orient="records", force_ascii=False)
        # st.download_button(
        #     label="Tải xuống JSON",
        #     data=json_data,
        #     file_name=f"customers_page_{page}.json",
        #     mime="application/json"
        # )
    else:
        st.warning("Không có dữ liệu khách hàng.")

# Tab 2: Dự đoán Churn
with tab2:
    st.header("Dự đoán khả năng rời bỏ dịch vụ")
    customer_id = st.text_input("CustomerID", key="predict_id")
    internet_service = st.selectbox("InternetService", ["DSL", "Fiber optic", "No"], key="predict_internet")
    payment_method = st.selectbox("PaymentMethod", ["Credit card (automatic)", "Electronic check", "Mailed check", "Bank transfer (automatic)"], key="predict_payment")

    if st.button("Dự đoán"):
        if db.prediction_results.find_one({"customerID": customer_id}):
            st.error(f"CustomerID {customer_id} đã tồn tại trong kết quả dự đoán.")
        else:
            churn_result = predict_churn(payment_method, internet_service)
            if churn_result is not None:
                st.success(f"Kết quả dự đoán: Khách hàng {'có' if churn_result == 'Yes' else 'không'} khả năng rời bỏ dịch vụ.")
                
                prediction_data = {
                    "customerID": customer_id,
                    "InternetService": internet_service,
                    "PaymentMethod": payment_method,
                    "ChurnPrediction": churn_result
                }
                
                services = {"customerID": customer_id, "InternetService": internet_service}
                contract_payment = {"customerID": customer_id, "PaymentMethod": payment_method}
                customer_history = {"customerID": customer_id, "Churn": churn_result}
                db.services.insert_one(services)
                db.contract_payment.insert_one(contract_payment)
                db.customer_history.insert_one(customer_history)
                db.prediction_results.insert_one({"customerID": customer_id, "prediction": churn_result})

                json_data = json.dumps(prediction_data, ensure_ascii=False, indent=4)
                st.download_button(
                    label="Tải xuống JSON (Kết quả dự đoán)",
                    data=json_data,
                    file_name=f"prediction_{customer_id}.json",
                    mime="application/json"
                )

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
            flat_data.pop("customerID", None)
            df = pd.DataFrame([flat_data])
            st.dataframe(df)
            json_data = df.to_json(orient="records", force_ascii=False)
            st.download_button(
                label="Tải xuống JSON",
                data=json_data,
                file_name=f"customer_{customer_id_search}.json",
                mime="application/json"
            )
        else:
            st.error(f"Không tìm thấy khách hàng với CustomerID: {customer_id_search}")

# Tab 4: Thống kê (sử dụng filter_customers và statistics)
with tab4:
    st.header("Thống kê khách hàng")
    # Gọi filter_customers với tất cả tham số là None để lấy toàn bộ dữ liệu
    all_customers = filter_customers(payment_method=None, internet_service=None, churn=None)
    if all_customers:
        stats_for_table, stats_for_chart = statistics(all_customers)
        if stats_for_table:
            df_stats_table = pd.DataFrame(stats_for_table)
            st.dataframe(df_stats_table.T)
            json_data_stats = df_stats_table.to_json(orient="records", force_ascii=False)
            st.download_button(
                label="Tải xuống JSON (Thống kê)",
                data=json_data_stats,
                file_name="statistics.json",
                mime="application/json"
            )

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
            st.warning("Không có dữ liệu thống kê.")
    else:
        st.warning("Không có dữ liệu khách hàng.")

# Tab 5: Hiển thị theo Churn
with tab5:
    st.header("Hiển thị theo Churn")
    all_customers = get_all_customers()  # Lấy tất cả khách hàng
    if all_customers:
        churn_data = display_by_churn(all_customers)
        for churn, customers in churn_data.items():
            st.subheader(f"Churn: {churn}")
            df_churn = pd.DataFrame(customers)
            st.dataframe(df_churn)
            json_data = df_churn.to_json(orient="records", force_ascii=False)
            st.download_button(
                label=f"Tải xuống JSON (Churn {churn})",
                data=json_data,
                file_name=f"churn_{churn}.json",
                mime="application/json"
            )
    else:
        st.warning("Không có dữ liệu khách hàng.")