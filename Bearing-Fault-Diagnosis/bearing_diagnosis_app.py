import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# Set the Page Icon and title
st.set_page_config(
    page_title="Bearing Fault Diagnosis",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="collapsed")


# Load trained model and the encoder
model = joblib.load("random_forest_model.pkl")
encoder = joblib.load("label_encoder.pkl")
scaler = joblib.load("scaler.pkl")




st.markdown("""
            # ⚙️ Bearing Fault Diagnosis Using Machine Learning
            
            
            """)


home, prediction, model_performance, about = st.tabs(
    ["Home", "Prediction", "Model Performance", "About"])

with home:
    st.caption(
        """
        Rolling element bearings are critical components in rotating machinery, and their failure can result in costly equipment downtime, reduced productivity and increased maintenance expenses. Traditional maintenance strategies often detect faults only after significant damage has occurred, highlighting the need for intelligent fault diagnosis systems capable of supporting predictive maintenance.  
        ---
        
      
        """
    )

    st.divider()

    st.markdown("""
                ### Why intelligent monitoring is essential for the reliability of rotating machinery and industrial systems  
                ---
                
                """)
    
    st.info("Disclaimer: The model supports predictive maintenance by providing information about the current bearing condition")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
                    ### ⚙️ Foundation of Rotation  
                    Bearings are the most critical components in motors, pumps, and gearboxes across all industrial sectors.
                    """)
    with col2:
        st.markdown("""
                    ### ⚠️ Failure Consequences 
                    Bearing defects account for a significant percentage of machine failures, leading to safety risks and operational halts.
                    """)
    with col3:
        st.markdown("""
                    ###  🧠 The AI Advantage 
                   Traditional maintenance relies on fixed schedules; machine learning enables condition-based monitoring to predict failure.
                    """)

    with col4:
        st.markdown("""
                    ### ⛓️ Strategic Impact
                   Transitioning to predictive maintenance reduces unnecessary servicing and extends the lifespan of industrial assets.
                    """)

    st.divider()

    st.subheader(
        "The technical and economic challenge of unexpected production downtime in high-stakes environments")
    st.divider()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
                    ### ⚠️ Reactive Maintenance  
                    Running equipment until it fails creates a cycle of emergency repairs and high costs.
                    """)
    with col2:
        st.markdown("""
                    ### ⛓️‍💥 Unexpected Failure 
                    Sudden breakdowns cause immediate stops in production lines, disrupting entire supply chains.
                    """)
    with col3:
        st.markdown("""
                    ###  📉 Financial Impact 
                    Revenue loss is compounded by secondary damage to machinery and high costs for emergency spare parts.
                    """)

    with col4:
        st.markdown("""
                    ### 🛠️ Diagnostic Need
                    Critical requirement for an automated system that can accurately classify fault types from raw sensor data.
                    """)

    st.divider()
    st.metric("Primary Metric", "94.57%", "Accuracy")


with prediction:
    st.title("Performance Metric")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.caption("Model Accuracy")
        st.metric("Accuracy", "94.57%", "reliable")
    with col2:
        st.caption("Model Precision")
        st.metric("Precision", "94.65%", "reliable")
    with col3:
        st.caption("Recall Score")
        st.metric("Recall", "94.35%", "reliable")
    with col4:
        st.caption("F1 Score")
        st.metric("F1", "94.38%", "reliable")
    st.sidebar.title("Make predictions")
    mode = st.sidebar.radio("Choose input mode:",
                            ("Manual Input", "CSV Upload"))

    st.divider()

    st.info("""
                Open the sidebar to make predictions `<<`
                """)

    # Manual input
    if mode == "Manual Input":
        st.sidebar.header("Input Bearing Features")
        max = st.sidebar.number_input("Maximum")
        min = st.sidebar.number_input("Minimum")
        mean = st.sidebar.number_input("Mean")
        std = st.sidebar.number_input("Standard Deviation")
        rms = st.sidebar.number_input("RMS")
        kurtosis = st.sidebar.number_input("Kurtosis")
        skewness = st.sidebar.number_input("Skewness")
        crest = st.sidebar.number_input("Crest Factor")
        form = st.sidebar.number_input("Form Factor")

        input_data = pd.DataFrame({
            "Max": [max],
            "Min": [min],
            "Mean": [mean],
            "Std": [std],
            "RMS": [rms],
            "Kurtosis": [kurtosis],
            "Skewness": [skewness],
            "Crest Factor": [crest],
            "Form Factor": [form]
        })

        # Add prediction button for manual inputation

        if st.sidebar.button("Predict", type="primary"):
            st.write("Prediction")
            prediction = model.predict(input_data)
            fault = encoder.inverse_transform(prediction)
            st.success(f"Prediction: {fault}")
        if st.sidebar.button("Prediction Probability", type="primary"):
            st.write("Probability of each of the classes")
            probability = model.predict_proba(input_data)
            st.dataframe(
                pd.DataFrame(
                    probability,
                    columns=encoder.classes_))

        # Uploading a CSV file
    else:
        try:
            st.sidebar.write(
                "Upload a feature dataset to run batch prediction (CSV)")
            uploaded_file = st.sidebar.file_uploader(
                "Choose a CSV file", type="csv")
            if uploaded_file is not None:
                data = pd.read_csv(uploaded_file)

                # ASk user to preview dataset
                st.subheader("Preview of Uploaded file")
                st.write(data.head())

                # Preprocessing configuration
                st.sidebar.subheader("Preprocessing Options")

                # Check and remove target column if present
                target_col = None
                possible_targets = ["fault", "target",
                                    "label", "Fault", "Target", "Label"]
                for col in possible_targets:
                    if col in data.columns:
                        target_col = col
                        break

                if target_col:
                    drop_target = st.sidebar.checkbox(
                        f"Drop target column ('{target_col}' before prediction)", value=True)
                    if drop_target:
                        features_data = data.drop(columns=[target_col])
                    else:
                        features_data = data.copy()
                else:
                    features_data = data.copy()

                # check for scaling requirement
                is_scaled = st.sidebar.radio(
                    "Is this dataset already scaled?",
                    options=["No (Apply loaded scaler)",
                             "Yes (Data is already scaled)"],
                    index=0
                )

                # Run prediction
                if st.sidebar.button("Run Predictions", type="primary"):

                    # prepare feature data
                    processed_data = features_data.copy()

                    if is_scaled == "No (Apply loaded scaler)":
                        processed_data = scaler.transform(processed_data)

                    # Generate predictions and probabilities
                    raw_preds = model.predict(processed_data)
                    pred_labels = encoder.inverse_transform(raw_preds)
                    pred_probs = model.predict_proba(processed_data)

                    # Output Results
                    st.subheader("Prediction Results")

                    results_data = data.copy()
                    results_data["Predicted_Fault"] = pred_labels

                    # Display the results summary
                    col1, col2 = st.columns(2)
                    with col1:
                        st.caption("Predicted Classes")
                        st.dataframe(results_data)
                    with col2:
                        st.caption("Probability of Classes")
                        prob_data = pd.DataFrame(
                            pred_probs, columns=encoder.classes_)
                        st.dataframe(prob_data)

                    # Downlaod options
                    csv_data = results_data.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Predictions CSV",
                        data=csv_data,
                        file_name="bearing_fault_predictions.csv",
                        mime="text/csv"
                    )

        except Exception as e:
            st.error(f"Error processing file {e}")


with model_performance:
    st.title("Insights from exploratory data analysis (EDA)")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Distribution of Training and Test sets")

        train_set = 1840
        test_set = 460

        pie = go.Figure(
            data=[
                go.Pie(
                    labels=["Training Set", "Test Set"],
                    values=[train_set, test_set],
                    hole=0.65,
                    marker=dict(
                        colors=[
                            "#10b981",
                            "#ef4444"
                        ]
                    ),
                    textinfo="label+percent"
                )
            ]
        )
        pie.update_layout(
            height=420,
            paper_bgcolor="#07111F",
            plot_bgcolor="#07111F",
            font_color="white",
            margin=dict(t=20, b=20)
        )

        st.plotly_chart(
            pie,
            use_container_width=True
        )

    with col2:
        st.caption("Features Mutual Information scores")
        st.image("Random_mutual_information.png",
                 caption="Mutual information of features")

    st.divider()
    st.write("The Random Forest Classifier Achieved the following:")
    performance_table = st.table({
        "Accuracy": "94.57%",
        "Precision": "94.65%",
        "Recall": "94.35%",
        "F1 Score": "94.38%"
    })

    st.divider()


with about:

    st.title("📌 About the Project")

    st.markdown("""
        ### **Overview**
        This intelligent diagnostic system leverages supervised machine learning to perform automated **Bearing Fault Diagnosis** using statistical vibration features. By classifying structural health parameters before catastrophic failure occurs, this tool supports predictive maintenance strategies to reduce industrial equipment downtime and maintenance costs.
        
        The underlying models are trained and validated using standard time-domain and frequency-domain features extracted from the **Case Western Reserve University (CWRU) Bearing Dataset**.
        """)

    # column for dataset set description
    dat1, dat2 = st.columns(2)

    with dat1:
        st.title("Dataset Metadata")
        st.table({
            "Characteristic": ["Dataset Source", "Application",
                               "Data Type",
                               "Number of Samples",
                               "Number of Predictor Variables",
                               "Number of Target Classes",
                               "Classification Type"],
            "Description": ["Case Western Reserve University Bearing Data Center",
                            "Bearing Fault Diagnosis",
                            "Statistical vibration features",
                            "2,300",
                            "9",
                            "10",
                            "Multiclass Classification"]})

    st.divider()

    with dat2:
        st.title("Description of Variables")
        st.table({
            "Variable": ["Max/Min",
                         "Mean",
                         "Standard Deviation",
                         "RMS",
                         "Skewness",
                         "Kurtosis",
                         "Crest Factor",
                         "Form",
                         "Fault"
                         ],
            "Description": ["Peak vibration amplitudes. Faults often cause spikes",
                            "Average value of the vibration signal",
                            "Measure of signal dispersion",
                            "Root Mean Square value of the vibration signal",
                            "Measure of signal asymmetry",
                            "Measure of signal peakedness",
                            "Ratio of peak value to RMS",
                            "Shape factor = RMS/mean_rectified. Related to signal shape",
                            "Bearing condition (Target Variable)"]

        })

    # Technical Architecture & Features
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### **Key Technical Features**")
        st.markdown("""
        * **Signal Features:** Max, Min, Mean, Standard Deviation, RMS, Kurtosis, Skewness, Crest Factor, Form Factor.
        * **Model Architecture:** Optimised Random Forest Classifier.
        * **Preprocessing Pipeline:** Standard Scaling & Label Encoding.
        * **Target Classes:** Normal State, Inner Race Fault, Outer Race Fault, Ball Fault.
        """)

    with col2:
        st.markdown("### **Project Metadata**")
        st.markdown("""
        * **Domain:** Predictive Maintenance / Condition Monitoring
        * **Framework:** Streamlit, Scikit-Learn, Pandas, Plotly
        * **Dataset:** CWRU Bearing Data Center
        * **Accuracy:** ~94.57% Test Set Accuracy
        """)

    st.markdown("---")

    st.title("Members of the Engineering Group ")
    # Group Members Section
    team = [
        {"name": "Ifeanyichukwu Imaga Agwu", "reg": "TR3/STD/AIM/111"},
        {"name": "Victor Chinedu Ezebuiro", "reg": "TR3/STD/AIM/114"},
        {"name": "Nduka, Chukwuemeka J.", "reg": "TR3/STD/AIM/131"},
        {"name": "Nnokwe Franklin Okechukwu", "reg": "TR3/STD/AIM/107"},
        {"name": "Anosike Joy Ogechi", "reg": "TR3/STD/AIM/140"},
        {"name": "Prosper Chukwuemeka Nwankwo", "reg": "TR3/STD/AIM/054"},
        {"name": "Henry Ajunwa Chibuike", "reg": "TR3/STD/AIM/134"},
        {"name": "Udi Anita Nzubechukwu", "reg": "TR3/STD/AIM/207"},
        {"name": "Okafor Enyinnaya Udeh", "reg": "TR3/STD/AIM/090"},
        {"name": "Chukwuemeka Chimbuchi", "reg": "TR3/STD/AIM/185"},
        {"name": "Isaac Godspower C.", "reg": "TR3/STD/AIM/080"}
    ]

    # Row 1 (3 members)
    r1_cols = st.columns(3)
    for col, member in zip(r1_cols, team[:3]):
        with col:
            st.write(f"**{member['name']}**\n\n{member['reg']}")

    st.divider()

    # Row 2 (3 members)
    r2_cols = st.columns(3)
    for col, member in zip(r2_cols, team[3:6]):
        with col:
            st.write(f"**{member['name']}**\n\n{member['reg']}")

    st.divider()

    # Row 3 (3 members)
    r3_cols = st.columns(3)
    for col, member in zip(r3_cols, team[6:9]):
        with col:
            st.write(f"**{member['name']}**\n\n{member['reg']}")

    st.divider()
    # Row 4 (2 members)
    r3_cols = st.columns(2)
    for col, member in zip(r3_cols, team[9:]):
        with col:
            st.write(f"**{member['name']}**\n\n{member['reg']}")
