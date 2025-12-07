# # anomaly_app.py

# import streamlit as st
# import numpy as np
# import matplotlib.pyplot as plt
# import pickle
# import shap

# def run_anomaly_app():

#     # ----------------------------
#     # Load Model + Threshold
#     # ----------------------------
#     with open("models/anomaly_model.pkl", "rb") as f:
#         model_data = pickle.load(f)

#     autoencoder = model_data["model"]
#     threshold = model_data["threshold"]

#     st.title("✈️ Aircraft Anomaly Detection Simulator")

#     # ----------------------------
#     # User Select Anomaly Injection
#     # ----------------------------
#     anomaly_type = st.selectbox(
#         "Select an anomaly to inject",
#         ["SensorBias", "HydraulicLeak", "EngineFailure",
#          "CabinPressureLoss", "BirdStrike", "FuelLeak", "ElectricalFault"]
#     )

#     n_steps = st.slider("Simulation Time Steps", 10, 50, 25)

#     # ----------------------------
#     # Generate Synthetic Flight Data
#     # ----------------------------
#     def generate_flight_data(n_steps, anomaly=None):
#         np.random.seed(42)
#         data = {
#             "Altitude_ft": np.random.normal(30000, 1000, n_steps),
#             "Airspeed_knots": np.random.normal(250, 10, n_steps),
#             "EngineRPM": np.random.normal(2500, 200, n_steps),
#             "EngineOilPressure_psi": np.random.normal(50, 5, n_steps),
#             "FuelFlow_pph": np.random.normal(800, 50, n_steps),
#             "Vibration_mm_s": np.random.normal(0.5, 0.1, n_steps),
#             "HydraulicPressure_psi": np.random.normal(3000, 100, n_steps),
#         }
#         X = np.column_stack(list(data.values()))

#         # Inject anomaly near the end
#         if anomaly == "SensorBias":
#             X[-5:, 0] += 3000
#         elif anomaly == "HydraulicLeak":
#             X[-5:, 6] -= 500
#         elif anomaly == "EngineFailure":
#             X[-5:, 2] -= 1000
#         elif anomaly == "CabinPressureLoss":
#             X[-5:, 0] += 5000
#         elif anomaly == "BirdStrike":
#             X[-5:, 5] += 2.0
#             X[-5:, 1] -= 50
#         elif anomaly == "FuelLeak":
#             X[-5:, 4] += 400
#         elif anomaly == "ElectricalFault":
#             X[-5:, 3] -= 15
#             X[-5:, 6] -= 800

#         return X, list(data.keys())

#     # ----------------------------
#     # Run Simulation
#     # ----------------------------
#     if st.button("🚀 Run Simulation"):
#         X, feature_names = generate_flight_data(n_steps, anomaly_type)
#         reconstructions = autoencoder.predict(X)
#         errors = np.mean((X - reconstructions) ** 2, axis=1)

#         # SHAP explainability
#         explainer = shap.Explainer(autoencoder, X)
#         shap_values = explainer(X)

#         # ----------------------------
#         # Plot Graph
#         # ----------------------------
#         fig, ax = plt.subplots(figsize=(10, 4))
#         ax.plot(errors, label="Reconstruction Error")
#         ax.axhline(threshold, color="red", linestyle="--", label="Threshold")
#         ax.set_title(f"Anomaly Detection Timeline (Injected: {anomaly_type})")
#         ax.set_xlabel("Time")
#         ax.set_ylabel("Reconstruction Error")
#         ax.legend()
#         st.pyplot(fig)

#         # ----------------------------
#         # Time-by-Time Logs + Summary Box
#         # ----------------------------
#         for t in range(n_steps):

#             status = "NORMAL"
#             if errors[t] > threshold:
#                 status = "🚨 ANOMALY DETECTED!"

#             metrics = {f: X[t, i] for i, f in enumerate(feature_names)}

#             st.markdown(
#                 f"**Time: {t} | Status: {status} | Rec. Error: {errors[t]:.5f} "
#                 f"(Threshold: {threshold:.5f})**"
#             )
#             st.write("   └─ Key Metrics:", metrics)

#             # ==================================================================
#             # 🟥 WHEN ANOMALY OCCURS → SHOW THE BIG SUMMARY BOX (YOUR UI DESIGN)
#             # ==================================================================
#             if status != "NORMAL":

#                 shap_vals = shap_values[t].values.flatten()
#                 shap_vals = np.nan_to_num(shap_vals)
#                 valid_range = len(feature_names)
#                 shap_t = np.argsort(np.abs(shap_vals))[-3:][::-1]

#                 top_factors = []
#                 for idx in shap_t:
#                     if 0 <= idx < valid_range:
#                         top_factors.append({
#                             "feature": feature_names[idx],
#                             "value": float(X[t, idx]),
#                             "shap": float(shap_vals[idx])
#                         })

#                 # ----------------------------
#                 # SUMMARY BOX (Exactly like your screenshot)
#                 # ----------------------------
#                 st.markdown("""
#                     <div style="
#                         border: 3px solid #ff4d4d;
#                         padding: 20px;
#                         border-radius: 10px;
#                         background-color: #ffe6e6;">
#                         <h2 style="color:#cc0000;">🚨 ANOMALY DETECTED</h2>
#                         <p><b>Possible Failure:</b> """ + anomaly_type + """</p>
#                         <p><b>Time Index:</b> """ + str(t) + """</p>
#                         <p><b>Reconstruction Error:</b> """ + str(errors[t]) + """</p>
#                         <p><b>Threshold:</b> """ + str(threshold) + """</p>
#                         <h4>Key Metrics:</h4>
#                 """, unsafe_allow_html=True)

#                 for k, v in metrics.items():
#                     st.markdown(f"**{k}:** {v:.3f}")

#                 st.markdown("<h4>🔎 Explainable AI (Top 3 Factors):</h4>", unsafe_allow_html=True)
#                 for f in top_factors:
#                     st.markdown(
#                         f"**{f['feature']}**: value={f['value']:.3f}, shap={f['shap']:.3f}"
#                     )

#                 st.markdown("</div>", unsafe_allow_html=True)

#                 # store popup data (unchanged)
#                 st.session_state["anomaly_popup"] = {
#                     "time": int(t),
#                     "rec_error": float(errors[t]),
#                     "threshold": float(threshold),
#                     "status": "ANOMALY DETECTED",
#                     "possible_failure": anomaly_type,
#                     "metrics": {k: float(v) for k, v in metrics.items()},
#                     "top_factors": top_factors
#                 }

#                 break


# # Run directly
# if __name__ == "__main__":
#     run_anomaly_app()

# anomaly_app.py

# import streamlit as st
# import numpy as np
# import matplotlib.pyplot as plt
# import pickle
# import shap

# def run_anomaly_app():

#     with open("models/anomaly_model.pkl", "rb") as f:
#         model_data = pickle.load(f)

#     autoencoder = model_data["model"]
#     threshold = model_data["threshold"]

#     st.title("✈️ Aircraft Anomaly Detection Simulator")

#     anomaly_type = st.selectbox(
#         "Select an anomaly to inject",
#         ["SensorBias", "HydraulicLeak", "EngineFailure",
#          "CabinPressureLoss", "BirdStrike", "FuelLeak", "ElectricalFault"]
#     )

#     n_steps = st.slider("Simulation Time Steps", 10, 50, 25)

#     # synthetic data
#     def generate_flight_data(n_steps, anomaly=None):
#         np.random.seed(42)
#         data = {
#             "Altitude_ft": np.random.normal(30000, 1000, n_steps),
#             "Airspeed_knots": np.random.normal(250, 10, n_steps),
#             "EngineRPM": np.random.normal(2500, 200, n_steps),
#             "EngineOilPressure_psi": np.random.normal(50, 5, n_steps),
#             "FuelFlow_pph": np.random.normal(800, 50, n_steps),
#             "Vibration_mm_s": np.random.normal(0.5, 0.1, n_steps),
#             "HydraulicPressure_psi": np.random.normal(3000, 100, n_steps),
#         }
#         X = np.column_stack(list(data.values()))

#         if anomaly == "SensorBias":
#             X[-5:, 0] += 3000
#         elif anomaly == "HydraulicLeak":
#             X[-5:, 6] -= 500
#         elif anomaly == "EngineFailure":
#             X[-5:, 2] -= 1000
#         elif anomaly == "CabinPressureLoss":
#             X[-5:, 0] += 5000
#         elif anomaly == "BirdStrike":
#             X[-5:, 5] += 2.0
#             X[-5:, 1] -= 50
#         elif anomaly == "FuelLeak":
#             X[-5:, 4] += 400
#         elif anomaly == "ElectricalFault":
#             X[-5:, 3] -= 15
#             X[-5:, 6] -= 800

#         return X, list(data.keys())

#     if st.button("🚀 Run Simulation"):

#         X, feature_names = generate_flight_data(n_steps, anomaly_type)
#         recon = autoencoder.predict(X)
#         errors = np.mean((X - recon) ** 2, axis=1)

#         explainer = shap.Explainer(autoencoder, X)
#         shap_values = explainer(X)

#         # plot
#         fig, ax = plt.subplots(figsize=(10, 4))
#         ax.plot(errors, label="Reconstruction Error")
#         ax.axhline(threshold, color="red", linestyle="--", label="Threshold")
#         ax.set_title(f"Anomaly Detection Timeline (Injected: {anomaly_type})")
#         ax.set_xlabel("Time")
#         ax.set_ylabel("Reconstruction Error")
#         ax.legend()
#         st.pyplot(fig)

#         # timeline loop
#         for t in range(n_steps):

#             status = "NORMAL"
#             if errors[t] > threshold:
#                 status = "🚨 ANOMALY DETECTED!"

#             metrics = {f: X[t, i] for i, f in enumerate(feature_names)}

#             st.markdown(
#                 f"**Time: {t} | Status: {status} | Rec. Error: {errors[t]:.5f} "
#                 f"(Threshold: {threshold:.5f})**"
#             )
#             st.write("   └─ Key Metrics:", metrics)

#             if status != "NORMAL":

#                 shap_vals = shap_values[t].values.flatten()
#                 shap_vals = np.nan_to_num(shap_vals)

#                 idxs = np.argsort(np.abs(shap_vals))[-3:][::-1]

#                 top_factors = []
#                 for idx in idxs:
#                     if idx < len(feature_names):
#                         top_factors.append({
#                             "feature": feature_names[idx],
#                             "value": float(X[t, idx]),
#                             "shap": float(shap_vals[idx])
#                         })

#                 # -----------------------------------------
#                 #  FINAL SUMMARY BOX (FULLY FIXED + EXPANDING)
#                 # -----------------------------------------

#                 html = f"""
#                 <div style="
#                     border: 3px solid #ff4d4d;
#                     padding: 25px;
#                     border-radius: 12px;
#                     background-color: #ffe6e6;
#                     margin-top: 20px;
#                 ">
#                     <h2 style="color:#cc0000; margin-top:0;">
#                         🚨 ANOMALY DETECTED
#                     </h2>

#                     <p><b>Possible Failure:</b> {anomaly_type}</p>
#                     <p><b>Time Index:</b> {t}</p>
#                     <p><b>Reconstruction Error:</b> {errors[t]:.5f}</p>
#                     <p><b>Threshold:</b> {threshold:.5f}</p>

#                     <h4 style="margin-top: 20px;"><b>Key Metrics:</b></h4>
#                 """

#                 # metrics inside box
#                 for k, v in metrics.items():
#                     html += f"<p style='margin:2px 0;'><b>{k}</b>: {v:.3f}</p>"

#                 # SHAP inside box
#                 html += """
#                     <h4 style="margin-top: 20px;">
#                         🔎 <b>Explainable AI (Top 3 Factors):</b>
#                     </h4>
#                 """

#                 for f in top_factors:
#                     html += (
#                         f"<p style='margin:2px 0;'>"
#                         f"<b>{f['feature']}</b>: value={f['value']:.3f}, shap={f['shap']:.3f}"
#                         "</p>"
#                     )

#                 html += "</div>"

#                 st.markdown(html, unsafe_allow_html=True)

#                 # store popup
#                 st.session_state["anomaly_popup"] = {
#                     "time": int(t),
#                     "rec_error": float(errors[t]),
#                     "threshold": float(threshold),
#                     "status": "ANOMALY DETECTED",
#                     "possible_failure": anomaly_type,
#                     "metrics": {k: float(v) for k, v in metrics.items()},
#                     "top_factors": top_factors
#                 }

#                 break


# if __name__ == "__main__":
#     run_anomaly_app()


import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pickle
import shap

def run_anomaly_app():

    with open("models/anomaly_model.pkl", "rb") as f:
        model_data = pickle.load(f)

    autoencoder = model_data["model"]
    threshold = model_data["threshold"]

    st.title("Aircraft Anomaly Detection Simulator")

    anomaly_type = st.selectbox(
        "Select an anomaly to inject",
        ["SensorBias", "HydraulicLeak", "EngineFailure",
         "CabinPressureLoss", "BirdStrike", "FuelLeak", "ElectricalFault"]
    )

    n_steps = st.slider("Simulation Time Steps", 10, 50, 25)

    # synthetic data
    def generate_flight_data(n_steps, anomaly=None):
        np.random.seed(42)
        data = {
            "Altitude_ft": np.random.normal(30000, 1000, n_steps),
            "Airspeed_knots": np.random.normal(250, 10, n_steps),
            "EngineRPM": np.random.normal(2500, 200, n_steps),
            "EngineOilPressure_psi": np.random.normal(50, 5, n_steps),
            "FuelFlow_pph": np.random.normal(800, 50, n_steps),
            "Vibration_mm_s": np.random.normal(0.5, 0.1, n_steps),
            "HydraulicPressure_psi": np.random.normal(3000, 100, n_steps),
        }
        X = np.column_stack(list(data.values()))

        if anomaly == "SensorBias":
            X[-5:, 0] += 3000
        elif anomaly == "HydraulicLeak":
            X[-5:, 6] -= 500
        elif anomaly == "EngineFailure":
            X[-5:, 2] -= 1000
        elif anomaly == "CabinPressureLoss":
            X[-5:, 0] += 5000
        elif anomaly == "BirdStrike":
            X[-5:, 5] += 2.0
            X[-5:, 1] -= 50
        elif anomaly == "FuelLeak":
            X[-5:, 4] += 400
        elif anomaly == "ElectricalFault":
            X[-5:, 3] -= 15
            X[-5:, 6] -= 800

        return X, list(data.keys())

    if st.button("Run Simulation"):

        X, feature_names = generate_flight_data(n_steps, anomaly_type)
        recon = autoencoder.predict(X)
        errors = np.mean((X - recon) ** 2, axis=1)

        explainer = shap.Explainer(autoencoder, X)
        shap_values = explainer(X)

        # plot
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(errors, label="Reconstruction Error")
        ax.axhline(threshold, color="red", linestyle="--", label="Threshold")
        ax.set_title(f"Anomaly Detection Timeline (Injected: {anomaly_type})")
        ax.set_xlabel("Time")
        ax.set_ylabel("Reconstruction Error")
        ax.legend()
        st.pyplot(fig)

        # timeline loop
        for t in range(n_steps):

            status = "NORMAL"
            if errors[t] > threshold:
                status = "🚨 ANOMALY DETECTED!"

            metrics = {f: X[t, i] for i, f in enumerate(feature_names)}

            st.markdown(
                f"**Time: {t} | Status: {status} | Rec. Error: {errors[t]:.5f} "
                f"(Threshold: {threshold:.5f})**"
            )
            st.write("   └─ Key Metrics:", metrics)

            if status != "NORMAL":

                shap_vals = shap_values[t].values.flatten()
                shap_vals = np.nan_to_num(shap_vals)

                idxs = np.argsort(np.abs(shap_vals))[-3:][::-1]

                top_factors = []
                for idx in idxs:
                    if idx < len(feature_names):
                        top_factors.append({
                            "feature": feature_names[idx],
                            "value": float(X[t, idx]),
                            "shap": float(shap_vals[idx])
                        })

                # -----------------------------------------
                #  FINAL SUMMARY BOX (FULLY FIXED + EXPANDING)
                # -----------------------------------------

                # Build metrics HTML
                metrics_html = ""
                for k, v in metrics.items():
                    metrics_html += "<p style='margin:2px 0;'><b>{}</b>: {:.3f}</p>".format(k, v)

                # Build factors HTML
                factors_html = ""
                for f in top_factors:
                    factors_html += "<p style='margin:2px 0;'><b>{}</b>: value={:.3f}, shap={:.3f}</p>".format(
                        f['feature'], f['value'], f['shap']
                    )

                # Build complete HTML
                html = """
                <div style="border: 3px solid #ff4d4d; padding: 25px; border-radius: 12px; background-color: #ffe6e6; margin-top: 20px;">
                    <h2 style="color:#cc0000; margin-top:0;">🚨 ANOMALY DETECTED</h2>
                    <p><b>Possible Failure:</b> {}</p>
                    <p><b>Time Index:</b> {}</p>
                    <p><b>Reconstruction Error:</b> {:.5f}</p>
                    <p><b>Threshold:</b> {:.5f}</p>
                    <h4 style="margin-top: 20px;"><b>Key Metrics:</b></h4>
                    {}
                    <h4 style="margin-top: 20px;">🔎 <b>Explainable AI (Top 3 Factors):</b></h4>
                    {}
                </div>
                """.format(anomaly_type, t, errors[t], threshold, metrics_html, factors_html)

                st.markdown(html, unsafe_allow_html=True)

                # store popup
                st.session_state["anomaly_popup"] = {
                    "time": int(t),
                    "rec_error": float(errors[t]),
                    "threshold": float(threshold),
                    "status": "ANOMALY DETECTED",
                    "possible_failure": anomaly_type,
                    "metrics": {k: float(v) for k, v in metrics.items()},
                    "top_factors": top_factors
                }

                break


if __name__ == "__main__":
    run_anomaly_app()