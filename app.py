import streamlit as st
import subprocess
import os
import pandas as pd

st.set_page_config(page_title="Attendance System", page_icon="📷", layout="wide")

st.title("Real-Time Facial Recognition Attendance System")
st.sidebar.title("Navigation")
menu = ["Dashboard", "Add New User", "Train Model", "Run Recognition", "Delete User"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Dashboard":
    st.subheader("Attendance Dashboard")
    if os.path.exists("attendance.csv") and os.path.getsize("attendance.csv") > 0:
        try:
            df = pd.read_csv("attendance.csv")
            st.dataframe(df, use_container_width=True)
            
            with open("attendance.csv", "rb") as f:
                st.download_button("Download Attendance Log", f, "attendance.csv", "text/csv")
        except Exception as e:
            st.error(f"Error reading attendance file: {e}")
    else:
        st.info("No attendance records found. Run Recognition to start logging.")

elif choice == "Add New User":
    st.subheader("Add New User")
    st.write("Enter the name of the new user and click capture. A webcam window will open to take multiple face samples.")
    name = st.text_input("Enter User Name (e.g., John_Doe)")
    if st.button("Capture Faces"):
        if name:
            st.info(f"Opening webcam to capture faces for {name}... Please look at the camera and wait until the window closes.")
            subprocess.run(["python", "capture.py", name])
            st.success(f"Faces captured successfully for {name}! Please go to 'Train Model' next.")
        else:
            st.warning("Please enter a valid name.")

elif choice == "Train Model":
    st.subheader("Train Recognition Model")
    st.write("Click the button below to train the ML model on the newly captured faces.")
    if st.button("Start Training"):
        with st.spinner("Training model with dataset..."):
            subprocess.run(["python", "model.py"])
        st.success("Model trained successfully! You can now run recognition.")

elif choice == "Run Recognition":
    st.subheader("Run Face Recognition")
    st.write("Click below to open the webcam and start taking attendance. Press 'q' in the camera window or click 'Stop Camera' below to stop.")
    
    col1, col2 = st.columns(2)
    with col1:
        start_btn = st.button("Start Camera")
    with col2:
        stop_btn = st.button("Stop Camera")
        
    if start_btn:
        if "rec_process" not in st.session_state or st.session_state.rec_process.poll() is not None:
            st.info("Starting camera... Looking for faces...")
            # Use Popen so Streamlit doesn't block
            st.session_state.rec_process = subprocess.Popen(["python", "recognize.py"])
        else:
            st.warning("Camera is already running!")
            
    if stop_btn:
        if "rec_process" in st.session_state and st.session_state.rec_process.poll() is None:
            st.session_state.rec_process.terminate()
            st.session_state.rec_process.wait()
            st.success("Recognition session ended. Check Dashboard for updates.")
        else:
            st.warning("Camera is not running.")

elif choice == "Delete User":
    st.subheader("Delete User Data")
    st.write("Remove a user's face data from the dataset. Remember to retrain the model afterwards!")
    
    if os.path.exists("dataset") and os.listdir("dataset"):
        users = [d for d in os.listdir("dataset") if os.path.isdir(os.path.join("dataset", d))]
        if users:
            user_to_delete = st.selectbox("Select User to Delete", ["Select..."] + users)
            if user_to_delete != "Select...":
                if st.button("Delete User"):
                    import shutil
                    import stat
                    import time
                    
                    dir_path = os.path.join("dataset", user_to_delete)
                    
                    def handle_remove_readonly(func, path, exc):
                        try:
                            os.chmod(path, stat.S_IWRITE)
                            func(path)
                        except Exception:
                            pass
                            
                    try:
                        # Handle read-only files (Common on Windows)
                        try:
                            shutil.rmtree(dir_path, onexc=handle_remove_readonly)
                        except TypeError:
                            shutil.rmtree(dir_path, onerror=handle_remove_readonly)
                            
                        # Double check if it was deleted
                        if not os.path.exists(dir_path):
                            st.success(f"Face data for {user_to_delete} deleted successfully! Please go to 'Train Model' to update the system.")
                            time.sleep(1.5)
                            st.rerun()
                        else:
                            st.error("Could not completely delete the folder. Ensure the camera is stopped and try again.")
                    except Exception as e:
                        st.error(f"Access Denied: Please make sure the camera is stopped and try again. Error: {e}")
        else:
            st.info("No users found.")
    else:
        st.info("Dataset folder is empty or does not exist.")
