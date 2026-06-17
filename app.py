import streamlit as st
import cv2
import requests
import base64
import time
import numpy as np
import threading
import tempfile
import os

st.set_page_config(
    page_title="VLM & MOT Stream Analyzer",
    page_icon="🎥",
    layout="wide"
)

# Initialize Session State
if 'rtsp_reader' not in st.session_state:
    st.session_state.rtsp_reader = None
if 'rtsp_url' not in st.session_state:
    st.session_state.rtsp_url = ""
if 'log_history' not in st.session_state:
    st.session_state.log_history = []

# Persistent Multi-Object Tracking State
if 'track_history' not in st.session_state:
    st.session_state.track_history = {}
if 'track_first_seen' not in st.session_state:
    st.session_state.track_first_seen = {}
if 'track_last_seen' not in st.session_state:
    st.session_state.track_last_seen = {}
if 'track_logged_entry' not in st.session_state:
    st.session_state.track_logged_entry = set()
if 'track_logged_exit' not in st.session_state:
    st.session_state.track_logged_exit = set()

def clear_tracking_state():
    st.session_state.track_history = {}
    st.session_state.track_first_seen = {}
    st.session_state.track_last_seen = {}
    st.session_state.track_logged_entry = set()
    st.session_state.track_logged_exit = set()

class RTSPThreadReader:
    def __init__(self, url):
        self.url = url
        self.cap = cv2.VideoCapture(url)
        self.latest_frame = None
        self.running = True
        self.thread = threading.Thread(target=self._grab_frames, daemon=True)
        self.thread.start()

    def _grab_frames(self):
        while self.running:
            ret = self.cap.grab()
            if not ret:
                time.sleep(2)
                self.cap.release()
                self.cap = cv2.VideoCapture(self.url)
                continue
            _, self.latest_frame = self.cap.retrieve()

    def get_frame(self):
        return self.latest_frame

    def stop(self):
        self.running = False
        self.cap.release()

# Fetch models from Ollama API
@st.cache_data(ttl=10)
def get_ollama_models(ollama_url):
    try:
        response = requests.get(f"{ollama_url}/api/tags", timeout=3)
        if response.status_code == 200:
            models = [m["name"] for m in response.json().get("models", [])]
            vision_keywords = ["moondream", "vision", "llava"]
            sorted_models = sorted(models, key=lambda x: not any(k in x.lower() for k in vision_keywords))
            return sorted_models
    except Exception:
        pass
    return ["moondream:latest", "llama3.2-vision:latest", "llava:latest"]

# Lazy-load YOLO model
@st.cache_resource
def load_yolo_model():
    from ultralytics import YOLO
    return YOLO("yolov8n.pt")

# Query VLM Model
def query_vlm(ollama_url, model, prompt, frame, jpeg_quality=60):
    # Downscale frame for speed
    h, w = frame.shape[:2]
    target_height = 360  # Optimized for CPU prefill
    target_width = int(w * (target_height / h))
    resized_frame = cv2.resize(frame, (target_width, target_height))

    # Base64 Encode with JPEG compression optimization
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    _, buffer = cv2.imencode('.jpg', resized_frame, encode_param)
    b64_image = base64.b64encode(buffer).decode('utf-8')

    payload = {
        "model": model,
        "prompt": prompt,
        "images": [b64_image],
        "stream": False
    }
    
    response = requests.post(f"{ollama_url}/api/generate", json=payload, timeout=120)
    if response.status_code == 200:
        return response.json().get("response", "")
    else:
        raise Exception(f"Ollama returned error code {response.status_code}")

# High-speed motion difference analysis (1ms execution time)
def detect_motion(prev_frame, curr_frame, sensitivity_threshold=1.5):
    if prev_frame is None:
        return True
    g1 = cv2.cvtColor(cv2.resize(prev_frame, (100, 100)), cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(cv2.resize(curr_frame, (100, 100)), cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(g1, g2)
    changed_pixels = np.sum(diff > 25)
    change_percent = (changed_pixels / diff.size) * 100
    return change_percent > sensitivity_threshold

# Generate thumbnails for detected objects
def generate_thumbnails(frame, boxes, names_dict):
    thumbnails_html = []
    h_img, w_img = frame.shape[:2]
    
    for box in boxes:
        cls_id = int(box.cls[0])
        cls_name = names_dict[cls_id]
        
        # Pull track ID if present
        track_suffix = ""
        if box.id is not None:
            track_suffix = f" #{int(box.id[0])}"
            
        xyxy = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = map(int, xyxy)
        
        # Add padding margin
        pad = 8
        x1_pad = max(0, x1 - pad)
        y1_pad = max(0, y1 - pad)
        x2_pad = min(w_img, x2 + pad)
        y2_pad = min(h_img, y2 + pad)
        
        crop = frame[y1_pad:y2_pad, x1_pad:x2_pad]
        if crop.size > 0:
            _, crop_buffer = cv2.imencode('.jpg', crop, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
            crop_b64 = base64.b64encode(crop_buffer).decode('utf-8')
            thumbnails_html.append(
                f'<div style="display:inline-block; margin-right:8px; text-align:center;">'
                f'<img src="data:image/jpeg;base64,{crop_b64}" style="height:60px; border:1px solid #ddd; border-radius:4px; vertical-align:middle;"><br/>'
                f'<span style="font-size:9px; color:#777; font-weight:bold;">{cls_name}{track_suffix}</span>'
                f'</div>'
            )
    return "".join(thumbnails_html)

# Draw motion trails for tracked targets
def draw_motion_trails(frame, track_history):
    for track_id, points in track_history.items():
        if len(points) < 2:
            continue
        # Select persistent unique color per track ID
        color = (
            int((track_id * 73) % 200 + 55),
            int((track_id * 149) % 200 + 55),
            int((track_id * 223) % 200 + 55)
        )
        for i in range(len(points) - 1):
            pt1 = points[i]
            pt2 = points[i+1]
            thickness = int(np.clip(1 + i * 0.15, 1, 4)) # Trail grows thicker at current position
            cv2.line(frame, pt1, pt2, color, thickness)

# Render UI
st.title("🎥 Hybrid VLM & MOT Stream Analyzer")
st.markdown("Analyze live video streams or files using high-speed multi-object tracking (**YOLOv8 + ByteTrack**) or rich semantic **VLMs (via Ollama)** on CPU.")

# Sidebar Configuration
st.sidebar.header("⚙️ Model Configuration")
analysis_engine = st.sidebar.radio(
    "Analysis Engine",
    options=["👤 Persistent Tracking (YOLOv8 + ByteTrack)", "🤖 Vision-Language Model (VLM)"],
    help="YOLOv8 with ByteTrack maintains ID tracking, movement trails, and calculates dwell-time on CPU. VLM provides deep conversational descriptions."
)

sample_interval = st.sidebar.slider("Sampling Interval (seconds)", min_value=0.1 if "YOLO" in analysis_engine else 1.0, max_value=30.0, value=0.5 if "YOLO" in analysis_engine else 3.0, step=0.1 if "YOLO" in analysis_engine else 0.5)

if "VLM" in analysis_engine:
    st.sidebar.subheader("🤖 VLM Parameters")
    ollama_endpoint = st.sidebar.text_input("Ollama Base URL", value="http://ollama:11434")
    available_models = get_ollama_models(ollama_endpoint)
    model_name = st.sidebar.selectbox("VLM Model", options=available_models)
    prompt_text = st.sidebar.text_area("VLM Prompt", value="Describe what is happening in this frame concisely. Identify people, actions, or notable objects.")
    jpeg_quality = st.sidebar.slider("JPEG Compression Quality", min_value=10, max_value=100, value=50, step=5)
    use_motion_skip = st.sidebar.checkbox("Motion Detection Auto-Skip", value=True)
    motion_sensitivity = st.sidebar.slider("Motion Sensitivity Threshold (%)", min_value=0.1, max_value=10.0, value=1.5, step=0.1)
else:
    st.sidebar.subheader("👤 YOLO + ByteTrack Parameters")
    confidence_threshold = st.sidebar.slider("Detection Confidence", min_value=0.1, max_value=1.0, value=0.25, step=0.05)
    target_classes = st.sidebar.multiselect("Classes to Track", options=["person", "bicycle", "car", "motorcycle", "backpack", "umbrella", "handbag", "dog", "cat"], default=["person"])
    
    coco_classes = {"person": 0, "bicycle": 1, "car": 2, "motorcycle": 3, "backpack": 24, "umbrella": 25, "handbag": 26, "dog": 16, "cat": 15}
    selected_class_ids = [coco_classes[cls] for cls in target_classes]
    
    show_trails = st.sidebar.checkbox("Draw Motion Trail Lines", value=True, help="Renders colorful trajectory paths illustrating the movement history of each tracked object.")

# Tabs for Mode
tab1, tab2 = st.tabs(["🔴 RTSP / Live Stream", "📤 Local Video Upload"])

# TAB 1: RTSP Stream
with tab1:
    st.subheader("RTSP Stream Processor")
    rtsp_input = st.text_input("RTSP Stream URL", value="rtsp://")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        frame_placeholder = st.empty()
        status_placeholder = st.empty()
    with col2:
        st.write("📋 **Real-time Event Log**")
        log_placeholder = st.empty()

    start_btn = st.button("Start Live Stream Analysis", key="start_rtsp")
    stop_btn = st.button("Stop Live Stream Analysis", key="stop_rtsp")

    if stop_btn:
        if st.session_state.rtsp_reader:
            st.session_state.rtsp_reader.stop()
            st.session_state.rtsp_reader = None
        st.success("Stopped RTSP processing.")
        st.rerun()

    if start_btn and rtsp_input and rtsp_input != "rtsp://":
        if st.session_state.rtsp_reader:
            st.session_state.rtsp_reader.stop()
        
        clear_tracking_state()
        st.session_state.rtsp_reader = RTSPThreadReader(rtsp_input)
        st.session_state.rtsp_url = rtsp_input
        st.success("Connected and streaming in background...")
        
        last_analyzed = 0
        last_frame_processed = None
        
        while st.session_state.rtsp_reader is not None:
            frame = st.session_state.rtsp_reader.get_frame()
            if frame is not None:
                now = time.time()
                if now - last_analyzed >= sample_interval:
                    last_analyzed = now
                    
                    if "VLM" in analysis_engine:
                        if use_motion_skip and last_frame_processed is not None:
                            is_moving = detect_motion(last_frame_processed, frame, motion_sensitivity)
                            if not is_moving:
                                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                frame_placeholder.image(rgb_frame, channels="RGB", use_column_width=True)
                                status_placeholder.info("⚡ Skipping frame: No motion detected.")
                                continue
                        
                        status_placeholder.info("🤖 Analyzing frame with VLM...")
                        try:
                            t_start = time.time()
                            analysis = query_vlm(ollama_endpoint, model_name, prompt_text, frame, jpeg_quality)
                            t_elapsed = time.time() - t_start
                            
                            last_frame_processed = frame.copy()
                            timestamp = time.strftime('%H:%M:%S')
                            event = f"**[{timestamp}]** *(VLM Inference: {t_elapsed:.1f}s)*:  \n{analysis}\n\n---"
                            st.session_state.log_history.insert(0, event)
                            status_placeholder.success(f"VLM Analysis completed in {t_elapsed:.1f}s!")
                            
                            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            frame_placeholder.image(rgb_frame, channels="RGB", use_column_width=True)
                        except Exception as e:
                            status_placeholder.error(f"Error during VLM inference: {e}")
                    else:
                        # YOLO + ByteTrack (Highly robust looping)
                        status_placeholder.info("👤 Running YOLOv8 + ByteTrack Tracker...")
                        try:
                            t_start = time.time()
                            yolo_model = load_yolo_model()
                            
                            classes_arg = selected_class_ids if selected_class_ids else None
                            results = yolo_model.track(frame, persist=True, tracker="bytetrack.yaml", conf=confidence_threshold, classes=classes_arg, verbose=False)
                            t_elapsed = time.time() - t_start
                            
                            annotated_frame = results[0].plot()
                            boxes = results[0].boxes
                            active_ids = set()
                            spatial_events = []
                            
                            if boxes.id is not None:
                                for box in boxes:
                                    if box.id is not None:
                                        track_id = int(box.id[0])
                                        active_ids.add(track_id)
                                        cls_id = int(box.cls[0])
                                        cls_name = yolo_model.names[cls_id]
                                        
                                        xyxy_coords = box.xyxy[0].cpu().numpy()
                                        x1, y1, x2, y2 = map(int, xyxy_coords)
                                        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                                        
                                        # Update track history centroids
                                        if track_id not in st.session_state.track_history:
                                            st.session_state.track_history[track_id] = []
                                        st.session_state.track_history[track_id].append((cx, cy))
                                        if len(st.session_state.track_history[track_id]) > 30:
                                            st.session_state.track_history[track_id].pop(0)
                                            
                                        # Entry logging
                                        if track_id not in st.session_state.track_first_seen:
                                            st.session_state.track_first_seen[track_id] = now
                                        st.session_state.track_last_seen[track_id] = now
                                        
                                        if track_id not in st.session_state.track_logged_entry:
                                            spatial_events.append(f"🟢 **{cls_name.capitalize()} #{track_id}** entered.")
                                            st.session_state.track_logged_entry.add(track_id)
                            
                            # Exit / Dwell-time logging
                            for track_id in list(st.session_state.track_first_seen.keys()):
                                if track_id not in active_ids and (now - st.session_state.track_last_seen[track_id]) > 3.0:
                                    if track_id not in st.session_state.track_logged_exit:
                                        dwell = st.session_state.track_last_seen[track_id] - st.session_state.track_first_seen[track_id]
                                        spatial_events.append(f"🔴 **ID #{track_id}** exited (Dwell time: {dwell:.1f}s).")
                                        st.session_state.track_logged_exit.add(track_id)
                            
                            # Render motion trails
                            if show_trails and st.session_state.track_history:
                                draw_motion_trails(annotated_frame, st.session_state.track_history)
                                
                            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                            frame_placeholder.image(rgb_frame, channels="RGB", use_column_width=True)
                            
                            # Count objects
                            detections = {}
                            for box in boxes:
                                cls_name = yolo_model.names[int(box.cls[0])]
                                detections[cls_name] = detections.get(cls_name, 0) + 1
                            det_summary = ", ".join([f"{count} {name}(s)" for name, count in detections.items()]) if detections else "No targets tracked"
                            
                            # Base64 crops
                            thumbs_html = generate_thumbnails(frame, boxes, yolo_model.names)
                            thumbs_div = f"<div style='margin-top:6px;'>{thumbs_html}</div>" if thumbs_html else ""
                            spatial_str = " | ".join(spatial_events) if spatial_events else ""
                            spatial_p = f"<div style='font-size:11px; color:#ef5350; font-weight:bold;'>{spatial_str}</div>" if spatial_str else ""
                            
                            timestamp = time.strftime('%H:%M:%S')
                            event = f"**[{timestamp}]** *(MOT Inference: {t_elapsed*1000:.1f}ms)*:  \n🎯 **Active Detections:** {det_summary}\n{spatial_p}{thumbs_div}\n\n---"
                            st.session_state.log_history.insert(0, event)
                            status_placeholder.success(f"MOT Ran in {t_elapsed*1000:.1f}ms!")
                        except Exception as e:
                            status_placeholder.error(f"Error during MOT tracking: {e}")
                else:
                    if "VLM" in analysis_engine or last_frame_processed is None:
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame_placeholder.image(rgb_frame, channels="RGB", use_column_width=True)
            else:
                status_placeholder.warning("Waiting for RTSP stream frames...")
                
            if st.session_state.log_history:
                log_placeholder.markdown("\n".join(st.session_state.log_history[:20]), unsafe_allow_html=True)
            time.sleep(0.01)

# TAB 2: Video Upload
with tab2:
    st.subheader("Offline Video Processor")
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])
    
    if uploaded_file is not None:
        st.info("Video uploaded successfully. Ready for frame-by-frame analysis.")
        
        col1_vid, col2_vid = st.columns([2, 1])
        with col1_vid:
            vid_frame_placeholder = st.empty()
            vid_status = st.empty()
            progress_bar = st.progress(0.0)
        with col2_vid:
            st.write("📋 **Video Event Log**")
            vid_log_placeholder = st.empty()
            
        process_vid_btn = st.button("Run Offline Video Analysis")
        
        if process_vid_btn:
            vid_logs = []
            clear_tracking_state()
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
                tfile.write(uploaded_file.read())
                temp_filename = tfile.name
                
            cap = cv2.VideoCapture(temp_filename)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            if total_frames <= 0 or fps <= 0:
                st.error("Error reading video headers.")
            else:
                duration = total_frames / fps
                st.write(f"🎞️ Duration: {duration:.1f}s | Frames: {total_frames} | FPS: {fps:.1f}")
                
                frame_step = int(fps * sample_interval)
                if frame_step <= 0:
                    frame_step = 1
                
                last_frame_processed = None
                current_frame_idx = 0
                
                while cap.isOpened():
                    if current_frame_idx > 0:
                        for _ in range(frame_step - 1):
                            grabbed = cap.grab()
                            if not grabbed:
                                break
                    
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    current_frame_idx += frame_step
                    time_sec = (current_frame_idx - frame_step) / fps
                    
                    progress_val = min((current_frame_idx) / total_frames, 1.0)
                    progress_bar.progress(progress_val)
                    
                    if "VLM" in analysis_engine:
                        if use_motion_skip and last_frame_processed is not None:
                            is_moving = detect_motion(last_frame_processed, frame, motion_sensitivity)
                            if not is_moving:
                                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                vid_frame_placeholder.image(rgb_frame, channels="RGB", use_column_width=True)
                                vid_status.info(f"⚡ Skipping frame at {time_sec:.1f}s: No motion detected.")
                                continue
                        
                        vid_status.info(f"🤖 Analyzing frame at timestamp: {time_sec:.1f}s / {duration:.1f}s...")
                        try:
                            t_start = time.time()
                            analysis = query_vlm(ollama_endpoint, model_name, prompt_text, frame, jpeg_quality)
                            t_elapsed = time.time() - t_start
                            
                            last_frame_processed = frame.copy()
                            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            vid_frame_placeholder.image(rgb_frame, channels="RGB", use_column_width=True)
                            
                            timestamp_str = f"{int(time_sec // 60):02d}:{int(time_sec % 60):02d}"
                            event = f"**[{timestamp_str}]** *(VLM Inference: {t_elapsed:.1f}s)*:  \n{analysis}\n\n---"
                            vid_logs.insert(0, event)
                            vid_log_placeholder.markdown("\n".join(vid_logs), unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"Error at frame {current_frame_idx}: {e}")
                    else:
                        # YOLO + ByteTrack Offline (Ultra robust indexing)
                        vid_status.info(f"👤 Tracking targets at {time_sec:.1f}s...")
                        try:
                            t_start = time.time()
                            yolo_model = load_yolo_model()
                            
                            classes_arg = selected_class_ids if selected_class_ids else None
                            results = yolo_model.track(frame, persist=True, tracker="bytetrack.yaml", conf=confidence_threshold, classes=classes_arg, verbose=False)
                            t_elapsed = time.time() - t_start
                            
                            annotated_frame = results[0].plot()
                            boxes = results[0].boxes
                            active_ids = set()
                            spatial_events = []
                            
                            if boxes.id is not None:
                                for box in boxes:
                                    if box.id is not None:
                                        track_id = int(box.id[0])
                                        active_ids.add(track_id)
                                        cls_id = int(box.cls[0])
                                        cls_name = yolo_model.names[cls_id]
                                        
                                        xyxy_coords = box.xyxy[0].cpu().numpy()
                                        x1, y1, x2, y2 = map(int, xyxy_coords)
                                        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                                        
                                        if track_id not in st.session_state.track_history:
                                            st.session_state.track_history[track_id] = []
                                        st.session_state.track_history[track_id].append((cx, cy))
                                        if len(st.session_state.track_history[track_id]) > 30:
                                            st.session_state.track_history[track_id].pop(0)
                                            
                                        if track_id not in st.session_state.track_first_seen:
                                            st.session_state.track_first_seen[track_id] = time_sec
                                        st.session_state.track_last_seen[track_id] = time_sec
                                        
                                        if track_id not in st.session_state.track_logged_entry:
                                            spatial_events.append(f"🟢 **{cls_name.capitalize()} #{track_id}** entered.")
                                            st.session_state.track_logged_entry.add(track_id)
                            
                            # Exit / Dwell-time logging
                            for track_id in list(st.session_state.track_first_seen.keys()):
                                if track_id not in active_ids and (time_sec - st.session_state.track_last_seen[track_id]) > (sample_interval * 3):
                                    if track_id not in st.session_state.track_logged_exit:
                                        dwell = st.session_state.track_last_seen[track_id] - st.session_state.track_first_seen[track_id]
                                        spatial_events.append(f"🔴 **ID #{track_id}** exited (Dwell time: {dwell:.1f}s).")
                                        st.session_state.track_logged_exit.add(track_id)
                                        
                            if show_trails and st.session_state.track_history:
                                draw_motion_trails(annotated_frame, st.session_state.track_history)
                                
                            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                            vid_frame_placeholder.image(rgb_frame, channels="RGB", use_column_width=True)
                            
                            # Count objects
                            detections = {}
                            for box in boxes:
                                cls_name = yolo_model.names[int(box.cls[0])]
                                detections[cls_name] = detections.get(cls_name, 0) + 1
                            det_summary = ", ".join([f"{count} {name}(s)" for name, count in detections.items()]) if detections else "No targets tracked"
                            
                            thumbs_html = generate_thumbnails(frame, boxes, yolo_model.names)
                            thumbs_div = f"<div style='margin-top:6px;'>{thumbs_html}</div>" if thumbs_html else ""
                            spatial_str = " | ".join(spatial_events) if spatial_events else ""
                            spatial_p = f"<div style='font-size:11px; color:#ef5350; font-weight:bold;'>{spatial_str}</div>" if spatial_str else ""
                            
                            timestamp_str = f"{int(time_sec // 60):02d}:{int(time_sec % 60):02d}"
                            event = f"**[{timestamp_str}]** *(MOT Inference: {t_elapsed*1000:.1f}ms)*:  \n🎯 **Active Detections:** {det_summary}\n{spatial_p}{thumbs_div}\n\n---"
                            vid_logs.insert(0, event)
                            vid_log_placeholder.markdown("\n".join(vid_logs), unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"Error during MOT tracking: {e}")
                
                progress_bar.progress(1.0)
                vid_status.success("Video processing complete!")
                cap.release()
                os.unlink(temp_filename)
