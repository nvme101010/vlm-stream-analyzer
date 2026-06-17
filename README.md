# 🎥 Hybrid VLM & Object Detection Stream Analyzer

An ultra-high-efficiency, local Computer Vision (CV) and Vision-Language Model (VLM) suite designed to monitor, analyze, and index RTSP camera streams and video files on CPU-bound hosts.

This application includes a beautiful **Streamlit Web-UI** that exposes two plug-and-play inference engines:
1. **👤 Person & Object Detection (YOLOv8):** Runs at **30-50+ FPS** natively on modern CPU cores, generating an interactive real-time visual event timeline with cropped target thumbnails.
2. **🤖 Vision-Language Model (Moondream/Llava via Ollama):** For deep semantic, conversational queries about frame contents (e.g. *"Describe actions, identify people, or find objects"*).

---

## ⚡ High-Performance CPU Optimizations

To run multi-modal AI on CPU-bound VMs without thread-thrashing or buffer lag, the pipeline implements three architectural pillars:

1. **Intelligent Motion-Based Auto-Skipping (1ms Overhead):**
   Uses a lightweight pixel luminance absolute difference matrix (`cv2.absdiff`) on downscaled frames. If consecutive frames are static (no motion), VLM calls are bypassed entirely—saving up to **90%** of processing times on security feeds.
2. **Sequential Skip-Grabbing (10x Faster Video Decoding):**
   Bypasses heavy `cap.set()` seeking in OpenCV (which forces keyframe seek calculations). Instead, the video decoder loops sequentially and uses `.grab()` to skip non-target packets off the buffer at lightning speed without decompressing them.
3. **In-Memory Visual Thumbnails:**
   When objects are detected, YOLO coordinates crop the frame in-memory, compress them to lightweight JPEGs (50-60% quality), and encode them directly into inline Base64 HTML `<img>` items. **No disk I/O clutter is generated on the host.**

---

## 🏗️ Architecture & Stack

```
                     +---------------------------------------+
                     |         Web Browser (Port 8501)       |
                     +-------------------+-------------------+
                                         |
                                         v
                     +-------------------+-------------------+
                     |      Streamlit VLM-CV App Container   |
                     +---------+-------------------+---------+
                               |                   |
                     (Internal | Port 11434)       | (In-Memory YOLO)
                               v                   v
+------------------------------+----+   +----------+----------+
|  Ollama Service Container         |   |  PyTorch / YOLOv8   |
|  - moondream:latest (1.7GB)       |   |  - yolov8n.pt (6MB) |
|  - llava:latest (4.7GB)           |   +---------------------+
+-----------------------------------+
```

---

## 🛠️ Quick Start (Docker Compose)

The application binds to the same bridge network as Ollama (`ollama-webui_default`) to maintain zero-latency communications.

### 1. File Structure
Create a dedicated folder on your host:
```bash
mkdir -p ~/vlm-stream-analyzer
cd ~/vlm-stream-analyzer
```

### 2. Deployment Recipe
Create the `docker-compose.yml`:
```yaml
services:
  vlm-stream-analyzer:
    build: .
    container_name: vlm-stream-analyzer
    restart: always
    ports:
      - "8501:8501"
    networks:
      - ollama-net

networks:
  ollama-net:
    name: ollama-webui_default
    external: true
```

### 3. Build and Spin Up
```bash
docker compose up -d --build
```

Access the UI at: **`http://<your_host_ip>:8501`**

---

## 🚀 Model Details
* **YOLOv8 Nano (`yolov8n.pt`):** Extremely lightweight (~6MB). Automatically downloaded on first selection. Perfect for person counting, security tripwires, and high-speed motion tracking.
* **Moondream (`moondream:latest`):** A fast, high-quality 1.7B parameter VLM ideal for CPU semantic descriptions.
* **Llava (`llava:latest`):** A stable 7B CLIP-based VLM fallback for deeper semantic descriptions.

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
