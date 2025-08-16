# --- Start of app.py (Final Version with Anti-Blocking Tweak) ---
from flask import Flask, render_template, request, send_file, jsonify, after_this_request
import yt_dlp
import cv2
from fpdf import FPDF
from PIL import Image
import os
import tempfile
import shutil
import uuid
import threading
import io
import re

app = Flask(__name__)

# A thread-safe dictionary to store the status of our jobs.
jobs = {}
jobs_lock = threading.Lock()

# --- Optimized Functions ---

def download_video(youtube_url, output_dir, job_id):
    """Downloads video and updates progress using a hook."""
    def progress_hook(d):
        with jobs_lock:
            if d['status'] == 'downloading' and job_id in jobs:
                percent_str = d.get('_percent_str', '0%').replace('%','').strip()
                try:
                    percent = float(percent_str)
                    # Download progress is the first 50% of the total progress
                    jobs[job_id]['progress'] = percent / 2
                    jobs[job_id]['stage'] = 'Downloading Video'
                except (ValueError, TypeError):
                    pass # Ignore if percentage is not a number

    ydl_opts = {
        'outtmpl': os.path.join(output_dir, 'video.%(ext)s'),
        'format': 'bestvideo[height<=720][ext=mp4]/best[height<=720][ext=mp4]',
        'progress_hooks': [progress_hook],
        # --- FIX: Add a browser-like User-Agent to avoid bot detection ---
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
        },
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(youtube_url, download=True)
            filename = ydl.prepare_filename(info)
            title = info.get('title', 'video_slides')
            return filename, title
        except Exception as e:
            print(f"Error downloading video: {e}")
            return None, None

def frames_to_pdf_generator(video_path, job_id, sampling_rate_fps=1, scene_change_threshold=0.7):
    """
    HIGHLY OPTIMIZED: Resizes frames before comparison and updates progress.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_interval = int(fps / sampling_rate_fps) or 1

    last_good_frame = None
    prev_hist = None
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    compare_size = (200, 112)

    while frame_count < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        ret, frame = cap.read()
        if not ret:
            break

        with jobs_lock:
            if job_id in jobs:
                analysis_progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                # Analysis progress is the second 50% of the total progress
                jobs[job_id]['progress'] = 50 + (analysis_progress / 2)
                jobs[job_id]['stage'] = 'Analyzing Video for Slides'

        small_frame = cv2.resize(frame, compare_size)
        hsv_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2HSV)
        current_hist = cv2.calcHist([hsv_frame], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(current_hist, current_hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        if prev_hist is not None:
            score = cv2.compareHist(prev_hist, current_hist, cv2.HISTCMP_CORREL)
            if score < scene_change_threshold:
                if last_good_frame is not None:
                    yield last_good_frame
        
        last_good_frame = frame
        prev_hist = current_hist
        frame_count += frame_interval

    if last_good_frame is not None:
        yield last_good_frame

    cap.release()

def save_frames_to_pdf(frame_generator, output_pdf):
    """Robust PDF creation using secure temporary files."""
    pdf = FPDF('P', 'mm', 'A4')
    pdf_w = 210
    margin = 10
    processed_frames = 0
    
    for frame in frame_generator:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_f:
            temp_img_path = temp_f.name
        
        try:
            is_success = cv2.imwrite(temp_img_path, frame)
            if not is_success:
                continue

            with Image.open(temp_img_path) as pil_img:
                img_w, img_h = pil_img.size
            
            aspect_ratio = img_w / img_h
            display_w = pdf_w - 2 * margin
            display_h = display_w / aspect_ratio

            pdf.add_page()
            pdf.image(temp_img_path, x=margin, y=margin, w=display_w, h=display_h)
            processed_frames += 1
        finally:
            if os.path.exists(temp_img_path):
                os.remove(temp_img_path)

    if processed_frames > 0:
        pdf.output(output_pdf)
        return True
    return False

# --- Background Task Definition ---

def create_pdf_task(youtube_url, job_id):
    """This function runs in the background thread."""
    temp_base = '/dev/shm' if os.path.exists('/dev/shm') else tempfile.gettempdir()
    temp_dir = tempfile.mkdtemp(dir=temp_base)
    
    with jobs_lock:
        jobs[job_id]['status'] = 'processing'
    
    try:
        video_path, video_title = download_video(youtube_url, temp_dir, job_id)
        if not video_path:
            raise Exception("Failed to download video.")
            
        frame_gen = frames_to_pdf_generator(video_path, job_id, sampling_rate_fps=1)
        
        safe_filename = re.sub(r'[^\w\s-]', '', video_title).strip()
        safe_filename = re.sub(r'[-\s]+', '-', safe_filename) + ".pdf"
        if not safe_filename: safe_filename = "video-slides.pdf"

        output_pdf = os.path.join(temp_dir, "output.pdf")
        
        success = save_frames_to_pdf(frame_gen, output_pdf)
        if not success:
            raise Exception("No unique frames found to create PDF.")

        with jobs_lock:
            if job_id in jobs:
                jobs[job_id]['status'] = 'complete'
                jobs[job_id]['filepath'] = output_pdf
                jobs[job_id]['temp_dir'] = temp_dir
                jobs[job_id]['filename'] = safe_filename
        print(f"[{job_id}] Job complete.")

    except Exception as e:
        print(f"[{job_id}] Job failed: {e}")
        with jobs_lock:
            if job_id in jobs:
                jobs[job_id]['status'] = 'failed'
        shutil.rmtree(temp_dir, ignore_errors=True)


# --- Flask Routes ---

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/convert", methods=["POST"])
def convert():
    youtube_url = request.form["youtube_url"]
    job_id = str(uuid.uuid4())
    
    with jobs_lock:
        jobs[job_id] = {'status': 'pending', 'progress': 0, 'stage': 'Initializing'}
    
    thread = threading.Thread(target=create_pdf_task, args=(youtube_url, job_id))
    thread.start()
    
    return jsonify({'job_id': job_id})

@app.route("/status/<job_id>")
def status(job_id):
    with jobs_lock:
        job = jobs.get(job_id, {})
    if not job:
        return jsonify({'status': 'not_found'}), 404
    return jsonify({
        'status': job.get('status'),
        'progress': job.get('progress'),
        'stage': job.get('stage'),
        'filename': job.get('filename')
    })

@app.route("/download/<job_id>")
def download(job_id):
    with jobs_lock:
        job = jobs.get(job_id, None)
    
    if job is None or job.get('status') != 'complete':
        return "Job not found or not complete.", 404
    
    filepath = job['filepath']
    temp_dir = job['temp_dir']
    filename = job.get('filename', 'slides.pdf')

    @after_this_request
    def cleanup(response):
        try:
            shutil.rmtree(temp_dir)
            with jobs_lock:
                if job_id in jobs:
                    del jobs[job_id]
            print(f"Cleaned up job {job_id}")
        except Exception as e:
            print(f"Error during cleanup for job {job_id}: {e}")
        return response

    return send_file(filepath, as_attachment=True, download_name=filename)

if __name__ == "__main__":
    app.run(debug=True)
# --- End of app.py ---
